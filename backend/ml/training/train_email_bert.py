"""BERT/DistilBERT fine-tuning for the email/NLP classifier (spec section 3's
"BERT embeddings for transformer model" -- the classical TF-IDF+RandomForest
model in train_email_classifier.py already meets every spec target on its
own, F1 0.980; this is the second half of the spec's stated architecture, a
completeness item, not a metrics necessity).

CPU-only fine-tune (no CUDA GPU on this machine -- torch.cuda.is_available()
is False; see README_EMAIL.md). Real measured cost on this machine: ~1.63s
per training step at batch size 8 / max_length 256, i.e. roughly 1.75 hours
per epoch over the 30,857-row train split -- a multi-hour background job for
the 3-epoch run below, not a quick script.

Resumable by construction: TrainingArguments checkpoints after every epoch
under ml/saved_models/bert_runs/full/checkpoints/, and this script
auto-resumes from the latest checkpoint found there if one exists -- an
interrupted run (sleep, shutdown, closed terminal) loses at most the current
in-progress epoch, not the whole run. Just re-run the same command to
continue.

Exports to ml/saved_models/bert_runs/full/model/ -- the exact path
app/pipeline/bert_email.py's BertEmailModel (email_classifier_backend="bert",
see app/core/config.py's email_bert_model_path) expects, alongside a
serving.json contract file declaring the label order, preprocessing version,
and max_length that loader checks before trusting this export.

Usage:
    python -m ml.training.train_email_bert
"""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.pipeline.preprocessing import redact_email_and_urls, strip_email_headers  # noqa: E402
from ml.training.experiment_identity import bind_experiment, file_hash  # noqa: E402
from ml.training.train_email_classifier import (  # noqa: E402
    SANITY_LEGITIMATE,
    SANITY_PHISHING,
    TARGET_F1,
    TARGET_PRECISION,
    TARGET_RECALL,
    _baseline_metrics,
)

PROCESSED_DIR = BACKEND_ROOT / "ml" / "data" / "processed_email"
SAVED_MODELS_DIR = BACKEND_ROOT / "ml" / "saved_models"
CHECKPOINT_DIR = SAVED_MODELS_DIR / "bert_checkpoints"
FINAL_MODEL_DIR = SAVED_MODELS_DIR / "bert_email_classifier"

MODEL_NAME = "distilbert-base-uncased"
# 256 was the original choice, but real emails average 212 tokens (median hits
# the cap: 62% of a 200-row sample truncated at exactly 256) -- a from-scratch
# CPU timing benchmark using a ~20-token dummy sentence badly underestimated
# real per-step cost as a result (measured ~13s/step on real data vs. ~1.6s/step
# on the dummy, an 8x gap, not CPU contention -- confirmed by measuring real
# tokenized lengths directly). 128 keeps most of a typical email's content
# (subject + opening lines, where most phishing signal concentrates) while
# roughly halving compute, given attention cost grows faster than linearly
# with sequence length.
MAX_LENGTH = 128
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
RANDOM_STATE = 42


def _clean_text(raw_text: str) -> str:
    """Header-stripped, redacted, but NOT lowercased/lemmatized/stopword-removed
    -- unlike the classical TF-IDF pipeline, BERT's own subword tokenizer wants
    natural casing and punctuation, so this stops one step earlier than
    preprocess_email_text(). Still applies the same strip_email_headers() +
    redact_email_and_urls() fixes as the classical pipeline and live serving,
    so this model doesn't reintroduce the header-noise or jose@monkey.org-style
    leakage issues documented in README_EMAIL.md.

    Matches feature_extraction_email.py's "bert_text" field exactly (same two
    calls, same order) -- whichever wires this model into live serving later
    should feed it that field directly rather than recomputing this."""
    return redact_email_and_urls(strip_email_headers(raw_text))


class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class WeightedTrainer(Trainer):
    """Same class_weight="balanced" idea train_email_classifier.py uses for its
    classical models, applied to BERT's loss -- the ~11% positive rate would
    otherwise let the model coast on predicting the majority class."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "average_precision": average_precision_score(labels, probs),
    }


def _sanity_check(model, tokenizer, device) -> dict:
    def _predict(text: str) -> str:
        clean = _clean_text(text)
        inputs = tokenizer(clean, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        return "phishing" if logits.argmax(dim=-1).item() == 1 else "legitimate"

    phishing_correct = sum(1 for t in SANITY_PHISHING if _predict(t) == "phishing")
    legit_correct = sum(1 for t in SANITY_LEGITIMATE if _predict(t) == "legitimate")
    return {
        "phishing_pass": f"{phishing_correct}/{len(SANITY_PHISHING)}",
        "legitimate_pass": f"{legit_correct}/{len(SANITY_LEGITIMATE)}",
        "all_passed": phishing_correct == len(SANITY_PHISHING) and legit_correct == len(SANITY_LEGITIMATE),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or resume an identified BERT experiment")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="40/16/16 examples, one epoch; separate artifacts")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    epochs = 1 if args.smoke else args.epochs
    run_dir = args.run_dir or SAVED_MODELS_DIR / "bert_runs" / ("smoke" if args.smoke else "full")
    checkpoint_dir = run_dir / "checkpoints"
    final_model_dir = run_dir / "model"
    features_path = PROCESSED_DIR / "email_features.parquet"
    if not features_path.exists():
        print(f"No feature matrix at {features_path}. Run: python -m ml.training.build_email_features")
        sys.exit(1)

    base_config = AutoConfig.from_pretrained(MODEL_NAME, local_files_only=args.local_files_only)
    revision = getattr(base_config, "_commit_hash", None)
    identity = {
        "dataset_sha256": file_hash(features_path), "model_name": MODEL_NAME,
        "pretrained_revision": revision,
        "smoke": args.smoke, "epochs": epochs, "seed": RANDOM_STATE,
        "max_length": MAX_LENGTH, "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "code_sha256": {p.name: file_hash(p) for p in (
            Path(__file__), BACKEND_ROOT / "app/pipeline/preprocessing.py",
            BACKEND_ROOT / "ml/training/experiment_identity.py")},
        "packages": {name: version(name) for name in ("torch", "transformers", "accelerate", "numpy", "pandas")},
    }
    resume_from_checkpoint = bind_experiment(run_dir, identity)
    if (run_dir / "metrics.json").exists():
        raise RuntimeError("This experiment is complete; use a new --run-dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} (CPU-only fine-tune expected on this machine)")

    df = pd.read_parquet(features_path)
    if not set(df.label).issubset({"legitimate", "phishing"}):
        raise RuntimeError("Unrecognized email labels")
    if df.groupby("tokens")["split"].nunique().gt(1).any():
        raise RuntimeError("Identical token text crosses splits; rebuild the email dataset")
    if df.groupby("tokens")["label"].nunique().gt(1).any():
        raise RuntimeError("Conflicting token labels; rebuild the email dataset")
    if args.smoke:
        # Include both labels; ordered head() previously yielded one-class checks.
        df = pd.concat([
            df[(df.split == split) & (df.label == label)].sample(n=count // 2, random_state=RANDOM_STATE)
            for split, count in (("train", 40), ("val", 16), ("test", 16))
            for label in ("legitimate", "phishing")
        ])
    print("Cleaning text (strip_email_headers + redact_email_and_urls, natural casing preserved for BERT) ...")
    df["clean_text"] = df["text"].apply(_clean_text)
    df["label_id"] = (df["label"] == "phishing").astype(int)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    if any(set(part.label_id) != {0, 1} for part in (train_df, val_df, test_df)):
        raise RuntimeError("Each split must contain both classes")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Must happen before from_pretrained(): the classification head (absent from
    # the pretrained checkpoint, see the MISSING/UNEXPECTED key report at load
    # time) gets randomly initialized right there, and TrainingArguments(seed=...)
    # only takes effect once the Trainer is constructed -- too late to make that
    # initialization reproducible.
    set_seed(RANDOM_STATE)
    print(f"Loading tokenizer/model ({MODEL_NAME}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=revision, local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, revision=revision, local_files_only=args.local_files_only,
        id2label={0: "legitimate", 1: "phishing"}, label2id={"legitimate": 0, "phishing": 1},
    ).to(device)

    train_dataset = EmailDataset(train_df["clean_text"], train_df["label_id"], tokenizer, MAX_LENGTH)
    val_dataset = EmailDataset(val_df["clean_text"], val_df["label_id"], tokenizer, MAX_LENGTH)
    test_dataset = EmailDataset(test_df["clean_text"], test_df["label_id"], tokenizer, MAX_LENGTH)

    class_counts = train_df["label_id"].value_counts().sort_index()
    class_weights = torch.tensor(
        len(train_df) / (2 * class_counts.values), dtype=torch.float
    )
    print(f"Class weights (legitimate, phishing): {class_weights.tolist()}")

    if resume_from_checkpoint:
        print(f"Verified compatible checkpoint(s) under {checkpoint_dir}; resuming.")

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        # Step-based, not epoch-based: this machine has only ~7.8GB RAM shared
        # with everything else running on it, and a real run OOM-killed itself
        # at step 2191/11247 with zero checkpoints saved (epoch boundaries are
        # 3749 steps apart) -- 4.6 hours of compute lost to nothing. 500 steps
        # is ~40-60 minutes at this pace, bounding future loss to that instead.
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
        seed=RANDOM_STATE,
        dataloader_num_workers=0,
        use_cpu=(device.type == "cpu"),
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
        class_weights=class_weights,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    print(f"Training {epochs} epochs; smoke={args.smoke} ...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint or None)

    print("Evaluating on held-out test split ...")
    test_predictions = trainer.predict(test_dataset)
    test_logits = test_predictions.predictions
    test_labels = test_predictions.label_ids
    test_probs = torch.softmax(torch.tensor(test_logits), dim=-1)[:, 1].numpy()
    test_preds = np.argmax(test_logits, axis=-1)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds, labels=[0, 1]).ravel()
    test_metrics = {
        "accuracy": accuracy_score(test_labels, test_preds),
        "precision": precision_score(test_labels, test_preds, zero_division=0),
        "recall": recall_score(test_labels, test_preds, zero_division=0),
        "f1": f1_score(test_labels, test_preds, zero_division=0),
        "average_precision": average_precision_score(test_labels, test_probs),
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
    }
    baseline_metrics = _baseline_metrics(train_df["label_id"], test_df["label_id"])
    sanity = _sanity_check(model, tokenizer, device)

    print(f"Test metrics: {test_metrics}")
    print(f"Baseline (always predict '{baseline_metrics['majority_class']}'): {baseline_metrics}")
    print(f"Sanity check: {sanity}")

    meets_targets = (
        test_metrics["f1"] >= TARGET_F1
        and test_metrics["precision"] >= TARGET_PRECISION
        and test_metrics["recall"] >= TARGET_RECALL
    )
    print(
        f"Meets spec-wide targets (F1>={TARGET_F1}, precision>={TARGET_PRECISION}, "
        f"recall>={TARGET_RECALL}): {meets_targets}"
    )

    if identity["dataset_sha256"] != file_hash(features_path):
        raise RuntimeError("Dataset changed during training; refusing final export")
    for path in (Path(__file__), BACKEND_ROOT / "app/pipeline/preprocessing.py",
                 BACKEND_ROOT / "ml/training/experiment_identity.py"):
        if identity["code_sha256"][path.name] != file_hash(path):
            raise RuntimeError("Training/preprocessing code changed; refusing final export")
    final_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    (final_model_dir / "serving.json").write_text(json.dumps({
        "labels": ["legitimate", "phishing"], "max_length": MAX_LENGTH,
        "preprocessing": "strip_headers_redact_v1", "smoke": args.smoke,
        "experiment": identity,
    }, indent=2), encoding="utf-8")
    print(f"Saved fine-tuned model to {final_model_dir}")

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "max_length": MAX_LENGTH,
                "num_epochs": epochs,
                "smoke": args.smoke,
                "experiment": identity,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "test_metrics": test_metrics,
                "baseline_metrics": baseline_metrics,
                "meets_spec_targets": meets_targets,
                "sanity_check": sanity,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
                "positive_rate_train": float(train_df["label_id"].mean()),
                "eval_history": trainer.state.log_history,
            },
            indent=2,
        )
    )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
