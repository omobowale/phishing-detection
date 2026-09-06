"""Local-only inference for a completed, explicitly selected BERT experiment."""
import json
from hashlib import sha256
from pathlib import Path
from threading import Lock


def directory_hash(path: Path) -> str:
    digest = sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file():
            digest.update(file.relative_to(path).as_posix().encode())
            digest.update(b"\0")
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


class IdentityVectorizer:
    def transform(self, texts):
        return list(texts)


class BertEmailModel:
    text_field = "bert_text"

    def __init__(self, path: Path):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        metadata_path = path / "serving.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"No completed BERT export at {path}; run train_email_bert first")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("labels") != ["legitimate", "phishing"]:
            raise RuntimeError("BERT export must declare legitimate=0, phishing=1")
        if metadata.get("preprocessing") != "strip_headers_redact_v1":
            raise RuntimeError("Unsupported BERT preprocessing contract")
        self.max_length = metadata["max_length"]
        if not isinstance(self.max_length, int) or self.max_length < 1:
            raise RuntimeError("Invalid BERT maximum sequence length")
        before = directory_hash(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        if self.model.config.num_labels != 2:
            raise RuntimeError("BERT export must have exactly two labels")
        self.model.to("cpu").eval()
        self.artifact_sha256 = directory_hash(path)
        if before != self.artifact_sha256:
            raise RuntimeError("BERT files changed during loading")
        # Bound simultaneous large tensor allocations on the CPU server.
        self._lock = Lock()

    def predict_proba(self, texts):
        import torch

        with self._lock, torch.inference_mode():
            encoded = self.tokenizer(list(texts), padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
            return torch.softmax(self.model(**encoded).logits, dim=-1).cpu().numpy()

    def predict(self, texts):
        return self.predict_proba(texts).argmax(axis=1)
