import json

import pytest

from app.models.detection_log import Prediction
from app.pipeline.bert_email import BertEmailModel, IdentityVectorizer
from app.pipeline.classifiers import TrainedClassifier
from app.pipeline.feature_extraction_email import extract_email_features
from ml.training.experiment_identity import bind_experiment


def test_checkpoint_identity_rejects_changed_dataset_and_unidentified_files(tmp_path):
    run = tmp_path / "run"
    identity = {"dataset": "first", "epochs": 3}
    assert bind_experiment(run, identity) is False
    (run / "checkpoints/checkpoint-10").mkdir(parents=True)
    assert bind_experiment(run, identity) is True
    with pytest.raises(RuntimeError, match="identity changed"):
        bind_experiment(run, {**identity, "dataset": "second"})
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    (orphan / "checkpoint-10").mkdir()
    with pytest.raises(RuntimeError, match="Unidentified"):
        bind_experiment(orphan, identity)


def test_bert_receives_natural_redacted_body_and_participates_in_fusion():
    class Model:
        text_field = "bert_text"

        def predict_proba(self, texts):
            assert texts == ["Please VERIFY  emailaddresstoken  at  urltoken "]
            return [[0.1, 0.9]]

    features = extract_email_features("Subject: Hidden header\n\nPlease VERIFY me@example.com at https://example.com")
    classifier = TrainedClassifier(email_model=Model(), email_vectorizer=IdentityVectorizer())
    prediction, confidence = classifier.predict(None, features)
    assert prediction == Prediction.phishing
    assert confidence == .9
    assert classifier.predict({"is_ip_address": True, "has_at_symbol": True,
                               "suspicious_tld": True, "brand_keyword_outside_domain": True},
                              features)[0] == Prediction.phishing


def test_bert_refuses_incomplete_or_unknown_label_exports(tmp_path):
    with pytest.raises(RuntimeError, match="completed BERT export"):
        BertEmailModel(tmp_path)
    (tmp_path / "serving.json").write_text(json.dumps({"labels": ["phishing", "legitimate"]}))
    with pytest.raises(RuntimeError, match="legitimate=0"):
        BertEmailModel(tmp_path)
