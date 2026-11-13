"""Regression tests for the ML logic review: training, evaluation and custody bugs."""
import json
from pathlib import Path

import numpy as np
import pytest


# ── Entity resolver: TF-IDF feature must be live during training ──────────────

def test_resolver_cosine_works_before_any_fit():
    from KnowledgeGraphApp.GraphStorage.EntityResolver import _TFIDFCache

    cache = _TFIDFCache()
    assert cache.cosine("Acme Robotics", "Acme Robotics Ltd") > 0.5
    assert cache.cosine("Acme Robotics", "Zephyr Bank") < 0.3
    assert cache.cosine("", "") == 0.0


def test_resolver_training_uses_the_tfidf_feature(tmp_path, monkeypatch):
    from KnowledgeGraphApp.GraphStorage import EntityResolver as module

    monkeypatch.setattr(module, "MODEL_PATH", tmp_path / "EntityResolver.joblib")
    monkeypatch.setattr(module, "_tfidf", module._TFIDFCache())  # a fresh process
    same = [("Acme Robotics", "Acme Robotics Ltd"), ("Zephyr Bank", "Zephyr Bank Inc"),
            ("Northwind Traders", "Northwind"), ("Blue Harbor Capital", "Blue Harbor"),
            ("Orion Systems", "Orion Systems Corp"), ("Delta Freight", "Delta Freight LLC")]
    different = [("Acme Robotics", "Zephyr Bank"), ("Northwind Traders", "Orion Systems"),
                 ("Blue Harbor Capital", "Delta Freight"), ("Zephyr Bank", "Orion Systems"),
                 ("Acme Robotics", "Delta Freight"), ("Northwind Traders", "Blue Harbor Capital")]
    gold = tmp_path / "ResolutionGold.jsonl"
    gold.write_text("\n".join(
        json.dumps({"name_a": a, "name_b": b, "label": label})
        for label, pairs in ((1, same), (0, different)) for a, b in pairs), encoding="utf-8")

    module.EntityResolver.train(str(gold))

    import joblib
    scaler = joblib.load(tmp_path / "EntityResolver.joblib").named_steps["scaler"]
    # Feature 1 is the TF-IDF cosine; it was constant 0.0 (variance 0) before the fix.
    assert scaler.var_[1] > 0


# ── Learned relations: threshold tuning ───────────────────────────────────────

def test_tune_threshold_counts_exactly_three_true_positives():
    from Tools.TrainLearned import tune_threshold

    # 47 positives: (3 / 47) * 47 is 2.9999999999999996 in floating point.
    labels = np.array([1] * 47 + [0] * 10)
    scores = np.array([0.9, 0.8, 0.7] + [0.01] * 44 + [0.02] * 10)
    assert tune_threshold(labels, scores, total_gold=47) == pytest.approx(0.7)


# ── Cluster metrics ───────────────────────────────────────────────────────────

def test_bcubed_reports_mention_count_for_weighting():
    from KnowledgeGraphApp.Extraction.MentionResolver import bcubed_metrics

    result = bcubed_metrics({"g1": ["a", "b"], "g2": ["c"]}, {"a": "p1", "b": "p1", "c": "p2"})
    assert result["mentions"] == 3 and result["precision"] == 1.0
    assert bcubed_metrics({}, {})["mentions"] == 0


# ── Split custody ─────────────────────────────────────────────────────────────

def test_custody_uses_manifest_next_to_the_data_file(tmp_path):
    from KnowledgeGraphApp.Extraction.SplitCustody import sibling_manifest, verify_split_custody

    (tmp_path / "Manifest.json").write_text(json.dumps({"assignments": [
        {"document_id": "doc-dev", "split": "regression_dev", "group_key": "t"}]}), encoding="utf-8")
    data_file = tmp_path / "train.jsonl"
    data_file.write_text("", encoding="utf-8")
    record = {"document_id": "doc-dev", "source": "dev"}  # no "split" field, as PrepareV2 writes

    with pytest.raises(ValueError, match="Authoritative split mismatch"):
        verify_split_custody([record], expected_split="train", manifest_path=sibling_manifest(data_file))


def test_custody_rejects_a_missing_explicit_manifest(tmp_path):
    from KnowledgeGraphApp.Extraction.SplitCustody import verify_split_custody

    with pytest.raises(FileNotFoundError, match="Split manifest not found"):
        verify_split_custody([{"document_id": "d"}], expected_split="train",
                             manifest_path=tmp_path / "Missing.json")


# ── Artifact guard ────────────────────────────────────────────────────────────

def test_artifact_guard_rejects_thresholds_without_model_hash(tmp_path):
    from KnowledgeGraphApp.Extraction.ArtifactGuard import file_sha256, validate_model_artifacts

    (tmp_path / "Model.pt").write_bytes(b"weights")
    (tmp_path / "Vocabulary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "TrainingManifest.json").write_text("{}", encoding="utf-8")
    thresholds = {"target_precision": 0.7, "relations": {
        "FOUNDED": {"threshold": 0.5, "disabled": False, "status": "calibrated", "precision": 0.9}}}
    (tmp_path / "Thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")

    with pytest.raises(ValueError, match="records no model_sha256"):
        validate_model_artifacts(tmp_path)

    thresholds["model_sha256"] = file_sha256(tmp_path / "Model.pt")
    (tmp_path / "Thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")
    assert validate_model_artifacts(tmp_path).is_valid


# ── Contextual model: training and evaluation (needs torch) ───────────────────

def _document(document_id, split, target, text="Acme Corp founded Beta Inc in Pune."):
    from KnowledgeGraphApp.Extraction.ContextualRelations import RELATION_TYPES

    index = RELATION_TYPES.index("FOUNDED")
    targets = [None] * len(RELATION_TYPES)
    targets[index] = target
    return {
        "document_id": document_id, "split": split, "group_key": document_id, "text": text,
        "mentions": [
            {"id": f"{document_id}:m1", "document_id": document_id, "text": "Acme Corp", "type": "ORG",
             "start": text.index("Acme"), "end": text.index("Acme") + 9},
            {"id": f"{document_id}:m2", "document_id": document_id, "text": "Beta Inc", "type": "ORG",
             "start": text.index("Beta"), "end": text.index("Beta") + 8},
        ],
        "annotated_groups": {"g1": [f"{document_id}:m1"], "g2": [f"{document_id}:m2"]},
        "candidates": [{"subject_group": "g1", "object_group": "g2", "targets": targets,
                        "loss_mask": [value is not None for value in targets]}],
    }


def _write_jsonl(path: Path, documents):
    path.write_text("\n".join(json.dumps(d) for d in documents), encoding="utf-8")
    return path


def test_training_refuses_positive_only_supervision(tmp_path):
    pytest.importorskip("torch")
    from Tools.TrainContextual import train_contextual_relations

    train = _write_jsonl(tmp_path / "train.jsonl", [_document(f"t{i}", "train", 1) for i in range(4)])
    val = _write_jsonl(tmp_path / "validation.jsonl", [_document("v0", "validation", 1)])

    with pytest.raises(ValueError, match="no explicit negative label"):
        train_contextual_relations([train], val, tmp_path / "out", epochs=1)

    manifest = train_contextual_relations([train], val, tmp_path / "out", epochs=1, allow_positive_only=True)
    assert manifest["positive_only_relations"] == ["FOUNDED"]
    # Precision is 1.0 by construction here, so F1 must not be the selection metric.
    assert manifest["selection_metric"] == "negative_val_loss_no_validation_negatives"


def test_training_with_negatives_selects_on_f1_and_records_shapes(tmp_path):
    pytest.importorskip("torch")
    from Tools.TrainContextual import train_contextual_relations

    train = _write_jsonl(tmp_path / "train.jsonl",
                         [_document(f"t{i}", "train", i % 2) for i in range(6)])
    val = _write_jsonl(tmp_path / "validation.jsonl",
                       [_document("v0", "validation", 1), _document("v1", "validation", 0)])

    manifest = train_contextual_relations([train], val, tmp_path / "out", epochs=1)
    assert manifest["positive_only_relations"] == []
    assert manifest["selection_metric"] == "micro_f1_over_relations_with_validation_negatives"
    assert (manifest["embed_dim"], manifest["num_layers"], manifest["max_tokens"]) == (128, 2, 128)


def test_training_skips_pairs_whose_markers_are_truncated():
    pytest.importorskip("torch")
    from Tools.TrainContextual import extract_samples_from_records, markers_fit

    far_apart = "Acme Corp " + "filler " * 200 + "founded Beta Inc in Pune."
    stats = {}
    samples = extract_samples_from_records(
        [_document("near", "train", 1), _document("far", "train", 1, text=far_apart)], stats=stats)

    assert len(samples) == 1 and stats["skipped_truncated"] == 1
    assert all(markers_fit(sample["marked_text"]) for sample in samples)


def test_wiring_test_reports_whether_loss_fell(tmp_path):
    pytest.importorskip("torch")
    from Tools.TrainContextual import train_contextual_relations

    train = _write_jsonl(tmp_path / "train.jsonl",
                         [_document(f"t{i}", "train", i % 2) for i in range(6)])
    val = _write_jsonl(tmp_path / "validation.jsonl", [_document("v0", "validation", 1)])

    result = train_contextual_relations([train], val, tmp_path / "out", wiring_test_only=True)
    assert result["final_loss"] < result["initial_loss"]
    assert result["wiring_test_passed"] is True


def test_validation_loss_is_averaged_per_supervised_label():
    torch = pytest.importorskip("torch")
    from KnowledgeGraphApp.Extraction.ContextualRelations import RELATION_TYPES
    from Tools.TrainContextual import evaluate_model

    relations = len(RELATION_TYPES)

    class FixedLogits(torch.nn.Module):
        def forward(self, input_ids, marker_positions):
            return {"relation_logits": torch.zeros(input_ids.shape[0], relations)}  # p = 0.5

    def batch(rows, supervised):
        mask = torch.zeros(rows, relations, dtype=torch.bool)
        mask[:, :supervised] = True
        return {"input_ids": torch.ones(rows, 4, dtype=torch.long),
                "marker_positions": torch.zeros(rows, 4, dtype=torch.long),
                "targets": torch.ones(rows, relations), "loss_mask": mask}

    # Every supervised label has loss ln 2, whatever the batch sizes are.
    metrics = evaluate_model(FixedLogits(), [batch(8, 3), batch(1, 1)], device="cpu")
    assert metrics["val_loss"] == pytest.approx(0.6931, abs=1e-3)
    # All labels are positive, so no relation is eligible for F1-based selection.
    assert metrics["selection"]["relations"] == []


def test_release_gate_audit_accepts_string_provenance():
    pytest.importorskip("torch")
    from Tools.EvaluateV2 import audit_acceptance_gates

    results = {"metrics": {"precision": 0.5, "recall": 0.2},
               "bootstrap_95ci": {"precision": [0.4, 0.6]}}
    records = [
        {"annotation_provenance": "upstream Re-DocRED; no new human finance review",
         "gold_relations": [["0", "ACQUIRED", "1"]]},
        {"annotation_provenance": {"reviewer": "analyst_a"}, "gold_relations": [["0", "ACQUIRED", "1"]]},
    ]
    assert audit_acceptance_gates(results, records)["finance_support"] == 1


def test_production_mode_evaluation_runs_and_dedupes_unmatched_facts():
    pytest.importorskip("torch")
    from KnowledgeGraphApp.Extraction.EntityRecognizer import NERRunner
    from KnowledgeGraphApp.Extraction.MentionResolver import NonDestructiveResolver
    from KnowledgeGraphApp.Ingestion.Preprocessor import get_nlp
    from Tools.EvaluateV2 import evaluate_document_production

    text = "Elon Musk founded Tesla in California."
    document = {"document_id": "doc1", "text": text, "gold_relations": [], "annotated_groups": {},
                "mentions": []}

    class Claim:
        relation = "FOUNDED"

        def __init__(self, subject, obj):
            self.subject_mention_id, self.object_mention_id = subject, obj

    class RepeatingExtractor:
        """Emits the same unmatched fact three times, as three mention pairs would."""
        def extract_from_document(self, document_id, text, mentions):
            assert len(mentions) >= 2  # NERRunner.run produced mentions
            return [Claim(mentions[0].id, mentions[1].id)] * 3

    result = evaluate_document_production(
        document, RepeatingExtractor(), NonDestructiveResolver(), NERRunner(), get_nlp())
    assert result["fp"] == 1 and result["pred_count"] == 1
