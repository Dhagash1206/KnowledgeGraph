"""Regression tests for the hyperparameter-report fixes: data size, duplicate
features, honest held-out reporting, negatives and early stopping."""
import json
from pathlib import Path
import random

import numpy as np
import pytest


# ── Entity resolver ───────────────────────────────────────────────────────────

def test_resolver_features_have_no_duplicate_column():
    from KnowledgeGraphApp.GraphStorage.EntityResolver import BASE_FEATURE_NAMES, _base_features

    pairs = [("International Business Machines", "IBM"), ("Tesla Motors", "Tesla"), ("Apple", "Google"),
             ("Elon Musk", "Elon R. Musk"), ("New York", "New York City"), ("Meta", "Microsoft"),
             ("Bank of Pune", "Bank of Delhi"), ("J. Rao", "Jaya Rao")]
    columns = np.array([_base_features(a, b) for a, b in pairs]).T
    assert len(columns) == len(BASE_FEATURE_NAMES)
    for first in range(len(columns)):
        for second in range(first + 1, len(columns)):
            assert not np.array_equal(columns[first], columns[second]), (
                BASE_FEATURE_NAMES[first], BASE_FEATURE_NAMES[second])


def test_resolver_feature_vector_layout():
    from KnowledgeGraphApp.GraphStorage.EntityResolver import (
        BASE_FEATURE_NAMES, FEATURE_NAMES, TYPED_BLOCKS, _feature_vector)

    width = len(BASE_FEATURE_NAMES)
    typed = _feature_vector("Tesla Motors", "Tesla", "ORG")
    untyped = _feature_vector("Tesla Motors", "Tesla", "PRODUCT")
    assert len(typed) == len(untyped) == len(FEATURE_NAMES)
    assert typed[:width] == untyped[:width]
    org = 1 + TYPED_BLOCKS.index("ORG")
    assert typed[org * width:(org + 1) * width] == typed[:width]
    # An unknown type is scored by the shared block alone.
    assert not any(untyped[width:])


def test_resolver_rejects_model_with_other_feature_count(tmp_path, monkeypatch, caplog):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from KnowledgeGraphApp.GraphStorage import EntityResolver as module

    stale = LogisticRegression().fit([[0.0] * 5, [1.0] * 5], [0, 1])
    joblib.dump(stale, tmp_path / "EntityResolver.joblib")
    monkeypatch.setattr(module, "MODEL_PATH", tmp_path / "EntityResolver.joblib")
    with caplog.at_level("WARNING"):
        resolver = module.EntityResolver()
    assert resolver._lr is None
    assert "retrain" in caplog.text


def test_shipped_resolver_matches_code_and_tuned_threshold():
    from KnowledgeGraphApp.GraphStorage.EntityResolver import FEATURE_NAMES, MODEL_PATH, EntityResolver
    from KnowledgeGraphApp.Extraction.EntityRecognizer import Entity
    from KnowledgeGraphApp.Settings import Settings

    report = json.loads(MODEL_PATH.with_suffix(".training.json").read_text(encoding="utf-8"))
    assert report["features"] == list(FEATURE_NAMES)
    assert report["training"]["pairs"] > 1000
    threshold = Settings.model_fields["resolution_threshold"].default
    assert threshold == report["threshold"]

    resolver = EntityResolver(threshold=threshold)
    assert resolver._lr is not None

    def merged(first, second, entity_type="ORG"):
        entities = [Entity(name, name, name, entity_type, 0, len(name), "test", "c:0") for name in (first, second)]
        mapping = resolver.resolve_within_doc(entities)
        return mapping[first] == mapping[second]

    assert merged("Microsoft Corporation", "Microsoft")
    assert merged("Alphabet Inc", "Alphabet")
    assert not merged("Google", "Microsoft")
    assert not merged("Elon Musk", "Jeff Bezos", "PERSON")


def test_resolver_merge_counts_include_transitive_and_unblocked_pairs():
    from KnowledgeGraphApp.Extraction.EntityRecognizer import Entity
    from Tools.TrainEntityResolver import merge_counts

    entities = [Entity(i, i, i, "ORG", 0, 1, "gold", "c") for i in ("a", "b", "c", "d")]
    clusters = {"a": 1, "b": 1, "c": 2, "d": 1}
    # a-b correct, b-c wrong; a-c is then wrong by transitivity; a-d and b-d never proposed.
    documents = [(entities, clusters, [("a", "b", 0.9), ("b", "c", 0.9)])]
    assert merge_counts(documents, 0.5) == {"tp": 1, "fp": 2, "fn": 2}
    assert merge_counts(documents, 0.95) == {"tp": 0, "fp": 0, "fn": 3}


# ── Relation scorer ───────────────────────────────────────────────────────────

def test_scorer_labels_mapped_relations_exactly_and_others_by_any_gold_link():
    from Tools.TrainRelationScorer import label_triple

    facts = {("1", "FOUNDED", "2")}
    related = {("1", "2"), ("2", "1"), ("3", "4"), ("4", "3")}
    assert label_triple("FOUNDED", "1", "2", facts, related) == 1
    assert label_triple("FOUNDED", "2", "1", facts, related) == 0   # wrong direction
    assert label_triple("OWNS", "1", "2", facts, related) == 0      # wrong type
    assert label_triple("RELATED_TO", "4", "3", facts, related) == 1
    assert label_triple("RELATED_TO", "1", "3", facts, related) == 0


def test_shipped_scorer_was_trained_on_real_triples():
    from KnowledgeGraphApp.Extraction.RelationScorer import MODEL_PATH, RelationScorer

    report_path = MODEL_PATH.with_suffix(".training.json")
    if not report_path.exists():
        pytest.skip("RelationScorer.joblib has not been retrained with Tools/TrainRelationScorer.py yet")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["training"]["rows"] > 20 * report["feature_count"]
    assert RelationScorer()._model is not None


# ── Learned relations: held-out threshold scores ──────────────────────────────

def test_held_out_metrics_score_each_half_with_the_other_halfs_threshold(monkeypatch):
    from Tools import TrainLearned

    relations = len(TrainLearned.RELATION_TYPES)
    document_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    labels = np.zeros((8, relations), dtype=np.int8)
    scores = np.zeros((8, relations))
    seen = []

    def fake_tune(half_labels, half_scores, total_gold):
        seen.append(len(half_labels))
        return 0.5

    monkeypatch.setattr(TrainLearned, "tune_threshold", fake_tune)
    monkeypatch.setattr(TrainLearned, "gold_facts", lambda record: record["facts"])
    first_relation = TrainLearned.RELATION_TYPES[0]
    development = [{"facts": {("a", first_relation, "b")}} for _ in range(4)]
    labels[[0, 2], 0] = 1
    scores[0, 0], scores[2, 0], scores[3, 0] = 0.9, 0.1, 0.8

    report = TrainLearned.held_out_threshold_metrics(development, labels, document_ids, scores)
    # Each tuning call sees only the four rows of the opposite half.
    assert seen == [4] * (2 * relations)
    # Four gold facts: one found, three missed (one of them never a candidate); one false alarm.
    assert report["per_relation"][first_relation] == {"tp": 1, "fp": 1, "fn": 3}


# ── Contextual v2: negatives ──────────────────────────────────────────────────

def _ontology():
    pairs = {"FOUNDED": [["PERSON", "ORG"]], "OWNS": [["ORG", "ORG"]], "ACQUIRED": [["ORG", "ORG"]]}
    return {"relations": {name: {"entity_type_pairs": value} for name, value in pairs.items()}}


def test_closed_world_negatives_never_contradict_gold_or_touch_finance_relations():
    from Tools.PrepareV2Dataset import closed_world_negatives

    groups = {"0": [], "1": [], "2": []}
    typed = {"0": {"PERSON"}, "1": {"ORG"}, "2": {"ORG"}}
    positives = {("0", "FOUNDED", "1"), ("1", "OWNS", "2")}
    negatives = closed_world_negatives(groups, typed, positives, _ontology(), random.Random(0), sample=False)

    assert not negatives & positives
    assert ("0", "FOUNDED", "2") in negatives and ("2", "OWNS", "1") in negatives
    # ACQUIRED has no Re-DocRED annotation, so its absence stays unknown.
    assert all(relation != "ACQUIRED" for _, relation, _ in negatives)
    # Type-incompatible pairs are not candidates at inference either.
    assert ("1", "FOUNDED", "0") not in negatives


def test_closed_world_negative_sampling_caps_pairs_per_document():
    from Tools.PrepareV2Dataset import closed_world_negatives

    groups = {str(i): [] for i in range(12)}
    typed = {key: {"ORG"} for key in groups}
    args = (groups, typed, set(), _ontology())
    everything = closed_world_negatives(*args, random.Random(0), sample=False)
    sampled = closed_world_negatives(*args, random.Random(0), sample=True, per_positive=8, minimum=5)
    assert len(everything) == 12 * 11 and len(sampled) == 5
    assert sampled <= everything


# ── Contextual v2: early stopping (needs torch) ───────────────────────────────

def _document(document_id, split, target, text="Acme Corp founded Beta Inc in Pune."):
    from KnowledgeGraphApp.Extraction.ContextualRelations import RELATION_TYPES

    targets = [None] * len(RELATION_TYPES)
    targets[RELATION_TYPES.index("FOUNDED")] = target
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


def test_training_stops_early_and_halves_the_learning_rate(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from Tools import TrainContextual

    train = _write_jsonl(tmp_path / "train.jsonl", [_document(f"t{i}", "train", i % 2) for i in range(6)])
    val = _write_jsonl(tmp_path / "validation.jsonl",
                       [_document("v0", "validation", 1), _document("v1", "validation", 0)])
    # Validation gets worse every epoch after the first, as in the shipped run.
    losses = iter([0.20, 0.25, 0.30, 0.35, 0.40, 0.45])

    def worsening(model, loader, device, threshold=0.5):
        return {"val_loss": next(losses), "selection": {"relations": ["FOUNDED"], "f1": 0.5},
                "micro": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "tp": 1, "fp": 1, "fn": 1},
                "per_relation": {}}

    monkeypatch.setattr(TrainContextual, "evaluate_model", worsening)
    manifest = TrainContextual.train_contextual_relations([train], val, tmp_path / "out", epochs=6, patience=2)

    assert manifest["stopped_early"] is True
    assert manifest["best_epoch"] == 1 and manifest["epochs_run"] == 3
    rates = [entry["lr"] for entry in manifest["training_history"]]
    assert rates[0] == pytest.approx(1e-3) and rates[2] == pytest.approx(5e-4)


def test_patience_zero_runs_every_epoch(tmp_path):
    pytest.importorskip("torch")
    from Tools.TrainContextual import train_contextual_relations

    train = _write_jsonl(tmp_path / "train.jsonl", [_document(f"t{i}", "train", i % 2) for i in range(6)])
    val = _write_jsonl(tmp_path / "validation.jsonl",
                       [_document("v0", "validation", 1), _document("v1", "validation", 0)])
    manifest = train_contextual_relations([train], val, tmp_path / "out", epochs=3, patience=0)
    assert manifest["epochs_run"] == 3 and manifest["stopped_early"] is False
