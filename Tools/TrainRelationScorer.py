"""Train the relation confidence scorer on triples the rule extractor really produces.

The production pipeline runs over Re-DocRED documents. Every triple it extracts
whose two arguments align to exactly one gold entity each receives a label:

  * FOUNDED, EMPLOYED_BY, MEMBER_OF, OWNS, SUBSIDIARY_OF map to Re-DocRED
    relations, so the triple is correct only if that exact directed fact is gold.
  * Any other relation type has no Re-DocRED equivalent; the triple counts as
    correct if gold lists some relation between the two entities (either
    direction), and as wrong if gold lists none.

Triples with an unaligned or ambiguous argument are left out, not called wrong.
Weights are fitted on training documents; scores are reported on the
odd-indexed development documents. The test split is not read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Tools.EvaluateExternal import RELATIONS, gold_facts, reconstruct
from Tools.TrainLearned import exclude_overlap
from KnowledgeGraphApp import ExtractionPipeline
from KnowledgeGraphApp.Extraction import RelationScorer as scorer_module
from KnowledgeGraphApp.Extraction.PredicateNormalizer import PredicateNormalizer, load_relations_config
from KnowledgeGraphApp.Extraction.RelationScorer import FEATURE_VERSION, RelationScorer, _DummyTriple
from KnowledgeGraphApp.Settings import get_settings

MAPPED_RELATIONS = {relation for relation, _ in RELATIONS.values()}


def build_classifier() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
    ])


def gold_cluster_links(result, mentions) -> dict[str, set[str]]:
    """Resolved entity ID -> gold clusters whose typed span it matches exactly."""
    spans = defaultdict(set)
    for mention in mentions:
        if mention["type"]:
            spans[mention["start"], mention["end"], mention["type"]].add(str(mention["cluster"]))
    chunks = {chunk.chunk_id: chunk for chunk in result["chunks"]}
    links = defaultdict(set)
    for entity in result["all_entities"]:
        chunk = chunks.get(entity.chunk_id)
        if chunk is None:
            continue
        span = chunk.start_char + entity.start_char, chunk.start_char + entity.end_char, entity.type
        links[result["id_map"].get(entity.entity_id, entity.entity_id)].update(spans.get(span, set()))
    return links


def run_pipeline_when_free(raw_bytes: bytes, filename: str, attempts: int = 120):
    """Run the pipeline, waiting while another process (the app, a test run) holds its lock."""
    for attempt in range(attempts):
        try:
            return ExtractionPipeline.run_pipeline_nlp(raw_bytes, filename, "txt")
        except RuntimeError as error:
            if "already running" not in str(error) or attempt == attempts - 1:
                raise
            time.sleep(0.5)


def label_triple(relation_type, subject_cluster, object_cluster, facts, related_pairs) -> int:
    if relation_type in MAPPED_RELATIONS:
        return int((subject_cluster, relation_type, object_cluster) in facts)
    return int((subject_cluster, object_cluster) in related_pairs)


def collect_rows(records, name: str, cache: Path, normalizer, whitelist) -> list[dict]:
    """Run the pipeline on every record and return one labelled row per usable triple."""
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode())
    # Code and model identity: changed extraction or resolution must not reuse old rows.
    for path in sorted(Path("KnowledgeGraphApp").rglob("*.py")) + [Path("Configuration/RelationTypes.yaml"),
                                                                   Path("TrainedModels/EntityResolver.joblib")]:
        digest.update(path.as_posix().encode())
        digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    if cache.exists():
        saved = joblib.load(cache)
        if saved["fingerprint"] == fingerprint:
            return saved["rows"]

    recorded = []
    original_score = RelationScorer.score

    def recording_score(self, triple):
        recorded.append(triple)
        return original_score(self, triple)

    rows, skipped = [], Counter()
    RelationScorer.score = recording_score
    try:
        for index, record in enumerate(records):
            text, _, _, mentions = reconstruct(record)
            recorded.clear()
            try:
                result = run_pipeline_when_free(text.encode(), f"{name}-{index}.txt")
            except ValueError as error:
                if "English only" not in str(error):
                    raise
                skipped["non_english_documents"] += 1
                continue
            links = gold_cluster_links(result, mentions)
            facts = gold_facts(record)
            related_pairs = {pair for label in record["labels"]
                             for pair in ((str(label["h"]), str(label["t"])), (str(label["t"]), str(label["h"])))}
            for triple in recorded:
                normalized = normalizer.normalize(triple)
                if normalized is None:
                    continue
                clusters = [links.get(result["id_map"].get(entity.entity_id, entity.entity_id), set())
                            for entity in (normalized.subject, normalized.object)]
                if len(clusters[0]) != 1 or len(clusters[1]) != 1:
                    skipped["unaligned_argument"] += 1
                    continue
                subject_cluster, object_cluster = next(iter(clusters[0])), next(iter(clusters[1]))
                if subject_cluster == object_cluster:
                    skipped["same_gold_entity"] += 1
                    continue
                rows.append({
                    "document": index,
                    "relation_type": normalized.relation_type,
                    "features": scorer_module._feature_vector(triple, whitelist),
                    "label": label_triple(normalized.relation_type, subject_cluster, object_cluster,
                                          facts, related_pairs),
                    "record": {"predicate": triple.predicate, "source": triple.source,
                               "dep_distance": triple.dep_distance, "has_negation": triple.has_negation,
                               "subject_type": triple.subject.type, "object_type": triple.object.type},
                })
            if (index + 1) % 250 == 0:
                print(f"{name}: {index + 1}/{len(records)} documents, {len(rows)} labelled triples", flush=True)
    finally:
        RelationScorer.score = original_score
    print(f"{name}: {len(rows)} labelled triples; skipped {dict(skipped)}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"fingerprint": fingerprint, "rows": rows}, cache, compress=3)
    return rows


def matrix(rows) -> tuple[np.ndarray, np.ndarray]:
    return np.array([row["features"] for row in rows]), np.array([row["label"] for row in rows])


def evaluate(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, probabilities >= 0.5, average="binary", zero_division=0)
    return {"rows": int(len(labels)), "positive": int(labels.sum()),
            "precision_at_0.5": round(float(precision), 4), "recall_at_0.5": round(float(recall), 4),
            "f1_at_0.5": round(float(f1), 4),
            "roc_auc": round(float(roc_auc_score(labels, probabilities)), 4) if 0 < labels.sum() < len(labels) else None,
            "brier": round(float(brier_score_loss(labels, probabilities)), 4)}


def gate_report(probabilities, labels, relation_types, gate: float) -> dict:
    """What the RELATED_TO confidence gate keeps on held-out triples."""
    related = np.array([relation == "RELATED_TO" for relation in relation_types])
    kept = related & (probabilities >= gate)
    return {"gate": gate, "related_to_triples": int(related.sum()),
            "related_to_correct": int(labels[related].sum()), "kept": int(kept.sum()),
            "kept_correct": int(labels[kept].sum())}


def main(args):
    settings = get_settings()
    settings.relation_backend = "rules"
    config = load_relations_config(settings.relations_config_path)
    whitelist = scorer_module._whitelist_lemmas(config)
    normalizer = PredicateNormalizer(settings.relations_config_path, include_extended_relations=True)

    development = json.loads(args.dev.read_text(encoding="utf-8"))
    training, excluded = exclude_overlap(json.loads(args.train.read_text(encoding="utf-8")), development)
    if args.limit:
        training, development = training[:args.limit], development[:args.limit]

    train_rows = collect_rows(training, "train", args.cache_dir / "scorer_train.joblib", normalizer, whitelist)
    dev_rows = collect_rows(development, "dev", args.cache_dir / "scorer_dev.joblib", normalizer, whitelist)
    validation_rows = [row for row in dev_rows if row["document"] % 2 == 1]

    features, labels = matrix(train_rows)
    groups = np.array([row["document"] for row in train_rows])
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cross_validation = cross_val_score(build_classifier(), features, labels, groups=groups, cv=folds, scoring="f1")
    model = build_classifier().fit(features, labels)

    validation_features, validation_labels = matrix(validation_rows)
    validation_types = [row["relation_type"] for row in validation_rows]
    new_probabilities = model.predict_proba(validation_features)[:, 1]
    report = {
        "feature_version": FEATURE_VERSION, "feature_count": int(features.shape[1]),
        "label_rule": "mapped relations: exact directed gold fact; other relations: any gold relation "
                      "between the two entities; unaligned arguments left out",
        "training": {"documents": len(training), "rows": int(len(labels)), "positive": int(labels.sum()),
                     "by_relation_type": dict(Counter(row["relation_type"] for row in train_rows).most_common()),
                     "excluded_training_titles": excluded},
        "cross_validation": {"scheme": "5-fold, stratified, grouped by document",
                             "f1_mean": round(float(cross_validation.mean()), 4),
                             "f1_std": round(float(cross_validation.std()), 4)},
        "validation_note": "odd-indexed dev documents; not used for fitting",
        "validation": evaluate(new_probabilities, validation_labels),
        "validation_related_to_gate": gate_report(new_probabilities, validation_labels, validation_types,
                                                  settings.min_related_confidence),
        "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "dev_sha256": hashlib.sha256(args.dev.read_bytes()).hexdigest(),
    }
    if args.previous.exists():
        previous = joblib.load(args.previous)
        if isinstance(previous, dict) and previous.get("feature_version") == FEATURE_VERSION:
            old_probabilities = previous["model"].predict_proba(validation_features)[:, 1]
            report["validation_previous_model"] = evaluate(old_probabilities, validation_labels)
            report["validation_previous_related_to_gate"] = gate_report(
                old_probabilities, validation_labels, validation_types, settings.min_related_confidence)
    if args.legacy_gold.exists():
        records = [json.loads(line) for line in args.legacy_gold.read_text().splitlines() if line.strip()]
        legacy_features = [scorer_module._feature_vector(_DummyTriple(record), whitelist) for record in records]
        predictions = model.predict_proba(legacy_features)[:, 1] >= 0.5
        report["legacy_gold_check"] = {
            "rows": len(records),
            "correct": int(sum(int(p) == record["label"] for p, record in zip(predictions, records)))}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"feature_version": FEATURE_VERSION, "model": model}, args.output)
    report["artifact_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("Datasets/External/ReDocRed/train_revised.json"))
    parser.add_argument("--dev", type=Path, default=Path("Datasets/External/ReDocRed/dev_revised.json"))
    parser.add_argument("--legacy-gold", type=Path, default=Path("Datasets/EvalGold/RelationsGold.jsonl"))
    parser.add_argument("--previous", type=Path, default=Path("TrainedModels/RelationScorer.joblib"),
                        help="Existing model scored on the same held-out rows for comparison.")
    parser.add_argument("--output", type=Path, default=Path("TrainedModels/RelationScorer.joblib"))
    parser.add_argument("--cache-dir", type=Path, default=Path("EvaluationResults/Evaluation/cache"))
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N documents of each split.")
    main(parser.parse_args())
