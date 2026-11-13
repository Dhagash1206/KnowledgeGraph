"""Train the entity resolver on Re-DocRED alias clusters and tune its merge threshold.

Every Re-DocRED entity lists the surface names used for it, so two names in the
same cluster are a positive pair and two names in different clusters of the
same type are a negative pair. Pairs come from the production blocking step,
so training sees the same candidates as inference.

Weights are fitted on the training documents. The merge threshold is chosen on
the even-indexed development documents and the reported scores come from the
odd-indexed ones, which take no part in either step. The test split is not read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from Tools.EvaluateExternal import reconstruct
from Tools.TrainLearned import exclude_overlap
from KnowledgeGraphApp.Extraction.EntityRecognizer import Entity
from KnowledgeGraphApp.GraphStorage import EntityResolver as resolver_module
from KnowledgeGraphApp.GraphStorage.EntityResolver import (
    FEATURE_NAMES,
    build_candidate_pairs,
    build_classifier,
    decided_by_rule,
    fit_document_names,
)

THRESHOLD_GRID = [round(0.05 + 0.01 * step, 2) for step in range(95)]


def document_entities(record) -> tuple[list[Entity], dict[str, int]]:
    """One entity per distinct (cluster, name, type); returns entities and their gold cluster."""
    _, _, _, mentions = reconstruct(record)
    seen, entities, cluster_of = set(), [], {}
    for mention in mentions:
        name = mention["text"].strip()
        if not mention["type"] or not name:
            continue
        key = (mention["cluster"], name, mention["type"])
        if key in seen:
            continue
        seen.add(key)
        entity_id = f"{mention['cluster']}:{len(entities)}"
        entities.append(Entity(entity_id, name, name, mention["type"], 0, len(name), "gold", "gold:0", ""))
        cluster_of[entity_id] = mention["cluster"]
    return entities, cluster_of


def classifier_rows(records):
    """Feature rows for the pairs the classifier decides (rule-settled pairs are left out)."""
    features, labels, document_index = [], [], []
    for index, record in enumerate(records):
        entities, cluster_of = document_entities(record)
        if not entities:
            continue
        fit_document_names(entities)
        for first, second in build_candidate_pairs(entities):
            if decided_by_rule(first, second) is not None:
                continue
            features.append(resolver_module._feature_vector(first.canonical_name, second.canonical_name, first.type))
            labels.append(int(cluster_of[first.entity_id] == cluster_of[second.entity_id]))
            document_index.append(index)
    return np.array(features), np.array(labels), np.array(document_index)


def scored_documents(records, model):
    """Per document: entities, gold clusters and the score of every candidate pair."""
    documents = []
    for record in records:
        entities, cluster_of = document_entities(record)
        if not entities:
            continue
        fit_document_names(entities)
        scored_pairs = []
        for first, second in build_candidate_pairs(entities):
            score = decided_by_rule(first, second)
            if score is None:
                vector = resolver_module._feature_vector(first.canonical_name, second.canonical_name, first.type)
                score = float(model.predict_proba([vector])[0][1])
            scored_pairs.append((first.entity_id, second.entity_id, score))
        documents.append((entities, cluster_of, scored_pairs))
    return documents


def merge_counts(documents, threshold: float) -> dict[str, int]:
    """Pairwise counts after transitive merging, over all same-type name pairs.

    A gold pair the blocking step never proposed is a miss, and a pair joined
    only through a chain of merges is counted like a direct merge.
    """
    true_positive = false_positive = false_negative = 0
    for entities, cluster_of, scored_pairs in documents:
        parent = {entity.entity_id: entity.entity_id for entity in entities}

        def find(identifier):
            while parent[identifier] != identifier:
                parent[identifier] = parent[parent[identifier]]
                identifier = parent[identifier]
            return identifier

        for first_id, second_id, score in scored_pairs:
            if score >= threshold:
                parent[find(first_id)] = find(second_id)
        for position, first in enumerate(entities):
            for second in entities[position + 1:]:
                if first.type != second.type:
                    continue
                gold = cluster_of[first.entity_id] == cluster_of[second.entity_id]
                predicted = find(first.entity_id) == find(second.entity_id)
                true_positive += gold and predicted
                false_positive += predicted and not gold
                false_negative += gold and not predicted
    return {"tp": int(true_positive), "fp": int(false_positive), "fn": int(false_negative)}


def summarize(counts: dict[str, int], beta: float) -> dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    weighted = ((1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
                if precision + recall else 0.0)
    return {**counts, "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), f"f{beta:g}": round(weighted, 4)}


def select_threshold(documents, beta: float) -> tuple[float, list[dict]]:
    curve = [{"threshold": threshold, **summarize(merge_counts(documents, threshold), beta)}
             for threshold in THRESHOLD_GRID]
    best_value = max(row[f"f{beta:g}"] for row in curve)
    # Highest threshold among ties: prefer fewer merges when the score is equal.
    best = [row for row in curve if row[f"f{beta:g}"] == best_value][-1]
    return best["threshold"], curve


def legacy_gold_check(gold_path: Path, model, threshold: float) -> dict:
    """Accuracy on the original hand-written pairs, which training no longer uses."""
    records = [json.loads(line) for line in gold_path.read_text().splitlines() if line.strip()]
    correct = 0
    for record in records:
        entity_type = record.get("type", "ORG")
        first, second = (Entity(key, record[key], record[key], entity_type, 0, len(record[key]), "gold", "gold:0", "")
                         for key in ("name_a", "name_b"))
        fit_document_names([first, second])
        score = decided_by_rule(first, second)
        if score is None:
            vector = resolver_module._feature_vector(first.canonical_name, second.canonical_name, first.type)
            score = float(model.predict_proba([vector])[0][1])
        correct += int(score >= threshold) == record["label"]
    return {"pairs": len(records), "correct": correct}


def main(args):
    development = json.loads(args.dev.read_text(encoding="utf-8"))
    training, excluded = exclude_overlap(json.loads(args.train.read_text(encoding="utf-8")), development)

    features, labels, document_index = classifier_rows(training)
    print(f"Training pairs: {len(labels)} ({int(labels.sum())} positive) from {len(training)} documents", flush=True)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cross_validation = cross_val_score(build_classifier(), features, labels, groups=document_index,
                                       cv=folds, scoring="f1")
    model = build_classifier().fit(features, labels)

    calibration = scored_documents(development[::2], model)
    validation = scored_documents(development[1::2], model)
    threshold, curve = select_threshold(calibration, args.beta)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    logistic = model.named_steps["lr"]
    report = {
        "features": list(FEATURE_NAMES),
        "standardized_weights": dict(zip(FEATURE_NAMES, (round(float(w), 4) for w in logistic.coef_[0]))),
        "intercept": round(float(logistic.intercept_[0]), 4),
        "training": {"documents": len(training), "pairs": int(len(labels)), "positive_pairs": int(labels.sum()),
                     "excluded_training_titles": excluded,
                     "pairs_left_to_rules": "exact name, person-name conflict, abbreviation score >= 0.9"},
        "cross_validation": {"scheme": "5-fold, stratified, grouped by document; classifier pairs at 0.5",
                             "f1_mean": round(float(cross_validation.mean()), 4),
                             "f1_std": round(float(cross_validation.std()), 4)},
        "threshold": threshold,
        "threshold_selection": f"maximum F{args.beta:g} of merged name pairs on even-indexed dev documents",
        "calibration": summarize(merge_counts(calibration, threshold), args.beta),
        "validation": summarize(merge_counts(validation, threshold), args.beta),
        "validation_note": "odd-indexed dev documents; used for neither weights nor threshold",
        "validation_at_previous_threshold_0.85": summarize(merge_counts(validation, 0.85), args.beta),
        "calibration_curve": curve[::5],
        "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "dev_sha256": hashlib.sha256(args.dev.read_bytes()).hexdigest(),
        "artifact_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    if args.legacy_gold.exists():
        report["legacy_gold_check"] = legacy_gold_check(args.legacy_gold, model, threshold)
    args.output.with_suffix(".training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "calibration_curve"}, indent=2))
    print(f"\nSet resolution_threshold: {threshold} in Configuration/Settings.yaml to use the tuned value.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("Datasets/External/ReDocRed/train_revised.json"))
    parser.add_argument("--dev", type=Path, default=Path("Datasets/External/ReDocRed/dev_revised.json"))
    parser.add_argument("--legacy-gold", type=Path, default=Path("Datasets/EvalGold/ResolutionGold.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("TrainedModels/EntityResolver.joblib"))
    parser.add_argument("--beta", type=float, default=0.5,
                        help="F-beta used to pick the threshold; below 1 favours precision, "
                             "because a wrong merge fuses two real entities into one node.")
    main(parser.parse_args())
