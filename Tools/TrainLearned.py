"""Fit relation labels from independent training data; tune on dev only.

The shipped thresholds are tuned on all development documents, so scores on
those same documents ("dev_gold_entity_metrics") are optimistic. The honest
figure is "dev_held_out_metrics": thresholds tuned on one half of the
development documents and scored on the other half, in both directions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from threadpoolctl import threadpool_limits

from Tools.EvaluateExternal import reconstruct, gold_facts
from KnowledgeGraphApp.Ingestion.Preprocessor import SentenceChunk
from KnowledgeGraphApp.Extraction.EntityRecognizer import Entity
from KnowledgeGraphApp.Extraction.LearnedRelations import FEATURE_VERSION, RELATION_TYPES, build_candidates, LearnedRelationExtractor
from KnowledgeGraphApp.Extraction.PredicateNormalizer import load_relations_config


def labeled_document(record):
    _, sentences, starts, mentions = reconstruct(record)
    chunks = [SentenceChunk(f"gold:{i}", "gold", i, text, starts[i], starts[i] + len(text))
              for i, text in enumerate(sentences)]
    entities, id_map = [], {}
    for index, mention in enumerate(mentions):
        if mention["type"] is None:
            continue
        cid = f"gold:{mention['sent_id']}"
        ent = Entity(f"mention:{index}", mention["text"], mention["text"], mention["type"],
                     mention["local_start"], mention["local_end"], "gold", cid, chunks[mention["sent_id"]].text)
        entities.append(ent)
        id_map[ent.entity_id] = str(mention["cluster"])
    return chunks, entities, id_map


def examples(records, allowed_pairs, training):
    rng = random.Random(20261110)
    candidates, labels, doc_ids = [], [], []
    total_gold, covered_gold = Counter(), Counter()
    for doc_index, record in enumerate(records):
        chunks, entities, ids = labeled_document(record)
        pairs = build_candidates(chunks, entities, ids, allowed_pairs)
        gold = gold_facts(record)
        total_gold.update(f[1] for f in gold)
        positives, negatives = [], []
        for pair in pairs:
            target = [int((pair.source_id, relation, pair.target_id) in gold) for relation in RELATION_TYPES]
            covered_gold.update(relation for relation, value in zip(RELATION_TYPES, target) if value)
            (positives if any(target) else negatives).append((pair, target))
        if training:
            # All positives; deterministic negatives from every document.
            negatives = rng.sample(negatives, min(len(negatives), max(20, len(positives) * 8)))
        for pair, target in positives + negatives:
            candidates.append(pair)
            labels.append(target)
            doc_ids.append(doc_index)
    return candidates, np.asarray(labels, dtype=np.int8), doc_ids, dict(total_gold), dict(covered_gold)


def tune_threshold(labels, probabilities, total_gold):
    # The gold denominator includes relations outside candidate/type coverage.
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    # recall * positives is a whole number; rounding keeps "tp >= 3" exact.
    tp = np.rint(recall[:-1] * labels.sum())
    full_recall = tp / max(total_gold, 1)
    p = precision[:-1]
    f1 = np.divide(2 * p * full_recall, p + full_recall, out=np.zeros_like(p), where=p + full_recall > 0)
    valid = (thresholds >= 0.05) & (tp >= 3)
    if not valid.any():
        return 1.01
    # Stable tie-break: highest threshold among equally scoring candidates.
    best = np.flatnonzero(valid & (f1 == f1[valid].max()))[-1]
    return float(thresholds[best])


def held_out_threshold_metrics(development, dev_y, dev_ids, scores):
    """Two-fold cross-fitting over development documents (even / odd index).

    Each half is scored with thresholds chosen on the other half, so no
    document is scored with a threshold it helped to select.
    """
    document_parity = np.asarray(dev_ids) % 2
    gold_by_parity = [Counter(), Counter()]
    for doc_index, record in enumerate(development):
        gold_by_parity[doc_index % 2].update(fact[1] for fact in gold_facts(record))
    totals, per_relation, thresholds_used = Counter(), {}, {}
    for index, relation in enumerate(RELATION_TYPES):
        counts = Counter()
        for scored_half in (0, 1):
            tuning, scoring = document_parity != scored_half, document_parity == scored_half
            threshold = tune_threshold(dev_y[tuning, index], scores[tuning, index],
                                       gold_by_parity[1 - scored_half].get(relation, 0))
            thresholds_used.setdefault(relation, []).append(threshold)
            predictions = scores[scoring, index] >= threshold
            tp = int((predictions & dev_y[scoring, index].astype(bool)).sum())
            counts.update(tp=tp, fp=int(predictions.sum()) - tp,
                          fn=gold_by_parity[scored_half].get(relation, 0) - tp)
        per_relation[relation] = dict(counts)
        totals.update(counts)
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"scheme": "thresholds tuned on one half of dev documents (even/odd), scored on the other; both halves summed",
            "per_relation": per_relation, "thresholds_per_half": thresholds_used,
            "totals": {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4), "recall": round(recall, 4),
                       "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0}}


def evaluate_existing(args):
    """Add held-out dev scores for an already trained artifact; the artifact is not rewritten."""
    development = json.loads(args.dev.read_text(encoding="utf-8"))
    artifact = joblib.load(args.output)
    allowed_pairs = {tuple(p) for pairs in artifact["type_pairs"].values() for p in pairs}
    dev_candidates, dev_y, dev_ids, _, _ = examples(development, allowed_pairs, False)
    scores = LearnedRelationExtractor(artifact).probabilities(dev_candidates)
    report_path = args.output.with_suffix(".training.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["dev_gold_entity_metrics_note"] = "optimistic: thresholds were tuned on these same documents"
    report["dev_held_out_metrics"] = held_out_threshold_metrics(development, dev_y, dev_ids, scores)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["dev_held_out_metrics"], indent=2), flush=True)


def exclude_overlap(training, development):
    """Prefer honest holdout separation over trusting upstream split names."""
    def text_key(record):
        return " ".join(reconstruct(record)[0].casefold().split())
    blocked_titles = {r["title"].strip().casefold() for r in development}
    blocked_text = {text_key(r) for r in development}
    retained, excluded = [], []
    for record in training:
        title, text = record["title"].strip().casefold(), text_key(record)
        if title in blocked_titles or text in blocked_text:
            excluded.append(record["title"])
            continue
        retained.append(record)
        blocked_titles.add(title)
        blocked_text.add(text)
    return retained, excluded


def train(args):
    start = time.monotonic()
    training = json.loads(args.train.read_text(encoding="utf-8"))
    development = json.loads(args.dev.read_text(encoding="utf-8"))
    training, excluded = exclude_overlap(training, development)
    print(f"Excluded {len(excluded)} train records overlapping dev or earlier train records", flush=True)
    config = load_relations_config()
    type_pairs = {r: config[r]["entity_type_pairs"] for r in RELATION_TYPES}
    allowed_pairs = {tuple(p) for pairs in type_pairs.values() for p in pairs}
    candidates, y, _, train_gold, train_covered = examples(training, allowed_pairs, True)
    dev_candidates, dev_y, dev_ids, dev_gold, dev_covered = examples(development, allowed_pairs, False)
    print(json.dumps({"training_documents": len(training), "training_pairs": len(candidates),
                      "development_pairs": len(dev_candidates), "positives": y.sum(axis=0).tolist()}), flush=True)
    # All feature vocabulary/IDF and classifier weights are fitted on train only.
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=100000, min_df=2,
                                 sublinear_tf=True, strip_accents="unicode", dtype=np.float64)
    x = vectorizer.fit_transform(c.text for c in candidates)
    classifiers = []
    with threadpool_limits(limits=2):
        for index, relation in enumerate(RELATION_TYPES):
            model = LogisticRegression(C=4.0, solver="liblinear", class_weight="balanced", max_iter=300,
                                       random_state=20261110)
            model.fit(x, y[:, index])
            classifiers.append(model)
            print(f"Trained {relation}: {int(y[:, index].sum())} positive pairs", flush=True)
    artifact = {"feature_version": FEATURE_VERSION, "relations": RELATION_TYPES, "type_pairs": type_pairs,
                "vectorizer": vectorizer, "classifiers": classifiers, "thresholds": {},
                "metadata": {"train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
                             "dev_sha256": hashlib.sha256(args.dev.read_bytes()).hexdigest(),
                             "train_documents": len(training), "dev_documents": len(development),
                             "excluded_training_titles": excluded,
                             "seed": 20261110, "negative_sampling": "max(20, 8 * positive_pairs) per document",
                             "feature_description": "identity-masked entity-pair context word TF-IDF 1-3 grams",
                             "threshold_selection": "per-label dev F1, at least 3 TP, threshold >= .05"}}
    extractor = LearnedRelationExtractor(artifact)
    scores = extractor.probabilities(dev_candidates)
    totals = Counter()
    metrics = {}
    for index, relation in enumerate(RELATION_TYPES):
        gold_count = dev_gold.get(relation, 0)  # a relation may be absent from dev
        threshold = tune_threshold(dev_y[:, index], scores[:, index], gold_count)
        artifact["thresholds"][relation] = threshold
        predictions = scores[:, index] >= threshold
        tp = int((predictions & dev_y[:, index].astype(bool)).sum())
        fp = int(predictions.sum()) - tp
        fn = gold_count - tp
        metrics[relation] = {"tp": tp, "fp": fp, "fn": fn, "threshold": threshold}
        totals.update(tp=tp, fp=fp, fn=fn)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output, compress=3)
    report = {"metadata": artifact["metadata"], "training_pairs": len(candidates), "dev_pairs": len(dev_candidates),
              "train_gold": train_gold, "train_candidate_coverage": train_covered, "dev_gold": dev_gold,
              "dev_candidate_coverage": dev_covered, "dev_gold_entity_metrics": metrics, "totals": dict(totals),
              "dev_gold_entity_metrics_note": "optimistic: thresholds were tuned on these same documents",
              "dev_held_out_metrics": held_out_threshold_metrics(development, dev_y, dev_ids, scores),
              "elapsed_seconds": round(time.monotonic() - start, 1),
              "artifact_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix(".training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("TrainedModels/LearnedRelations.joblib"))
    parser.add_argument("--evaluate-existing", action="store_true",
                        help="Only add held-out dev scores to the report of the artifact at --output.")
    arguments = parser.parse_args()
    if arguments.evaluate_existing:
        evaluate_existing(arguments)
    elif arguments.train is None:
        parser.error("--train is required unless --evaluate-existing is given")
    else:
        train(arguments)
