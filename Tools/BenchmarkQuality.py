#!/usr/bin/env python
"""Benchmark NLP quality against gold sets and optionally train scorers.

Usage:
    python Tools/BenchmarkQuality.py --eval Datasets/EvalGold/
    python Tools/BenchmarkQuality.py --eval Datasets/EvalGold/ --train
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate_ner(gold_path: Path) -> dict:
    """Compute span-level F1 for NER on NerGold.jsonl."""
    from KnowledgeGraphApp.Extraction.EntityRecognizer import NERRunner
    from KnowledgeGraphApp.Ingestion.Preprocessor import get_nlp

    records = [json.loads(ln) for ln in gold_path.read_text().splitlines() if ln.strip()]
    if not records:
        print("  [NER] Gold set empty — skipping")
        return {}

    nlp = get_nlp()
    ner = NERRunner()
    tp = fp = fn = 0

    for rec in records:
        text = rec["text"]
        gold_spans = {(s["start"], s["end"], s["type"]) for s in rec["entities"]}
        doc = nlp(text)
        pred_spans = {
            (e.start_char, e.end_char, e.type)
            for e in ner.run(doc, "eval", text)
        }
        tp += len(gold_spans & pred_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def train_relation_scorer(gold_path: Path) -> dict:
    from KnowledgeGraphApp.Extraction.RelationScorer import RelationScorer
    print("  Training relation scorer ...")
    return RelationScorer.train(str(gold_path))


def evaluate_relations(gold_path: Path) -> dict:
    """Evaluate full-text extraction (including NER/coreference), not training rows.

    The small hand-authored corpus is a regression set, not an independent
    estimate of general-domain accuracy. Unknown predicates count as RELATED_TO.
    """
    from KnowledgeGraphApp.Ingestion.Preprocessor import get_nlp, SentenceChunk, build_windows
    from KnowledgeGraphApp.Extraction.EntityRecognizer import NERRunner
    from KnowledgeGraphApp.Extraction.CorefResolver import RuleBasedCorefResolver
    from KnowledgeGraphApp.Extraction.RelationExtractor import DepPatternExtractor
    from KnowledgeGraphApp.Extraction.PredicateNormalizer import PredicateNormalizer

    records = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    nlp, ner = get_nlp(), NERRunner()
    extractor, normalizer = DepPatternExtractor(), PredicateNormalizer()
    tp = fp = fn = exact = 0
    for record in records:
        # No language gate: these are labeled English snippets, including very
        # short sentences for which automatic language detection is unreliable.
        chunks = [SentenceChunk(f"eval:{i}", "eval", i, s.text, s.start_char, s.end_char)
                  for i, s in enumerate(nlp(record["text"]).sents)]
        by_chunk = {c.chunk_id: ner.run(nlp(c.text), c.chunk_id, c.text) for c in chunks}
        entities = {e.entity_id: e for group in by_chunk.values() for e in group}
        coref = RuleBasedCorefResolver().resolve(chunks, by_chunk)
        predicted = set()
        for window, center in build_windows(chunks):
            raw = extractor.extract(
                nlp(" ".join(c.text for c in window)), list(entities.values()),
                center.chunk_id, center.sent_idx, coref, entities, window_chunks=window,
            )
            predicted.update((t.subject.canonical_name.casefold(), t.relation_type,
                              t.object.canonical_name.casefold())
                             for t in normalizer.normalize_batch(raw))
        gold = {(subject.casefold(), relation, obj.casefold()) for subject, relation, obj in record["relations"]}
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        exact += predicted == gold
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0,
        "exact_documents": exact, "documents": len(records),
    }


def train_entity_resolver(gold_path: Path) -> dict:
    from KnowledgeGraphApp.GraphStorage.EntityResolver import EntityResolver
    print("  Training entity resolver ...")
    return EntityResolver.train(str(gold_path))


def main():
    parser = argparse.ArgumentParser(description="Doc2Graph benchmark & training")
    parser.add_argument("--eval", required=True, help="Path to Datasets/EvalGold/ directory")
    parser.add_argument("--train", action="store_true", help="Train scorers on gold sets")
    args = parser.parse_args()

    eval_dir = Path(args.eval)
    print(f"\n[Doc2Graph Benchmark] ({eval_dir})\n" + "-" * 40)

    # NER
    ner_gold = eval_dir / "NerGold.jsonl"
    if ner_gold.exists():
        result = evaluate_ner(ner_gold)
        if result:
            print(f"  NER    P={result['precision']}  R={result['recall']}  F1={result['f1']}")
    else:
        print("  [NER] NerGold.jsonl not found — skipping")

    relation_text_gold = eval_dir / "RelationTextGold.jsonl"
    if relation_text_gold.exists():
        result = evaluate_relations(relation_text_gold)
        print(f"  Relations (regression set) P={result['precision']}  R={result['recall']}  F1={result['f1']}")
        print(f"  Exact relation documents: {result['exact_documents']}/{result['documents']}")

    # Train scorers
    if args.train:
        print()
        rel_gold = eval_dir / "RelationsGold.jsonl"
        if rel_gold.exists():
            r = train_relation_scorer(rel_gold)
            print(f"  Relation scorer CV F1: {r['cv_f1_mean']:.3f} +/- {r['cv_f1_std']:.3f}")
        else:
            print("  [Relation scorer] RelationsGold.jsonl not found — skipping")

        res_gold = eval_dir / "ResolutionGold.jsonl"
        if res_gold.exists():
            r = train_entity_resolver(res_gold)
            print(f"  Entity resolver  CV F1: {r['cv_f1_mean']:.3f} +/- {r['cv_f1_std']:.3f}")
        else:
            print("  [Entity resolver] ResolutionGold.jsonl not found — skipping")

    print()


if __name__ == "__main__":
    main()
