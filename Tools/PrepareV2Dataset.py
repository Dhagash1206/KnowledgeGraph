"""Prepare conservative Re-DocRED v2 supervision without fabricating negatives.

This builds a development artifact, not reviewed finance data or a fresh test set.

By default every absent label is unknown, which leaves the model nothing to
learn rejection from. --closed-world-negatives adds explicit negatives for the
five relations Re-DocRED annotates (its revision was made to list them
exhaustively): a type-compatible entity pair without the gold fact is labelled
0 for that relation. The four finance relations stay unknown.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import random
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from Tools.EvaluateExternal import RELATIONS, reconstruct, gold_facts
from KnowledgeGraphApp.Extraction.DatasetV2 import pair_supervision, development_split


def record_keys(record):
    return record['title'].strip().casefold(), ' '.join(reconstruct(record)[0].casefold().split())


# Relations with a Re-DocRED property; only these can be closed-world.
REDOCRED_RELATIONS = frozenset(relation for relation, _ in RELATIONS.values())


def closed_world_negatives(groups, typed, positives, ontology, rng, sample, per_positive=8, minimum=20):
    """(source, relation, target) facts that are type-compatible but absent from gold.

    With `sample` (training split) whole entity pairs are sampled, at most
    max(minimum, per_positive * positive pairs) per document, so a kept pair is
    negative for every relation its types allow. Evaluation splits keep every
    pair: sampled negatives there would overstate precision.
    """
    positive_pairs = {(source, target) for source, _, target in positives}
    negatives_by_pair = {}
    for source in groups:
        for target in groups:
            if source == target:
                continue
            absent = [(source, relation, target) for relation in sorted(REDOCRED_RELATIONS)
                      if relation in ontology['relations']
                      and any([a, b] in ontology['relations'][relation]['entity_type_pairs']
                              for a in typed[source] for b in typed[target])
                      and (source, relation, target) not in positives]
            if absent:
                negatives_by_pair[source, target] = absent
    # Pairs holding a positive keep their other labels as negatives too.
    kept = [pair for pair in negatives_by_pair if pair in positive_pairs]
    pool = sorted(pair for pair in negatives_by_pair if pair not in positive_pairs)
    kept += (rng.sample(pool, min(len(pool), max(minimum, per_positive * len(positive_pairs))))
             if sample else pool)
    return {fact for pair in kept for fact in negatives_by_pair[pair]}


def convert_record(record, origin, index, ontology, negative_rng=None, sample_negatives=True):
    text, sentences, starts, mentions = reconstruct(record)
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    document_id = f'redocred:{origin}:{index}:{content_hash[:12]}'
    groups = {str(i): [] for i in range(len(record['vertexSet']))}
    typed = {str(i): set() for i in range(len(record['vertexSet']))}
    output_mentions = []
    for i, mention in enumerate(mentions):
        mid = f'{document_id}:m{i}'
        groups[str(mention['cluster'])].append(mid)
        if mention['type']:
            typed[str(mention['cluster'])].add(mention['type'])
        output_mentions.append({'id': mid, 'document_id': document_id, 'text': mention['text'],
                                'start': mention['start'], 'end': mention['end'],
                                'type': mention['type'] or 'OTHER', 'sentence_idx': mention['sent_id'],
                                'detector': 'dataset_annotation'})
    relations = list(ontology['relations'])
    positives = gold_facts(record)
    negatives = (closed_world_negatives(groups, typed, positives, ontology, negative_rng, sample_negatives)
                 if negative_rng is not None else set())
    candidates = []
    for source in groups:
        for target in groups:
            if source == target:
                continue
            supported = any([a, b] in config['entity_type_pairs'] for config in ontology['relations'].values()
                            for a in typed[source] for b in typed[target])
            if not supported and not any(s == source and t == target for s, _, t in positives):
                continue
            candidates.append({'subject_group': source, 'object_group': target,
                               'type_supported': supported,
                               **pair_supervision({source}, {target}, relations, positives, negatives),
                               'assertion_targets': {'polarity': None, 'modality': None, 'lifecycle': None},
                               'assertion_loss_mask': {'polarity': False, 'modality': False, 'lifecycle': False}})
    return {'schema_version': '2.dataset.1', 'document_id': document_id, 'source': origin,
            'source_index': index, 'title': record['title'], 'content_hash': content_hash,
            'text': text, 'sentences': [{'start': start, 'text': sentence} for start, sentence in zip(starts, sentences)],
            'mentions': output_mentions, 'annotated_groups': groups,
            'gold_relations': sorted(positives), 'candidates': candidates,
            'supervision_scope': 'document_entity_group; not a claim that every mention pair expresses the relation',
            'annotation_provenance': 'upstream Re-DocRED; no new human finance review',
            'negative_policy': ('closed-world sample for Re-DocRED relations' if negative_rng is not None
                                else 'none; absent labels are unknown'),
            'evidence_annotation_status': 'not converted; do not train evidence selection from full document text'}


def prepare(args):
    ontology = yaml.safe_load(args.ontology.read_text(encoding='utf-8'))
    closed_world = getattr(args, 'closed_world_negatives', False)
    sources = {name: args.data_dir / f'{name}_revised.json' for name in ('train', 'dev', 'test')}
    records = {name: json.loads(path.read_text(encoding='utf-8')) for name, path in sources.items()}
    blocked_titles, blocked_texts = set(), set()
    for name in ('dev', 'test'):
        for record in records[name]:
            title, text = record_keys(record)
            blocked_titles.add(title)
            blocked_texts.add(text)
    if args.output.exists():
        raise FileExistsError('Use a new output directory; existing datasets are immutable')
    args.output.mkdir(parents=True)
    counts, assignment, exclusions = Counter(), [], []
    negative_rng = random.Random(20261114) if closed_world else None
    handles = {name: (args.output / f'{name}.jsonl').open('w', encoding='utf-8')
               for name in ('train', 'validation', 'calibration', 'regression_dev', 'regression_test')}
    try:
        for origin, rows in records.items():
            for index, record in enumerate(rows):
                title, text = record_keys(record)
                if origin == 'train':
                    if title in blocked_titles or text in blocked_texts:
                        exclusions.append({'source_index': index, 'title': record['title'],
                                           'reason': 'title/text overlap with regression data or earlier training record'})
                        continue
                    blocked_titles.add(title)
                    blocked_texts.add(text)
                    split = development_split(title)
                else:
                    split = f'regression_{origin}'
                converted = convert_record(record, origin, index, ontology, negative_rng,
                                           sample_negatives=split == 'train')
                handles[split].write(json.dumps(converted, ensure_ascii=False) + '\n')
                counts[split] += 1
                for candidate in converted['candidates']:
                    counts['candidates'] += 1
                    counts['positive_labels'] += candidate['targets'].count(1)
                    counts['explicit_negative_labels'] += candidate['targets'].count(0)
                    counts['unknown_labels'] += candidate['targets'].count(None)
                assignment.append({'document_id': converted['document_id'], 'split': split,
                                   'group_key': title, 'content_hash': converted['content_hash']})
    finally:
        for handle in handles.values():
            handle.close()
    manifest = {'schema_version': '2.dataset.1', 'seed': 'doc2graph-v2-20261111',
                'label_order': list(ontology['relations']), 'counts': dict(counts),
                'source_hashes': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
                'ontology_sha256': hashlib.sha256(args.ontology.read_bytes()).hexdigest(),
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'masking_code_sha256': hashlib.sha256(Path('KnowledgeGraphApp/Extraction/DatasetV2.py').read_bytes()).hexdigest(),
                'output_hashes': {name: hashlib.sha256((args.output / f'{name}.jsonl').read_bytes()).hexdigest() for name in handles},
                'exclusions': exclusions, 'assignments': assignment,
                'release_holdout_available': False,
                'negative_policy': ('closed-world: type-compatible pairs without the gold fact are explicit '
                                    'negatives for FOUNDED, EMPLOYED_BY, MEMBER_OF, OWNS, SUBSIDIARY_OF '
                                    '(sampled in train, all pairs in the other splits)'
                                    if closed_world else 'none; absent labels are unknown'),
                'limitations': ['Old dev/test are regression-only; training pool previously used in experiments.',
                               'Exact title/text dedup only; semantic company/deal/template overlap remains unaudited.',
                               ('Negatives assume Re-DocRED is exhaustive for its five relations; '
                                'no finance negatives or assertion supervision.' if closed_world else
                                'All absent labels are unknown. No explicit negative or finance assertion supervision.'),
                               'Document group facts must not be broadcast as local mention-pair evidence.',
                               'Not sufficient for production training or a finance quality claim.']}
    (args.output / 'Manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'counts': dict(counts), 'excluded': len(exclusions), 'release_holdout_available': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('Datasets/External/ReDocRed'))
    parser.add_argument('--ontology', type=Path, default=Path('Configuration/OntologyV2.yaml'))
    parser.add_argument('--output', type=Path, default=Path('Datasets/prepared/v2_redocred'))
    parser.add_argument('--closed-world-negatives', action='store_true',
                        help='Label sampled type-compatible pairs without a gold fact as explicit negatives '
                             'for the five Re-DocRED relations.')
    prepare(parser.parse_args())
