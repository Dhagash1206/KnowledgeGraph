"""Train lexical baselines, select on development, refit full official training data."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from KnowledgeGraphApp.Extraction.SemevalDataset import read_dataset
from KnowledgeGraphApp.Extraction.LexicalSvm import UTDSVM, score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('Datasets/External/Semeval2010'))
    p.add_argument('--output', type=Path, default=Path('TrainedModels/LexicalSvm'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'Model.joblib').exists():
        raise FileExistsError('Use a new output directory to preserve prior runs')
    train_path = args.data/'TRAIN_FILE.TXT'
    records = read_dataset(train_path)
    if len(records) != 8000:
        raise ValueError('Expected 8000 official training records')
    groups = [hashlib.sha256(r['text'].encode()).hexdigest() for r in records]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    ti, vi = next(splitter.split(records, [r['label'] for r in records], groups))
    train, val = [records[i] for i in ti], [records[i] for i in vi]
    results, best, best_model = [], -1, None
    start = time.perf_counter()
    for stages in (False, True):
        for c in (0.01, 0.1, 1, 10):
            for weight in (None, 'balanced'):
                model = UTDSVM(c, weight, stages).fit(train)
                metrics = score([r['label'] for r in val], model.predict(val))
                row = dict(c=c, class_weight=weight, two_stage=stages, **metrics)
                results.append(row)
                print(f'{stages=} {c=} {weight=} dev F1={metrics["macro_f1"]:.4f}', flush=True)
                if metrics['macro_f1'] > best:
                    best, best_model, selected = metrics['macro_f1'], model, row
    (args.output/'Development.json').write_text(json.dumps(dict(
        train_ids=[r['example_id'] for r in train], validation_ids=[r['example_id'] for r in val],
        runs=results, selected=selected), indent=2), encoding='utf-8')
    final = UTDSVM(selected['c'], selected['class_weight'], selected['two_stage']).fit(records)
    joblib.dump(final, args.output/'Model.joblib')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(schema_version='1.lexical_svm', system='UTD-inspired lexical baseline',
        sklearn_version=sklearn.__version__, training_count=len(records), seed=42,
        train_sha256=digest(train_path), model_sha256=digest(args.output/'Model.joblib'),
        selected=selected, seconds=time.perf_counter()-start, final_refit=True,
        omitted_features=['POS','dependency paths','WordNet','semantic roles','external knowledge'])
    (args.output/'Manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
