"""Train lexical baselines, select by cross-validation, refit on full official training data.

One fold is set aside before any selection. Settings are chosen by mean
macro-F1 over the other four folds; the chosen settings are then scored once on
the set-aside fold, which gives a development figure selection never saw.
"""
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
    fold_indices = [held for _, held in splitter.split(records, [r['label'] for r in records], groups)]
    # The first fold used to be both the selection set and the reported score.
    report_fold, selection_folds = fold_indices[0], fold_indices[1:]

    def subset(indices):
        return [records[i] for i in indices]

    results, best = [], -1
    start = time.perf_counter()
    for stages in (False, True):
        for c in (0.01, 0.1, 1, 10):
            for weight in (None, 'balanced'):
                fold_scores = []
                for held_out in range(len(selection_folds)):
                    fit_rows = subset(i for k, fold in enumerate(selection_folds) if k != held_out for i in fold)
                    val_rows = subset(selection_folds[held_out])
                    model = UTDSVM(c, weight, stages).fit(fit_rows)
                    fold_scores.append(score([r['label'] for r in val_rows], model.predict(val_rows))['macro_f1'])
                mean_f1 = sum(fold_scores) / len(fold_scores)
                row = dict(c=c, class_weight=weight, two_stage=stages, cv_macro_f1=mean_f1,
                           cv_fold_macro_f1=fold_scores)
                results.append(row)
                print(f'{stages=} {c=} {weight=} CV F1={mean_f1:.4f}', flush=True)
                if mean_f1 > best:
                    best, selected = mean_f1, row
    selection_rows, report_rows = subset(i for fold in selection_folds for i in fold), subset(report_fold)
    held_out_model = UTDSVM(selected['c'], selected['class_weight'], selected['two_stage']).fit(selection_rows)
    held_out_metrics = score([r['label'] for r in report_rows], held_out_model.predict(report_rows))
    selected = dict(selected, **held_out_metrics,
                    metrics_source='fold set aside before selection; not used to choose settings')
    print(f'Selected {selected["c"]=} {selected["class_weight"]=} {selected["two_stage"]=} '
          f'held-out F1={held_out_metrics["macro_f1"]:.4f}', flush=True)
    (args.output/'Development.json').write_text(json.dumps(dict(
        selection_ids=[r['example_id'] for r in selection_rows],
        held_out_ids=[r['example_id'] for r in report_rows],
        runs=results, selected=selected), indent=2), encoding='utf-8')
    final = UTDSVM(selected['c'], selected['class_weight'], selected['two_stage']).fit(records)
    joblib.dump(final, args.output/'Model.joblib')
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = dict(schema_version='1.lexical_svm', system='UTD-inspired lexical baseline',
        sklearn_version=sklearn.__version__, training_count=len(records), seed=42,
        train_sha256=digest(train_path), model_sha256=digest(args.output/'Model.joblib'),
        selected=selected, seconds=time.perf_counter()-start, final_refit=True,
        omitted_features=['POS','dependency paths','WordNet','semantic roles','external knowledge'])
    (args.output/'Manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
