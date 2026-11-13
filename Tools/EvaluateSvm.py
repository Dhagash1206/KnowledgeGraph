"""Evaluate a frozen model; save predictions suitable for the official scorer."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
from KnowledgeGraphApp.Extraction.SemevalDataset import read_dataset
from KnowledgeGraphApp.Extraction.LexicalSvm import load_model, score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, default=Path('TrainedModels/LexicalSvm'))
    p.add_argument('--test', type=Path, default=Path('Datasets/External/Semeval2010/TEST_FILE_FULL.TXT'))
    p.add_argument('--output', type=Path, default=Path('EvaluationResults/SvmTestResults'))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/'Metrics.json').exists():
        raise FileExistsError('Preserve previous evaluation; choose another output')
    manifest = json.loads((a.model/'Manifest.json').read_text())
    if hashlib.sha256((a.model/'Model.joblib').read_bytes()).hexdigest() != manifest['model_sha256']:
        raise ValueError('Model hash mismatch')
    model = load_model(a.model/'Model.joblib')  # trusted local artifact only
    rows = read_dataset(a.test)
    if len(rows) != 2717:
        raise ValueError('Expected official 2717-example test')
    predictions = model.predict(rows)
    for name, labels in [('predictions',predictions),('gold',[r['label'] for r in rows])]:
        (a.output/f'{name}.txt').write_text(''.join(f'{r["example_id"]}\t{y}\n' for r,y in zip(rows,labels)),encoding='utf-8')
    result = score([r['label'] for r in rows],predictions)
    result.update(test_sha256=hashlib.sha256(a.test.read_bytes()).hexdigest(),
                  model_sha256=manifest['model_sha256'], scope='supplied noun pairs; lexical baseline')
    (a.output/'Metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
