"""Predict a relation for explicit <e1> and <e2> phrases."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
from KnowledgeGraphApp.Extraction.SemevalDataset import parse_marked
from KnowledgeGraphApp.Extraction.LexicalSvm import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sentence', help='Sentence containing <e1>...</e1> and <e2>...</e2>')
    p.add_argument('--model', type=Path, default=Path('TrainedModels/LexicalSvm'))
    a = p.parse_args()
    manifest = json.loads((a.model/'Manifest.json').read_text())
    path = a.model/'Model.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['model_sha256']:
        raise ValueError('Model hash mismatch')
    record = parse_marked(a.sentence)
    model = load_model(path)
    label = model.predict([record])[0]
    print(json.dumps(dict(**record, label=label, calibrated_confidence=None,
                          system=manifest['system']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
