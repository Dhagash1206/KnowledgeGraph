"""UTD-inspired lexical SVM baseline. No neural components or probability claims."""
import re
import sys
import types
import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from KnowledgeGraphApp.Extraction.SemevalDataset import FAMILIES, family


def features(record):
    text = record['text']
    tokens = [(m.group().lower(), m.start(), m.end()) for m in re.finditer(r'\w+|[^\w\s]', text)]
    out = {}
    def add(name, value):
        out[f'{name}={value}'] = 1
    for key in ('e1', 'e2'):
        start, end = record[key]
        if not 0 <= start < end <= len(text):
            raise ValueError('Invalid phrase offsets')
        inside = [w for w, a, b in tokens if a >= start and b <= end]
        add(key, text[start:end].lower())
        for w in inside:
            add(key+'_word', w)
        before = [w for w, a, b in tokens if b <= start]
        after = [w for w, a, b in tokens if a >= end]
        for j, w in enumerate(reversed(before[-2:])):
            add(key+f'_before{j}', w)
        if after:
            add(key+'_after', after[0])
    left, right = sorted((record['e1'], record['e2']))
    between = [w for w, a, b in tokens if a >= left[1] and b <= right[0]]
    add('order', record['e1'][0] < record['e2'][0])
    add('between_sequence', ' '.join(between))
    for w in between:
        add('between_word', w)
        add('between_prefix5', w[:5])
    for a, b in zip(between, between[1:]):
        add('between_bigram', a+' '+b)
    return out


class UTDSVM:
    def __init__(self, c=1.0, class_weight=None, two_stage=True):
        self.c, self.class_weight, self.two_stage = c, class_weight, two_stage
        self.vectorizer = DictVectorizer()
        self.directions = {}

    def _classifier(self):
        return LinearSVC(C=self.c, class_weight=self.class_weight, random_state=42, max_iter=10000)

    @staticmethod
    def _compact_indices(matrix):
        # Recent scikit-learn builds this matrix with 64-bit indices, which
        # LinearSVC (liblinear) rejects; the matrix is far below the 32-bit limit.
        matrix.indices = matrix.indices.astype(np.int32, copy=False)
        matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
        return matrix

    def fit(self, records):
        x = self._compact_indices(self.vectorizer.fit_transform([features(r) for r in records]))
        labels = np.array([r['label'] for r in records])
        self.main = self._classifier().fit(x, [family(y) for y in labels] if self.two_stage else labels)
        if self.two_stage:
            for f in FAMILIES:
                indices = np.array([i for i, y in enumerate(labels) if family(y) == f])
                if len(indices) == 0:
                    raise ValueError(f'Missing family {f}')
                ys = labels[indices]
                self.directions[f] = (str(ys[0]) if len(set(ys)) == 1 else self._classifier().fit(x[indices], ys))
        return self

    def predict(self, records):
        x = self._compact_indices(self.vectorizer.transform([features(r) for r in records]))
        pred = self.main.predict(x)
        if not self.two_stage:
            return pred.tolist()
        result = []
        for i, f in enumerate(pred):
            if f == 'Other':
                result.append(f)
            else:
                model = self.directions[f]
                result.append(model if isinstance(model, str) else str(model.predict(x[i])[0]))
        return result


def load_model(model_path):
    """Load Model.joblib, which was pickled while this module was named src.nlp.utd_svm.

    The pickle stores that old module path; aliasing it during loading keeps the
    checkpoint byte-identical, so its recorded model_sha256 still matches.
    """
    legacy_names = ("src", "src.nlp", "src.nlp.utd_svm")
    added_names = [name for name in legacy_names if name not in sys.modules]
    for name in legacy_names[:-1]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules.setdefault("src.nlp.utd_svm", sys.modules[__name__])
    try:
        return joblib.load(model_path)
    finally:
        for name in added_names:
            sys.modules.pop(name, None)


def score(gold, predicted):
    """Direction-aware nine-family macro F1; Other affects FP/FN but is not averaged."""
    if len(gold) != len(predicted) or not gold:
        raise ValueError('Nonempty aligned predictions required')
    rows = {}
    for f in FAMILIES:
        tp = sum(g == p and family(g) == f for g, p in zip(gold, predicted))
        actual = sum(family(g) == f for g in gold)
        emitted = sum(family(p) == f for p in predicted)
        p, r = tp/emitted if emitted else 0, tp/actual if actual else 0
        rows[f] = dict(tp=tp, fp=emitted-tp, fn=actual-tp, precision=p, recall=r,
                       f1=2*p*r/(p+r) if p+r else 0)
    return dict(macro_f1=sum(v['f1'] for v in rows.values())/9,
                accuracy=sum(g == p for g,p in zip(gold,predicted))/len(gold), per_family=rows)
