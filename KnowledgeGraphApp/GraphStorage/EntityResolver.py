"""Entity resolver — classical ML: TF-IDF cosine + Jaro-Winkler + initials-key.

Blocking:
  1. Token-overlap (stopword-filtered, corporate suffixes removed, bucket cap 100)
  2. Initials-key  ("IBM" ↔ "International Business Machines")

Scoring (Logistic Regression; Tools/TrainEntityResolver.py trains it on
Re-DocRED alias clusters and tunes the merge threshold on held-out documents):
  - jaro_winkler similarity
  - TF-IDF char n-gram cosine similarity
  - token overlap ratio
  - abbreviation score (one name is a prefix abbreviation of the other)
  - token containment, same first / last word, word-count ratio, single-word flag
  Each feature appears once for all types and once per PERSON / ORG / GPE,
  because "X" vs "X Y" is usually one company but two different places.

Cross-document resolution:
  Before writing a new entity, query Neo4j full-text index + initials_key
  property index for existing candidates and score with the same LR.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional, Any

from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from KnowledgeGraphApp.Extraction.EntityRecognizer import Entity

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = _PROJECT_ROOT / "TrainedModels" / "EntityResolver.joblib"

# ── Corporate suffix stoplist ─────────────────────────────────────────────────

CORP_STOP = frozenset({
    "inc", "corp", "ltd", "llc", "co", "company", "group",
    "university", "foundation", "institute", "association",
    "organization", "agency", "corporation", "enterprises",
    "holdings", "partners", "solutions", "services", "systems",
    "technologies", "technology", "international", "global",
})


# ── Utility functions ─────────────────────────────────────────────────────────

def _tokenize(name: str) -> list[str]:
    tokens = re.sub(r"[^\w\s]", " ", name.lower()).split()
    return [t for t in tokens if t not in CORP_STOP and len(t) > 1]


def compute_initials_key(name: str) -> str:
    """'International Business Machines' → 'IBM', 'Elon Musk' → 'EM', 'IBM' → 'IBM'."""
    clean = re.sub(r"[^\w\s]", "", name).strip()
    words = clean.split()
    if (
        len(words) == 1
        and re.fullmatch(r"[A-Z]{2,}[a-z]?", words[0])
        and len(words[0]) <= 8
    ):
        return words[0].upper()
    tokens = re.sub(r"[^\w\s]", " ", name).split()
    expanded_tokens = []
    for token in tokens:
        expanded_tokens.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", token))
    return "".join(
        t[0].upper() for t in expanded_tokens
        if t and t[0].isalpha() and t.casefold() not in {"a", "an", "the", "of", "and"}
    )


def _token_overlap_ratio(a: str, b: str) -> float:
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _abbreviation_score(a: str, b: str) -> float:
    """1.0 if one name is a plausible abbreviation of the other (initials match)."""
    ka, kb = compute_initials_key(a), compute_initials_key(b)
    # 'IBM' matches key of 'International Business Machines'
    a_clean = re.sub(r"\W", "", a.upper())
    b_clean = re.sub(r"\W", "", b.upper())
    if ka == b_clean or kb == a_clean:
        return 1.0
    if ka == kb and len(ka) >= 2:
        return 0.8
    # Mixed-case acronyms such as "IISc" may include a final disambiguating
    # letter while still explicitly abbreviating the full organization name.
    for acronym, initials in ((a_clean, kb), (b_clean, ka)):
        if (
            2 <= len(initials) <= len(acronym) <= 8
            and acronym.startswith(initials)
        ):
            return 0.9
    return 0.0


def _is_person_name_conflict(a: Entity, b: Entity) -> bool:
    """Reject merges where two multi-token people only share a surname."""
    if a.type != "PERSON" or b.type != "PERSON":
        return False
    a_tokens = _tokenize(a.canonical_name)
    b_tokens = _tokenize(b.canonical_name)
    if len(a_tokens) < 2 or len(b_tokens) < 2:
        return False
    if a_tokens[-1] != b_tokens[-1] or a_tokens[0] == b_tokens[0]:
        return False
    return (
        _abbreviation_score(a.canonical_name, b.canonical_name) == 0.0
        and a.canonical_name.casefold() not in b.canonical_name.casefold()
        and b.canonical_name.casefold() not in a.canonical_name.casefold()
    )


def _lucene_escape(text: str) -> str:
    """Escape Lucene special characters for Neo4j full-text queries."""
    specials = r'\+-&&||!(){}[]^"~*?:/'
    return "".join(f"\\{c}" if c in specials else c for c in text)


# ── TF-IDF cosine ─────────────────────────────────────────────────────────────

class _TFIDFCache:
    """Lazy char-ngram TF-IDF similarity (refit per batch)."""

    def __init__(self) -> None:
        self._vec: Optional[TfidfVectorizer] = None

    def fit(self, names: list[str]) -> None:
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._matrix = self._vec.fit_transform(names)
        self._names = names

    def cosine(self, a: str, b: str) -> float:
        vectorizer = self._vec
        if vectorizer is None:
            # Not fitted yet (a graph lookup before any document batch): use
            # the two names alone instead of silently scoring 0.0.
            try:
                vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit([a, b])
            except ValueError:  # both names empty: no n-grams to compare
                return 0.0
        return float(cosine_similarity(vectorizer.transform([a]), vectorizer.transform([b]))[0][0])


_tfidf = _TFIDFCache()


# ── Feature vector ────────────────────────────────────────────────────────────

# A separate binary "initials match" feature used to sit next to the
# abbreviation score. It was that same score thresholded at 0.8, so the two
# were perfectly collinear on the training data and received equal weights.
BASE_FEATURE_NAMES = (
    "jaro_winkler", "tfidf_cosine", "token_overlap", "abbreviation_score",
    "token_containment", "same_last_word", "same_first_word",
    "word_count_ratio", "single_word_name",
)
TYPED_BLOCKS = ("PERSON", "ORG", "GPE")
FEATURE_NAMES = (
    BASE_FEATURE_NAMES
    + tuple(f"{entity_type}:{name}" for entity_type in TYPED_BLOCKS for name in BASE_FEATURE_NAMES)
    + tuple(f"is_{entity_type}" for entity_type in TYPED_BLOCKS)
)


def _base_features(a_name: str, b_name: str) -> list[float]:
    a_tokens, b_tokens = set(_tokenize(a_name)), set(_tokenize(b_name))
    a_words, b_words = a_name.casefold().split(), b_name.casefold().split()
    both_have_words = bool(a_words) and bool(b_words)
    shorter, longer = sorted((len(a_words), len(b_words)))
    return [
        JaroWinkler.similarity(a_name, b_name),
        _tfidf.cosine(a_name, b_name),
        _token_overlap_ratio(a_name, b_name),
        _abbreviation_score(a_name, b_name),
        # Share of the shorter name's tokens found in the longer one.
        len(a_tokens & b_tokens) / min(len(a_tokens), len(b_tokens)) if a_tokens and b_tokens else 0.0,
        float(both_have_words and a_words[-1] == b_words[-1]),
        float(both_have_words and a_words[0] == b_words[0]),
        shorter / longer if longer else 0.0,
        float(shorter == 1),
    ]


def _feature_vector(a_name: str, b_name: str, entity_type: Optional[str] = None) -> list[float]:
    """Shared block, then the same values in the block of the entity type.

    A type outside TYPED_BLOCKS (or None) leaves the typed blocks at zero, so
    such pairs are scored by the shared weights alone.
    """
    base = _base_features(a_name, b_name)
    vector = list(base)
    for block_type in TYPED_BLOCKS:
        vector.extend(base if entity_type == block_type else [0.0] * len(base))
    vector.extend(float(entity_type == block_type) for block_type in TYPED_BLOCKS)
    return vector


def decided_by_rule(a: Entity, b: Entity) -> Optional[float]:
    """Score for pairs settled before the classifier is consulted, else None."""
    if a.type == b.type and a.canonical_name.casefold() == b.canonical_name.casefold():
        return 1.0
    if _is_person_name_conflict(a, b):
        return 0.0
    if _abbreviation_score(a.canonical_name, b.canonical_name) >= 0.9:
        return 1.0
    return None


def fit_document_names(entities: list[Entity]) -> None:
    """Fit the TF-IDF space on one document's names, as resolution does."""
    _tfidf.fit([e.canonical_name for e in entities])


def build_classifier():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
    ])


# ── Blocking ──────────────────────────────────────────────────────────────────

def build_candidate_pairs(
    entities: list[Entity],
    bucket_cap: int = 100,
) -> list[tuple[Entity, Entity]]:
    """Return candidate pairs via token-overlap and initials-key blocking."""
    pairs: set[tuple[str, str]] = set()  # store (id_a, id_b) tuples
    eid_map = {e.entity_id: e for e in entities}

    # 1. Token-overlap blocking
    tok_idx: dict[str, list[str]] = defaultdict(list)
    for e in entities:
        for tok in _tokenize(e.canonical_name):
            tok_idx[tok].append(e.entity_id)

    for bucket in tok_idx.values():
        if len(bucket) > bucket_cap:
            continue
        for a_id, b_id in combinations(bucket, 2):
            key = (min(a_id, b_id), max(a_id, b_id))
            pairs.add(key)

    # 2. Initials-key blocking
    init_idx: dict[str, list[str]] = defaultdict(list)
    for e in entities:
        k = compute_initials_key(e.canonical_name)
        keys = {k}
        if len(k) >= 2:
            keys.update(k[:size] for size in range(2, len(k) + 1))
        for key in keys:
            init_idx[key].append(e.entity_id)

    for bucket in init_idx.values():
        if len(bucket) > bucket_cap:
            continue
        for a_id, b_id in combinations(bucket, 2):
            key = (min(a_id, b_id), max(a_id, b_id))
            pairs.add(key)

    # Filter: same entity type, distinct entity IDs
    result = []
    for a_id, b_id in pairs:
        if a_id == b_id:
            continue
        a, b = eid_map.get(a_id), eid_map.get(b_id)
        if a and b and a.type == b.type:
            result.append((a, b))
    return result


# ── Resolver ──────────────────────────────────────────────────────────────────

class EntityResolver:
    """Merge near-duplicate entities within and across documents."""

    def __init__(self, threshold: float = 0.85):
        self._threshold = threshold
        self._lr = self._load_model()

    def _load_model(self):
        if MODEL_PATH.exists():
            try:
                import joblib
                model = joblib.load(MODEL_PATH)
                expected = getattr(model, "n_features_in_", len(FEATURE_NAMES))
                if expected != len(FEATURE_NAMES):
                    # Without this check every prediction would raise and the
                    # resolver would silently use the heuristic instead.
                    raise ValueError(f"model expects {expected} features, "
                                     f"this code builds {len(FEATURE_NAMES)}; retrain it")
                return model
            except Exception as exc:
                logger.warning("Entity resolver model %s could not be loaded (%s); "
                               "using the heuristic score.", MODEL_PATH, exc)
        return None

    def _score(self, a: Entity, b: Entity) -> float:
        rule_score = decided_by_rule(a, b)
        if rule_score is not None:
            return rule_score
        fv = _feature_vector(a.canonical_name, b.canonical_name, a.type)
        if self._lr is not None:
            try:
                return float(self._lr.predict_proba([fv])[0][1])
            except Exception:
                pass
        # Heuristic fallback
        jw = JaroWinkler.similarity(a.canonical_name, b.canonical_name)
        abbr = _abbreviation_score(a.canonical_name, b.canonical_name)
        return max(jw * 0.7 + abbr * 0.3, jw if abbr > 0.8 else 0.0)

    def resolve_within_doc(
        self, entities: list[Entity]
    ) -> dict[str, str]:
        """Return {entity_id: canonical_entity_id} mapping (Union-Find)."""
        if not entities:
            return {}

        fit_document_names(entities)
        pairs = build_candidate_pairs(entities)

        # Union-Find
        parent = {e.entity_id: e.entity_id for e in entities}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def preferred_root(first: str, second: str) -> tuple[str, str]:
            by_id = {entity.entity_id: entity for entity in entities}
            first_entity = by_id[first]
            second_entity = by_id[second]
            first_score = (
                len(_tokenize(first_entity.canonical_name)),
                len(first_entity.canonical_name),
                first_entity.canonical_name.casefold(),
            )
            second_score = (
                len(_tokenize(second_entity.canonical_name)),
                len(second_entity.canonical_name),
                second_entity.canonical_name.casefold(),
            )
            return (second, first) if second_score > first_score else (first, second)

        for a, b in pairs:
            if self._score(a, b) >= self._threshold:
                ra, rb = find(a.entity_id), find(b.entity_id)
                if ra != rb:
                    root, child = preferred_root(ra, rb)
                    parent[child] = root

        return {eid: find(eid) for eid in parent}

    async def resolve_against_graph(
        self, entity: Any, session
    ) -> Optional[str]:
        """Check Neo4j for an existing entity matching this one.

        Returns the existing entity ID to merge into, or None to create new.
        """
        name = getattr(entity, "canonical_name", None) or getattr(entity, "name", "")
        escaped = _lucene_escape(name)
        init_key = getattr(entity, "initials_key", None) or compute_initials_key(name)

        # Full-text candidates
        ft_result = await session.run(
            """
            CALL db.index.fulltext.queryNodes('entity_fulltext', $q)
            YIELD node
            WHERE node.type = $type
            RETURN node.id AS id, node.name AS name
            LIMIT 20
            """,
            q=escaped, type=entity.type,
        )
        ft_rows = [r.data() async for r in ft_result]

        # Initials-key candidates (catches abbreviation expansions)
        ik_result = await session.run(
            """
            MATCH (n:Entity {initials_key: $ik, type: $type})
            RETURN n.id AS id, n.name AS name
            """,
            ik=init_key, type=entity.type,
        )
        ik_rows = [r.data() async for r in ik_result]

        # Deduplicate candidates
        seen, candidates = set(), []
        for row in ft_rows + ik_rows:
            if row["id"] not in seen:
                seen.add(row["id"])
                candidates.append(row)

        for cand in candidates:
            cand_name = cand["name"]
            fv = _feature_vector(name, cand_name, entity.type)
            if self._lr is not None:
                try:
                    prob = float(self._lr.predict_proba([fv])[0][1])
                    if prob >= self._threshold:
                        return cand["id"]
                    continue
                except Exception:
                    pass
            # Heuristic fallback
            jw = JaroWinkler.similarity(name, cand_name)
            abbr = _abbreviation_score(name, cand_name)
            score = max(jw * 0.7 + abbr * 0.3, jw if abbr > 0.8 else 0.0)
            if score >= self._threshold:
                return cand["id"]

        return None

    # ── Training ──────────────────────────────────────────────────────────────

    @staticmethod
    def train(gold_path: str) -> dict:
        """Train LR on a small flat gold file ({"name_a", "name_b", "label"} rows).

        The shipped model comes from Tools/TrainEntityResolver.py, which has
        thousands of pairs; this stays for quick experiments and tests.
        """
        import joblib
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        records = [json.loads(ln) for ln in Path(gold_path).read_text().splitlines() if ln.strip()]
        if not records:
            raise ValueError("Gold set is empty.")

        # Fit the TF-IDF space first, as resolve_within_doc does at inference.
        # Without this the cosine feature was 0.0 for every training row.
        _tfidf.fit([name for rec in records for name in (rec["name_a"], rec["name_b"])])

        X, y = [], []
        for rec in records:
            # Format: {"name_a": "...", "name_b": "...", "label": 1 or 0}
            fv = _feature_vector(rec["name_a"], rec["name_b"])
            X.append(fv)
            y.append(rec["label"])

        X_arr, y_arr = np.array(X), np.array(y)

        pipe = build_classifier()

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X_arr, y_arr, cv=skf, scoring="f1")

        pipe.fit(X_arr, y_arr)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(pipe, MODEL_PATH)

        return {
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "n_samples": len(y),
        }
