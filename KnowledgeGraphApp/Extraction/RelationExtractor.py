"""Dependency-pattern relation extractor.

Extracts (subject, predicate, object) triples from spaCy dependency parses
using four pattern types:
  - SVO:        subject–verb–object
  - Appositive: token with dep_="appos"
  - Prepositional: verb/noun + prep + pobj
  - Nominal:    role-noun ("founder", "CEO") with nmod/poss child
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import re

from spacy.tokens import Doc, Token, Span

from KnowledgeGraphApp.Extraction.EntityRecognizer import Entity
from KnowledgeGraphApp.Ingestion.Preprocessor import SentenceChunk


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RawTriple:
    """An unscored, unresolved (subject, predicate, object) triple."""
    subject: Entity
    predicate: str          # verb lemma or pattern key (e.g. "join", "appos", "found")
    object: Entity
    prep: Optional[str]     # preposition token if applicable, else None
    source: str             # "svo" | "appos" | "prep" | "nominal"
    sent_idx: int           # sentence index in the document
    chunk_id: str           # actual source chunk ID (for evidence attribution)

    # The dep-tree distance between subject and object tokens
    dep_distance: int = 0
    # Whether a negation is in the governor's subtree
    has_negation: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────

ROLE_NOUNS = frozenset({
    "founder", "ceo", "director", "chairman", "president",
    "head", "chief", "owner", "partner", "investor",
    "manager", "officer", "executive",
})


_PRODUCT_KIND_NOUNS = frozenset({
    "project", "platform", "system", "product", "tool", "software",
    "application", "app", "framework", "library",
})


def _build_span_map(entities: list[Entity]) -> dict[tuple[int, int], Entity]:
    """Map (start_char, end_char) → Entity for fast lookup."""
    return {(e.start_char, e.end_char): e for e in entities}


def _token_to_entity(
    token: Token,
    span_map: dict[tuple[int, int], Entity],
    chunk_id: str = "",
    coref_map: dict[str, str] | None = None,
    entity_by_id: dict[str, Entity] | None = None,
) -> Optional[Entity]:
    """Try to find an NER entity covering this token (or its head noun phrase or via coreference)."""
    clean_text = token.text.rstrip(".,;:!?\"')}]")
    t_start, t_end = token.idx, token.idx + len(clean_text)

    # 1. Exact or overlapping character span in span_map (current sentence / chunk)
    for (s, e), ent in span_map.items():
        if (s == token.idx and e == token.idx + len(token.text)) or (s == t_start and e == t_end):
            if coref_map and entity_by_id and ent.entity_id in coref_map:
                return entity_by_id.get(coref_map[ent.entity_id], ent)
            return ent
        if not (e <= t_start or s >= t_end):
            if coref_map and entity_by_id and ent.entity_id in coref_map:
                return entity_by_id.get(coref_map[ent.entity_id], ent)
            return ent

    # 2. Try noun chunks overlapping with span_map
    for chunk in token.doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            for (s, e), ent in span_map.items():
                # A possessive company is not the head of "Acme's shareholders"
                # or "Acme's CFO". Only use the named noun-phrase head.
                if s <= chunk.root.idx < e:
                    if coref_map and entity_by_id and ent.entity_id in coref_map:
                        return entity_by_id.get(coref_map[ent.entity_id], ent)
                    return ent

    # 2b. "the DocSense project": a kind noun modified by a product name is that product.
    if token.lemma_.lower() in _PRODUCT_KIND_NOUNS:
        for modifier in token.children:
            if modifier.dep_ == "compound":
                for (s, e), ent in span_map.items():
                    if ent.type == "PRODUCT" and s <= modifier.idx < e:
                        return ent

    # 3. Exact string match within current chunk's span_map only
    exact_matches = [
        ent for ent in span_map.values()
        if ent.text.rstrip(".,;:!?\"')}]").casefold() == clean_text.casefold()
    ]
    if exact_matches:
        ent = min(exact_matches, key=lambda candidate: abs(candidate.start_char - t_start))
        if coref_map and entity_by_id and ent.entity_id in coref_map:
            return entity_by_id.get(coref_map[ent.entity_id], ent)
        return ent

    # 4. Coreference resolution for pronouns and definite NPs
    if coref_map and entity_by_id:
        tok_low = clean_text.lower()
        position_key = f"{chunk_id}:ref:{token.idx}"
        target_id = coref_map.get(position_key, coref_map.get(f"{chunk_id}:{tok_low}"))
        if not target_id:
            for chunk in token.doc.noun_chunks:
                if chunk.start <= token.i < chunk.end:
                    # "its software" is the software, not the owner "its"
                    # refers to; only a whole-phrase reference resolves.
                    possessed = chunk[0].dep_ == "poss" and token.i != chunk[0].i
                    target_id = coref_map.get(f"{chunk_id}:{chunk.text.lower()}") if possessed else coref_map.get(
                        f"{chunk_id}:ref:{chunk.start_char}",
                        coref_map.get(f"{chunk_id}:{chunk.text.lower()}"),
                    )
                    if target_id:
                        break
        if target_id and target_id in entity_by_id:
            return entity_by_id[target_id]

    return None


def _dep_distance(a: Token, b: Token) -> int:
    """Shortest path through the common ancestor, including both endpoints."""
    distances = {token.i: depth for depth, token in enumerate([a, *a.ancestors])}
    return min(
        (depth + distances[token.i] for depth, token in enumerate([b, *b.ancestors])
         if token.i in distances),
        default=999,
    )


def _has_negation(token: Token) -> bool:
    """Check direct negation on the token or its auxiliary children."""
    if any(c.dep_ == "neg" for c in token.children):
        return True
    for c in token.children:
        if c.dep_ in ("aux", "auxpass") and any(gc.dep_ == "neg" for gc in c.children):
            return True
    return False


def _subject_token(verb: Token) -> Optional[Token]:
    """Recover subjects shared by coordinated verbs and controlled clauses."""
    direct = next((c for c in verb.children if c.dep_ in {"nsubj", "nsubjpass"}), None)
    if direct is not None:
        if direct.lower_ in {"who", "which", "that"} and verb.dep_ == "relcl":
            return verb.head
        return direct
    if verb.dep_ in {"conj", "xcomp"} and verb.head.pos_ in {"VERB", "AUX"}:
        if verb.dep_ == "xcomp" and verb.head.lemma_ in {"ask", "tell", "instruct", "allow"}:
            return next((c for c in verb.head.children if c.dep_ == "dobj"), None)
        return _subject_token(verb.head)
    if verb.dep_ in {"acl", "relcl"}:
        if verb.head.lemma_ in {"agreement", "plan", "decision", "offer"}:
            return _subject_token(verb.head.head) if verb.head.head.pos_ == "VERB" else None
        return verb.head
    return None


def _is_planned(verb: Token) -> bool:
    return any(t.lemma_ in {"plan", "expect", "intend", "agree", "agreement", "offer"}
               or any(c.lower_ in {"will", "would", "may", "might", "could"} for c in t.children)
               for t in [verb, *verb.ancestors])


def _predicate(verb: Token) -> str:
    # An agreement/plan is not evidence that an acquisition has closed.
    if verb.lemma_ in {"acquire", "buy", "purchase"}:
        if _is_planned(verb):
            return "planned_acquire"
    if verb.lemma_ in {"use", "integrate", "deploy", "adopt"} and _is_planned(verb):
        return "planned_use"
    return verb.lemma_.lower()


_GROUP_HEAD_NOUNS = frozenset({
    "team", "department", "committee", "board", "unit", "office", "council",
    "group", "taskforce", "lab", "laboratory",
})
# Heads that name no identifiable thing once their determiner is removed
# ("the next phase", "the first step").
_VAGUE_HEAD_NOUNS = frozenset({
    "phase", "stage", "step", "part", "portion", "rest", "effort", "process",
    "work", "thing", "way", "kind", "sort", "lot", "number", "time",
})
_LEADING_DETERMINER = re.compile(
    r"^(?:the|a|an|its|their|his|her|our|this|that|these|those)\s+", re.IGNORECASE)


def _noun_phrase_concept(token: Token, chunk_id: str, first: Optional[Token] = None) -> Optional[Entity]:
    """Create a concept entity for a supported verb's common-noun object.

    A common noun phrase is not a product: it is typed CONCEPT, or ORG when
    its head names a group of people ("the NLP team").
    """
    for noun_chunk in token.doc.noun_chunks:
        if noun_chunk.root.i != token.i:
            continue
        # `first` widens the phrase to a participle the parser split off.
        if first is not None and first.i < noun_chunk.start:
            noun_chunk = token.doc[first.i:noun_chunk.end]
        text = noun_chunk.text.strip()
        name = _LEADING_DETERMINER.sub("", " ".join(text.split()).strip(".,;:"))
        head = noun_chunk.root.lemma_.lower()
        if len(name.split()) < 2 or head in _VAGUE_HEAD_NOUNS:
            return None
        return Entity(
            entity_id=f"{chunk_id}:concept:{noun_chunk.start_char}:{noun_chunk.end_char}",
            text=text,
            canonical_name=name,
            type="ORG" if head in _GROUP_HEAD_NOUNS else "CONCEPT",
            start_char=noun_chunk.start_char,
            end_char=noun_chunk.end_char,
            source="noun_phrase",
            chunk_id=chunk_id,
            sentence=token.doc.text,
        )
    return None


_ROLE_VERBS = frozenset({"lead", "head", "manage", "run", "direct", "oversee"})
# Verbs whose common-noun object is worth a concept node of its own.
_CONCEPT_OBJECT_VERBS = frozenset({
    "provide", "supply", "store", "use", "develop", "lead", "contribute",
    "monitor", "predict", "forecast", "contain",
})
_FUNDING_NOUNS = frozenset({"funding", "fund", "financing", "grant", "support", "money"})
_RECEIVE_VERBS = frozenset({"receive", "obtain", "secure", "get", "win"})


def _named_by_clause(token: Token, span_map, chunk_id, coref_map, entity_by_id) -> Optional[Entity]:
    """Resolve "a project called DocSense" to the entity it names."""
    for clause in token.children:
        if clause.dep_ == "acl" and clause.lemma_ in {"call", "name", "title", "dub"}:
            for named in clause.children:
                if named.dep_ in {"oprd", "dobj", "attr"}:
                    return _token_to_entity(named, span_map, chunk_id, coref_map, entity_by_id)
    return None


# ── Extractor ─────────────────────────────────────────────────────────────────

class DepPatternExtractor:
    """Extract relation triples from spaCy dependency parses."""

    def extract(
        self,
        doc: Doc,
        entities: list[Entity],
        center_chunk_id: str,
        center_sent_idx: int,
        coref_map: dict[str, str] | None = None,
        entity_by_id: dict[str, Entity] | None = None,
        window_chunks: list[SentenceChunk] | None = None,
    ) -> list[RawTriple]:
        # Entity offsets and coreference keys belong to individual chunks.
        # Slice the parsed window back into those chunks before lookup; this
        # also attributes each relation to its actual evidence sentence.
        if window_chunks is not None:
            if doc.text != " ".join(chunk.text for chunk in window_chunks):
                raise ValueError("Parsed window text does not match its chunks")
            triples = []
            offset = 0
            for chunk in window_chunks:
                span = doc.char_span(offset, offset + len(chunk.text))
                if span is None:
                    raise ValueError("Chunk boundaries do not align with parsed tokens")
                triples.extend(self.extract(
                    span.as_doc(), [e for e in entities if e.chunk_id == chunk.chunk_id],
                    chunk.chunk_id, chunk.sent_idx, coref_map, entity_by_id,
                ))
                offset += len(chunk.text) + 1
            return triples

        span_map = _build_span_map(entities)
        triples: list[RawTriple] = []
        for sent in doc.sents:
            triples += self._svo(sent, span_map, center_chunk_id, center_sent_idx, coref_map, entity_by_id)
            triples += self._appositive(sent, span_map, center_chunk_id, center_sent_idx, coref_map, entity_by_id)
            triples += self._prepositional(sent, span_map, center_chunk_id, center_sent_idx, coref_map, entity_by_id)
            triples += self._nominal(sent, span_map, center_chunk_id, center_sent_idx, coref_map, entity_by_id)
            triples += self._business_patterns(sent, span_map, center_chunk_id, center_sent_idx, coref_map, entity_by_id)
        return triples

    # ── Pattern implementations ───────────────────────────────────────────────

    def _svo(
        self, sent: Span, span_map, chunk_id, sent_idx,
        coref_map: dict[str, str] | None = None,
        entity_by_id: dict[str, Entity] | None = None,
    ) -> list[RawTriple]:
        """SVO: 'Musk founded Tesla', 'Google hired Alice'."""
        triples = []
        for token in sent:
            if token.pos_ != "VERB":
                continue
            # Find subject
            subj_tok = _subject_token(token)
            # Find object
            obj_tok = next(
                (c for c in token.children if c.dep_ in {"dobj", "attr", "oprd"}),
                None,
            )
            # Convert passive voice to actor -> patient before normalization.
            if subj_tok is not None and (subj_tok.dep_ == "nsubjpass" or
                    any(c.dep_ == "agent" for c in token.children)):
                agent = next((c for c in token.children if c.dep_ == "agent"), None)
                actor = next((c for c in agent.children if c.dep_ == "pobj"), None) if agent is not None else None
                subj_tok, obj_tok = actor, subj_tok
            # "owns 22% of Acme": the percentage modifies the named target.
            if obj_tok is not None and token.lemma_ in {"own", "hold"}:
                of = next((c for c in obj_tok.children if c.dep_ == "prep" and c.lower_ == "of"), None)
                if of is not None:
                    obj_tok = next((c for c in of.children if c.dep_ == "pobj"), obj_tok)
            # "uses named entity recognition": the parser reads the modifier
            # "named" as a verb, leaving "uses" without an object.
            participle = None
            if obj_tok is None and token.lemma_ in {"use", "provide", "develop"}:
                participle = next((c for c in token.children
                                   if c.dep_ == "xcomp" and c.tag_ == "VBN" and c.i == token.i + 1), None)
                if participle is not None:
                    obj_tok = next((c for c in participle.children if c.dep_ in {"oprd", "dobj"}), None)
            if subj_tok is None or obj_tok is None:
                continue
            # "partnered with X to develop Y": the infinitive states a goal.
            # Planned predicates keep their own explicit modality.
            if (
                token.tag_ == "VB"
                and any(c.dep_ == "aux" and c.lower_ == "to" for c in token.children)
                and not _predicate(token).startswith("planned_")
            ):
                continue

            prep = next(
                (child for child in token.children
                 if child.dep_ == "prep" and child.text.lower() == "in"),
                None,
            )
            prep_text = prep.text if prep is not None else next(
                (c.text for c in token.children if c.dep_ == "prep"), None
            )

            # Expand coordination across subjects and objects (Cartesian product)
            subj_tokens = [subj_tok, *subj_tok.conjuncts]
            obj_tokens = [obj_tok, *obj_tok.conjuncts]

            for s_tok in subj_tokens:
                subj_ent = _token_to_entity(s_tok, span_map, chunk_id, coref_map, entity_by_id)
                if subj_ent is None:
                    continue

                for o_tok in obj_tokens:
                    obj_ent = _token_to_entity(o_tok, span_map, chunk_id, coref_map, entity_by_id)
                    if token.lemma_ == "store" and prep is not None:
                        storage_obj = next(
                            (child for child in prep.children if child.dep_ == "pobj"),
                            None,
                        )
                        storage_ent = (
                            _token_to_entity(
                                storage_obj, span_map, chunk_id, coref_map, entity_by_id
                            )
                            if storage_obj is not None else None
                        )
                        if storage_ent is not None:
                            obj_ent = storage_ent
                    # "funded the initial phase of the project": the vague
                    # head only scopes the named thing after "of".
                    if obj_ent is None and o_tok.lemma_.lower() in _VAGUE_HEAD_NOUNS:
                        of = next((c for c in o_tok.children
                                   if c.dep_ == "prep" and c.lower_ == "of"), None)
                        whole = next((c for c in of.children if c.dep_ == "pobj"), None) if of is not None else None
                        if whole is not None:
                            obj_ent = _token_to_entity(whole, span_map, chunk_id, coref_map, entity_by_id)
                    if obj_ent is None and token.lemma_ in _CONCEPT_OBJECT_VERBS:
                        obj_ent = _noun_phrase_concept(o_tok, chunk_id, participle)
                    if (
                        token.lemma_ == "store"
                        and subj_ent is not None
                        and obj_ent is not None
                        and subj_ent.entity_id == obj_ent.entity_id
                    ):
                        continue
                    if obj_ent is None:
                        continue

                    triples.append(RawTriple(
                        subject=subj_ent,
                        predicate=_predicate(token),
                        object=obj_ent,
                        prep=prep_text,
                        source="svo",
                        sent_idx=sent_idx,
                        chunk_id=chunk_id,
                        dep_distance=_dep_distance(s_tok, o_tok),
                        has_negation=_has_negation(token) or (
                            token.dep_ in {"conj", "xcomp", "acl"} and _has_negation(token.head)
                        ),
                    ))
        return triples

    def _appositive(
        self, sent: Span, span_map, chunk_id, sent_idx,
        coref_map: dict[str, str] | None = None,
        entity_by_id: dict[str, Entity] | None = None,
    ) -> list[RawTriple]:
        """Appositive: 'Elon Musk, CEO of Tesla' → (Musk, appos, Tesla)."""
        triples = []
        for token in sent:
            if token.dep_ != "appos":
                continue
            head_ent  = _token_to_entity(token.head, span_map, chunk_id, coref_map, entity_by_id)
            appos_ent = _token_to_entity(token, span_map, chunk_id, coref_map, entity_by_id)
            if appos_ent is None:
                # e.g. "Tim Cook, CEO of Apple" -> CEO is appos, Apple is prep.pobj
                for child in token.children:
                    if child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                appos_ent = _token_to_entity(
                                    grandchild, span_map, chunk_id, coref_map, entity_by_id
                                )
                                break
            if head_ent and appos_ent:
                triples.append(RawTriple(
                    subject=head_ent,
                    predicate="appos",
                    object=appos_ent,
                    prep=None,
                    source="appos",
                    sent_idx=sent_idx,
                    chunk_id=chunk_id,
                    dep_distance=1,
                    has_negation=_has_negation(token),
                ))
        return triples

    def _prepositional(
        self, sent: Span, span_map, chunk_id, sent_idx,
        coref_map: dict[str, str] | None = None,
        entity_by_id: dict[str, Entity] | None = None,
    ) -> list[RawTriple]:
        """Prepositional: 'director of Tesla', 'based in California'."""
        triples = []
        for token in sent:
            if token.dep_ != "prep":
                continue
            pobj = next((c for c in token.children if c.dep_ == "pobj"), None)
            if pobj is None:
                continue
            governor = token.head
            if governor.lemma_.lower() in {"subsidiary", "adviser", "advisor"}:
                # Dedicated patterns preserve subject direction and modality.
                continue
            if governor.lemma_.lower() == "report" and governor.pos_ != "VERB":
                continue
            subject_token = None
            subject_ent_override = None
            # Reduced clauses such as "TechNova Solutions partnered with
            # IISc" are attached to the subject noun by spaCy. Use that
            # noun as the subject before walking to an unrelated ancestor.
            if (
                governor.pos_ in {"VERB", "AUX"}
                and governor.dep_ in {"acl", "relcl"}
            ):
                subject_token = governor.head

            # In "AIRI in Bengaluru" and "IISc, located in Bengaluru",
            # the location preposition can attach to a nearby verb/adjective.
            # A preceding organization is the subject of the location fact.
            if token.text.lower() in {"in", "at"} and pobj is not None and governor.lemma_ in {"locate", "base", "headquarter"}:
                preceding_orgs = [
                    entity for entity in span_map.values()
                    if entity.type in {"ORG", "PERSON"}
                    and entity.end_char <= token.idx
                ]
                if preceding_orgs:
                    nearest = max(preceding_orgs, key=lambda entity: entity.end_char)
                    if _token_to_entity(pobj, span_map, chunk_id, coref_map, entity_by_id):
                        subject_ent_override = nearest

            if governor.pos_ in {"VERB", "AUX", "ADJ"}:
                if subject_token is None:
                    subject_token = next(
                        (c for c in governor.children if c.dep_ in {"nsubj", "nsubjpass"}),
                        None,
                    )
                if subject_token is None and governor.pos_ == "ADJ":
                    subject_token = governor.head
            else:
                subject_token = governor
            if subject_token is None:
                # Nominal modifiers such as "funded the phase of DocSense"
                # attach the preposition to a noun, not to the governing verb.
                for ancestor in governor.ancestors:
                    if ancestor.pos_ == "VERB":
                        subject_token = next(
                            (c for c in ancestor.children
                             if c.dep_ in {"nsubj", "nsubjpass"}),
                            None,
                        )
                        if subject_token is not None:
                            governor = ancestor
                            break
            if subject_token is None:
                # SpaCy may attach a reduced clause ("Solutions partnered")
                # or a participle ("IISc, located") to the preceding noun.
                # Use the nearest preceding entity as the grammatical subject
                # only when the dependency tree supplies no subject.
                preceding = [
                    entity for entity in span_map.values()
                    if entity.end_char <= token.idx
                ]
                if preceding:
                    subject_ent_override = max(
                        preceding, key=lambda entity: entity.end_char
                    )
            if (
                governor.lemma_ in {"join", "partner"}
                and token.text.lower() in {"in", "at"}
                and pobj is not None
            ):
                verb_object = next(
                    (child for child in governor.children
                     if child.dep_ in {"dobj", "attr", "oprd"}),
                    None,
                )
                joined_entity = _token_to_entity(
                    verb_object, span_map, chunk_id, coref_map, entity_by_id
                ) if verb_object is not None else None
                if joined_entity is not None:
                    subject_ent_override = joined_entity
                    predicate = "locate"
                else:
                    predicate = governor.lemma_.lower()
            else:
                predicate = governor.lemma_.lower()

            if subject_ent_override is not None:
                head_ent = subject_ent_override
            elif subject_token is None:
                head_ent = subject_ent_override
            else:
                head_ent = _token_to_entity(subject_token, span_map, chunk_id, coref_map, entity_by_id)
            if head_ent is None:
                for ancestor in (governor, *governor.ancestors):
                    if ancestor.pos_ != "VERB":
                        continue
                    candidate = next(
                        (child for child in ancestor.children
                         if child.dep_ in {"nsubj", "nsubjpass"}),
                        None,
                    )
                    if candidate is not None:
                        candidate_ent = _token_to_entity(
                            candidate, span_map, chunk_id, coref_map, entity_by_id
                        )
                        if candidate_ent is not None:
                            head_ent = candidate_ent
                            subject_token = candidate
                            # "Patel leads the NLP team at Acme": the person
                            # holds that role at the organization.
                            if (
                                token.lower_ == "at"
                                and governor.dep_ == "dobj"
                                and ancestor.lemma_ in _ROLE_VERBS
                            ):
                                predicate = "role_at"
                            governor = ancestor
                            break
            subjects = [head_ent] if head_ent else []
            # "She collaborated with Patel on DocSense": Patel worked on it too.
            if head_ent and token.lower_ == "on" and governor.lemma_ in {"collaborate", "work"}:
                for sibling in governor.children:
                    if sibling.dep_ == "prep" and sibling.lower_ == "with":
                        for partner in sibling.children:
                            if partner.dep_ != "pobj":
                                continue
                            for partner_tok in (partner, *partner.conjuncts):
                                partner_ent = _token_to_entity(
                                    partner_tok, span_map, chunk_id, coref_map, entity_by_id)
                                if partner_ent is not None:
                                    subjects.append(partner_ent)
            for object_token in (pobj, *pobj.conjuncts):
                obj_ent = _token_to_entity(object_token, span_map, chunk_id, coref_map, entity_by_id)
                if obj_ent is None:
                    obj_ent = _named_by_clause(object_token, span_map, chunk_id, coref_map, entity_by_id)
                if obj_ent is None:
                    continue
                for subject_ent in subjects:
                    triples.append(RawTriple(
                        subject=subject_ent,
                        predicate=predicate,
                        object=obj_ent,
                        prep=token.text,
                        source="prep",
                        sent_idx=sent_idx,
                        chunk_id=chunk_id,
                        dep_distance=_dep_distance(subject_token or governor, object_token),
                        has_negation=_has_negation(governor),
                    ))
        return triples

    def _nominal(
        self, sent: Span, span_map, chunk_id, sent_idx,
        coref_map: dict[str, str] | None = None,
        entity_by_id: dict[str, Entity] | None = None,
    ) -> list[RawTriple]:
        """Nominal: 'Tesla founder Musk' or 'Musk, founder of Tesla'."""
        triples = []
        for token in sent:
            if token.pos_ not in {"NOUN", "PROPN"}:
                continue
            if token.lemma_.lower() not in ROLE_NOUNS:
                continue
            # Possessive, compound, nmod, or preposition child
            org_tok = next(
                (c for c in token.children if c.dep_ in {"compound", "nmod", "poss", "prep"}),
                None,
            )
            # When child is a preposition, e.g. "founder of Tesla", step down to its pobj child
            if org_tok is not None and org_tok.dep_ == "prep":
                pobj = next((gc for gc in org_tok.children if gc.dep_ == "pobj"), None)
                if pobj is not None:
                    org_tok = pobj
            if org_tok is None:
                continue

            # The head noun phrase = the person
            person_tok = token.head
            if person_tok.pos_ in ("AUX", "VERB"):
                subj = next((c for c in person_tok.children if c.dep_ in ("nsubj", "nsubjpass")), None)
                if subj is not None:
                    person_tok = subj

            org_ent    = _token_to_entity(org_tok, span_map, chunk_id, coref_map, entity_by_id)
            person_ent = _token_to_entity(person_tok, span_map, chunk_id, coref_map, entity_by_id)
            if org_ent and person_ent:
                triples.append(RawTriple(
                    subject=person_ent,
                    predicate=token.lemma_,
                    object=org_ent,
                    prep=None,
                    source="nominal",
                    sent_idx=sent_idx,
                    chunk_id=chunk_id,
                    dep_distance=_dep_distance(person_tok, org_tok),
                    has_negation=_has_negation(token),
                ))
        return triples

    def _business_patterns(self, sent, span_map, chunk_id, sent_idx, coref_map, entity_by_id):
        """Finance/role constructions with explicit lexical and type guards."""
        triples = []
        entities = sorted((e for e in span_map.values()
                           if sent.start_char <= e.start_char < sent.end_char),
                          key=lambda e: e.start_char)

        def emit(subject, predicate, obj, source="nominal", negated=False):
            if subject and obj and subject.canonical_name != obj.canonical_name:
                triples.append(RawTriple(subject, predicate, obj, None, source,
                                         sent_idx, chunk_id, 2, negated))

        def ent(token):
            return _token_to_entity(token, span_map, chunk_id, coref_map, entity_by_id)

        # Organization's role, Person / Organization role Person.
        for org in entities:
            if org.type != "ORG":
                continue
            for person in entities:
                if person.type != "PERSON" or person.start_char < org.end_char:
                    continue
                between = sent.doc.text[org.end_char:person.start_char]
                role = re.fullmatch(
                    r"(?:['\u2019]s)?\s+(?:Chief\s+(?:Executive|Financial|Technology|Operating)\s+Officer|"
                    r"(?:engineering\s+)?head|(?:independent\s+|product\s+)?director|"
                    r"founder(?:\s+and\s+CEO)?|CEO|CFO|CTO)\s*,?\s*", between, re.I)
                if role:
                    emit(person, "role_at", org)
                    if re.search(r"\bfounder\b", between, re.I):
                        emit(person, "founder", org)

        # A city-based apposition names the immediately preceding company.
        for org, place in zip(entities, entities[1:]):
            between = sent.doc.text[org.end_char:place.start_char]
            if (org.type == "ORG" and place.type == "GPE"
                    and re.fullmatch(r"[\s,()\"\u201c\u201d]*(?:a|an)?\s*", between)
                    and sent.doc.text[place.end_char:].startswith("-based")):
                emit(org, "located_at", place)

        for token in sent:
            lemma = token.lemma_.lower()
            # Axis acted as [financial adviser to Acme].
            if lemma in {"adviser", "advisor", "subsidiary"}:
                prep = next((c for c in token.children if c.dep_ == "prep" and c.lower_ in {"to", "of"}), None)
                obj_tok = next((c for c in prep.children if c.dep_ == "pobj"), None) if prep is not None else None
                governing = next((a for a in token.ancestors if a.pos_ in {"VERB", "AUX"}), None)
                subj_tok = _subject_token(governing) if governing is not None else None
                if subj_tok is not None and obj_tok is not None:
                    predicate = "adviser" if lemma != "subsidiary" else (
                        "planned_subsidiary" if _is_planned(governing) else "subsidiary")
                    emit(ent(subj_tok), predicate,
                         ent(obj_tok), negated=_has_negation(governing))
            # A loan from Bank explicitly identifies the borrower's financing.
            if lemma == "loan":
                prep = next((c for c in token.children if c.dep_ == "prep" and c.lower_ == "from"), None)
                if prep is None and token.dep_ == "dobj" and token.head.lemma_ in {"receive", "obtain", "secure", "take"}:
                    prep = next((c for c in token.head.children if c.dep_ == "prep" and c.lower_ == "from"), None)
                lender = next((c for c in prep.children if c.dep_ == "pobj"), None) if prep is not None else None
                verb = next((a for a in token.ancestors if a.pos_ == "VERB"), None)
                borrower = _subject_token(verb) if verb is not None else None
                if lender is not None and borrower is not None:
                    emit(ent(borrower), "planned_loan_from" if _is_planned(verb) else "loan_from",
                         ent(lender), negated=_has_negation(verb))
            # A named product/division in apposition to a possessed noun.
            if lemma in {"platform", "product", "division"}:
                owner = next((c for c in token.children if c.dep_ == "poss"), None)
                product = next((c for c in token.children if c.dep_ == "appos"), None)
                if owner is not None and product is not None:
                    emit(ent(owner), "has_product", ent(product))
            if lemma == "assessment":
                prep = next((c for c in token.children if c.dep_ == "prep" and c.lower_ == "of"), None)
                target = next((c for c in prep.children if c.dep_ == "pobj"), None) if prep is not None else None
                verb = token.head if token.head.lemma_ in {"perform", "conduct"} else None
                subject = _subject_token(verb) if verb is not None else None
                if target is not None and subject is not None:
                    target_ent = ent(target)
                    if target_ent is None and target.lemma_ in {"system", "account", "operation"}:
                        poss = next((c for c in target.children if c.dep_ == "poss"), None)
                        target_ent = ent(poss) if poss is not None else None
                    emit(ent(subject), "assess", target_ent, negated=_has_negation(verb))
            # Reduced passive: committee, chaired by Person.
            if lemma in {"chair", "lead", "head"} and token.dep_ in {"acl", "relcl"}:
                agent = next((c for c in token.children if c.dep_ == "agent"), None)
                actor = next((c for c in agent.children if c.dep_ == "pobj"), None) if agent is not None else None
                if actor is not None:
                    target = ent(token.head)
                    if target is None:
                        # Parser may attach the clause to "recommendation from
                        # Committee, chaired ...". Require an adjacent comma.
                        previous = [e for e in entities if e.type == "ORG" and e.end_char <= token.idx
                                    and re.fullmatch(r"\s*,\s*", sent.doc.text[e.end_char:token.idx])]
                        target = previous[-1] if previous else None
                    emit(ent(actor), "lead", target, negated=_has_negation(token))
            if lemma == "in" and token.dep_ == "prep":
                obj_tok = next((c for c in token.children if c.dep_ == "pobj"), None)
                if obj_tok is not None and token.head.lemma_ in {"centre", "center", "site", "facility"}:
                    verb = next((a for a in token.ancestors if a.lemma_ == "operate"), None)
                    subject = _subject_token(verb) if verb is not None else None
                    if subject is not None:
                        emit(ent(subject), "located_at", ent(obj_tok), negated=_has_negation(verb))
            # "Neel Shah ... will serve as its CEO": a role held at the
            # organization the possessive refers to.
            if lemma == "serve":
                as_prep = next((c for c in token.children if c.dep_ == "prep" and c.lower_ == "as"), None)
                role = next((c for c in as_prep.children if c.dep_ == "pobj"), None) if as_prep is not None else None
                owner = next((c for c in role.children if c.dep_ == "poss"), None) if role is not None else None
                holder = _subject_token(token)
                if owner is not None and holder is not None and role.lemma_.lower() in ROLE_NOUNS:
                    holder_ent = ent(holder)
                    if holder_ent is None or holder_ent.type != "PERSON":
                        # "its director, Neel Shah, will ..." names the holder in apposition.
                        named = next((c for c in holder.children if c.dep_ == "appos"), None)
                        holder_ent = ent(named) if named is not None else None
                    owner_ent = ent(owner)
                    if holder_ent and owner_ent and holder_ent.type == "PERSON" and owner_ent.type == "ORG":
                        emit(holder_ent, role.lemma_.lower(), owner_ent, negated=_has_negation(token))
                        emit(holder_ent, "role_at", owner_ent, negated=_has_negation(token))
            # Agentless passive: "These relationships are stored in Neo4j" is a
            # statement about the product currently under discussion.
            if (
                lemma == "store" and coref_map and entity_by_id
                and not any(c.dep_ == "agent" for c in token.children)
            ):
                patient = next((c for c in token.children if c.dep_ == "nsubjpass"), None)
                storage_prep = next((c for c in token.children if c.dep_ == "prep" and c.lower_ == "in"), None)
                storage = next((c for c in storage_prep.children if c.dep_ == "pobj"), None) if storage_prep is not None else None
                topic_ent = entity_by_id.get(coref_map.get(f"{chunk_id}:topic", ""))
                if (patient is not None and storage is not None and topic_ent is not None
                        and ent(patient) is None):
                    storage_ent = ent(storage)
                    if storage_ent and storage_ent.canonical_name != topic_ent.canonical_name:
                        triples.append(RawTriple(topic_ent, "store", storage_ent, "in", "prep",
                                                 sent_idx, chunk_id, 2, _has_negation(token)))
            # "It receives funding through Grant GR-5521 from the Fund": the
            # source funds the recipient, by way of the named instrument.
            if lemma in _RECEIVE_VERBS and token.pos_ == "VERB" and any(
                    c.dep_ == "dobj" and c.lemma_.lower() in _FUNDING_NOUNS for c in token.children):
                recipient = _subject_token(token)
                recipient_ent = ent(recipient) if recipient is not None else None
                by_prep = {}
                for prep in (t for t in token.subtree if t.dep_ == "prep"):
                    target = next((c for c in prep.children if c.dep_ == "pobj"), None)
                    if target is not None:
                        by_prep.setdefault(prep.lower_, ent(target))
                funder = by_prep.get("from")
                instrument = by_prep.get("through") or by_prep.get("via") or by_prep.get("under")
                negated = _has_negation(token)
                emit(funder, "fund", recipient_ent, negated=negated)
                emit(instrument, "fund", recipient_ent, negated=negated)
                emit(funder, "provide", instrument, negated=negated)
            # "Risk RSK-09 concerns delayed sensor maintenance": the parser
            # reads "concerns" as a noun inside one long noun phrase.
            if lemma == "concern" and token.pos_ == "NOUN" and token.i > sent.start and token.i < token.head.i:
                topic_ent = ent(token.nbor(-1))
                about = sent.doc[token.i + 1:token.head.i + 1]
                name = _LEADING_DETERMINER.sub("", about.text.strip(" .,;:"))
                if topic_ent is not None and len(name.split()) >= 2:
                    emit(topic_ent, "concern", Entity(
                        entity_id=f"{chunk_id}:concept:{about.start_char}:{about.end_char}",
                        text=about.text, canonical_name=name, type="CONCEPT",
                        start_char=about.start_char, end_char=about.end_char,
                        source="noun_phrase", chunk_id=chunk_id, sentence=sent.doc.text))
            if lemma == "shareholder":
                owner = next((c for c in token.children if c.dep_ == "poss"), None)
                if token.dep_ == "nsubj" and owner is not None:
                    holder = next((c for c in token.head.children if c.dep_ == "attr"), None)
                    if holder is not None:
                        emit(ent(holder), "own", ent(owner), negated=_has_negation(token.head))
        return triples
