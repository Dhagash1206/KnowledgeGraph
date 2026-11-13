"""Relations whose second entity is stated in a nearby sentence, not the same clause.

Sentence-level dependency patterns cannot see these. Each rule here needs an
explicit textual trigger and an unambiguous antecedent; when the antecedent is
missing or ambiguous, no relation is produced.

  - Elliptical ownership:  "Cedar owns 22% of Meridian. Mehta owns 51%."
  - Approval condition:    "Completion is conditional on approval from the CCI."
  - Deal due diligence:    "Grant Thornton conducted financial due diligence."
"""
from __future__ import annotations

import re

from KnowledgeGraphApp.Extraction.EntityRecognizer import Entity
from KnowledgeGraphApp.Extraction.PredicateNormalizer import NormalizedTriple

EXTRACTOR = "discourse"
# Stated facts with an inferred argument rank below directly parsed ones.
ELLIPSIS_CONFIDENCE = 0.75
DEAL_CONFIDENCE = 0.65
# An elided "of <company>" refers back at most this many sentences.
OWNERSHIP_SENTENCE_SPAN = 2

_PERCENT = r"\d+(?:\.\d+)?\s?%"
_OWNS_OF = re.compile(rf"\b(?:owns?|holds?)\s+{_PERCENT}\s+of\s+(?:the\s+)?$", re.IGNORECASE)
_OWNS_ELIDED = re.compile(rf"^,?\s*(?:will\s+)?(?:owns?|holds?)\s+{_PERCENT}(?!\s+of\b)", re.IGNORECASE)
_APPROVAL_FROM = re.compile(
    r"\b(?:conditional\s+(?:up)?on|subject\s+to|requires?|pending|awaits?|awaiting)\s+"
    r"(?:[\w-]+\s+){0,3}?(?:approval|clearance|consent)\s+(?:from|of|by)\s+(?:the\s+)?$",
    re.IGNORECASE,
)
_DUE_DILIGENCE = re.compile(
    r"^\s+(?:conducted|performed|carried\s+out|completed)\s+(?:[\w-]+\s+){0,3}?due\s+diligence\b"
    r"(?!\s+(?:on|of)\b)",
    re.IGNORECASE,
)


def _triple(relation: str, subject: Entity, obj: Entity, chunk, predicate: str,
            confidence: float) -> NormalizedTriple:
    return NormalizedTriple(
        relation_type=relation, subject=subject, object=obj, raw_predicate=predicate,
        chunk_id=chunk.chunk_id, sent_idx=chunk.sent_idx, confidence=confidence,
        evidence=chunk.text, extractor=EXTRACTOR,
    )


def _only(entities: list[Entity]) -> Entity | None:
    """The entity, if all candidates name the same thing; None when ambiguous."""
    return entities[0] if len({entity.canonical_name for entity in entities}) == 1 else None


def infer_discourse_relations(chunks, entities_by_chunk, normalized) -> list[NormalizedTriple]:
    """Return relations completed from document context."""
    acquirers = [t.subject for t in normalized if t.relation_type in {"PLANS_TO_ACQUIRE", "ACQUIRED"}]
    targets = [t.object for t in normalized if t.relation_type in {"PLANS_TO_ACQUIRE", "ACQUIRED"}]
    acquirer, target = _only(acquirers) if acquirers else None, _only(targets) if targets else None

    inferred: list[NormalizedTriple] = []
    owned: Entity | None = None
    owned_position = -1
    for position, chunk in enumerate(chunks):
        entities = sorted(entities_by_chunk.get(chunk.chunk_id, []), key=lambda e: e.start_char)
        for entity in entities:
            before, after = chunk.text[:entity.start_char], chunk.text[entity.end_char:]

            if entity.type == "ORG" and _OWNS_OF.search(before):
                owned, owned_position = entity, position
            elif (
                entity.type in {"PERSON", "ORG"} and _OWNS_ELIDED.match(after)
                and owned is not None and position - owned_position <= OWNERSHIP_SENTENCE_SPAN
                and entity.canonical_name != owned.canonical_name
            ):
                inferred.append(_triple("OWNS", entity, owned, chunk, "own", ELLIPSIS_CONFIDENCE))

            if entity.type == "ORG" and acquirer is not None and _APPROVAL_FROM.search(before):
                if entity.canonical_name != acquirer.canonical_name:
                    inferred.append(_triple("REQUIRES_APPROVAL_FROM", acquirer, entity, chunk,
                                            "approval", DEAL_CONFIDENCE))

            # In a document about one acquisition, unqualified due diligence
            # is the review of the company being acquired.
            if entity.type == "ORG" and target is not None and _DUE_DILIGENCE.match(after):
                if entity.canonical_name not in {target.canonical_name,
                                                 acquirer.canonical_name if acquirer else ""}:
                    inferred.append(_triple("ASSESSED", entity, target, chunk,
                                            "due_diligence", DEAL_CONFIDENCE))
    return inferred
