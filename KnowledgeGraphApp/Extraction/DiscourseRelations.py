"""Relations whose second entity is stated in a nearby sentence, not the same clause.

Sentence-level dependency patterns cannot see these. Each rule here needs an
explicit textual trigger and an unambiguous antecedent; when the antecedent is
missing or ambiguous, no relation is produced.

  - Elliptical ownership:  "Cedar owns 22% of Meridian. Mehta owns 51%."
  - Approval condition:    "Completion is conditional on approval from the CCI."
  - Deal due diligence:    "Grant Thornton conducted financial due diligence."
  - Consortium membership: "X is a partnership of ... Other partners are A and B."
  - Programme coordinator: "The programme is coordinated by A."
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


# "Meridian Research Consortium is a partnership of universities ..."
_GROUP_DEFINITION = re.compile(
    r"^\s+is\s+an?\s+(?:partnership|consortium|alliance|coalition|network|association)\s+of\b",
    re.IGNORECASE,
)
_PARTNERS_ARE = re.compile(r"\b(?:partners|members)\s+(?:are|include)\s+(?:the\s+)?$", re.IGNORECASE)
_RUN_BY = re.compile(
    r"\b[Tt]he\s+(?P<noun>programme|program|initiative|project|consortium|partnership)\s+is\s+"
    r"(?P<verb>coordinated|led|managed|run|administered)\s+by\s+(?:the\s+)?$"
)
GROUP_CONFIDENCE = 0.65


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

    # The document's one consortium-like body, and the sentence defining it.
    groups = [(entity, chunk.text) for chunk in chunks
              for entity in entities_by_chunk.get(chunk.chunk_id, [])
              if entity.type == "ORG" and _GROUP_DEFINITION.match(chunk.text[entity.end_char:])]
    group = _only([entity for entity, _ in groups]) if groups else None
    group_definition = groups[0][1].lower() if groups else ""

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

        if group is None:
            continue
        organizations = [e for e in entities if e.type == "ORG" and e.canonical_name != group.canonical_name]
        for entity in organizations:
            before = chunk.text[:entity.start_char]
            if _PARTNERS_ARE.search(before):
                # The listed partners are this and every later organization.
                members = {e.canonical_name: e for e in organizations if e.start_char >= entity.start_char}
                inferred.extend(_triple("MEMBER_OF", member, group, chunk, "partner", GROUP_CONFIDENCE)
                                for member in members.values())
            run_by = _RUN_BY.search(before)
            # "The programme" is the group's own only if its definition names one.
            if run_by and (run_by["noun"].lower() in {"consortium", "partnership"}
                           or run_by["noun"].lower() in group_definition):
                relation = "COORDINATES" if run_by["verb"].lower() in {"coordinated", "administered"} else "LEADS"
                inferred.append(_triple(relation, entity, group, chunk, run_by["verb"].lower(), GROUP_CONFIDENCE))
    return inferred
