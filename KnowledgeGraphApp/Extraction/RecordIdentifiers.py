"""Record identifiers such as "INS-01" or "PRJ-220" in database-style prose.

An identifier is a second name for a record, not a separate thing:

  - "Meridian Institute of Technology (INS-01)"  the ID names the entity before it
  - "WP-03, Sensor Data Integration, is ..."      the ID names the entity after it
  - "Grant GR-5521", "Dataset DS-77"              kind word + ID is the record's name
  - "... belongs to INS-01."                      a later bare ID refers to that record

Every mention of an identifier is given its record's canonical name, so entity
resolution merges them into one node and the ID survives as an alias.
"""
from __future__ import annotations

from dataclasses import replace
import re

from KnowledgeGraphApp.Extraction.EntityRecognizer import Entity, _canonical_name
from KnowledgeGraphApp.Extraction.PredicateNormalizer import NormalizedTriple

SOURCE = "record_identifier"
EXTRACTOR = "discourse"
CONTAINED_RECORD_CONFIDENCE = 0.65
# "RiverWatch contains work packages. WP-03 ..." links records this far ahead.
CONTAINER_SENTENCE_SPAN = 2

RECORD_ID = re.compile(r"(?<![\w-])([A-Z]{2,5})-(\d{1,6})(?![\w-])")

# The kind word in front of an identifier says what the record is.
KIND_TYPES: dict[str, str] = {
    "grant": "CONCEPT", "risk": "CONCEPT", "task": "CONCEPT", "ticket": "CONCEPT",
    "milestone": "CONCEPT", "contract": "CONCEPT", "invoice": "CONCEPT", "order": "CONCEPT",
    "dataset": "PRODUCT", "model": "PRODUCT", "project": "PRODUCT", "publication": "PRODUCT",
    "report": "PRODUCT", "sensor": "PRODUCT", "product": "PRODUCT",
    "meeting": "EVENT", "event": "EVENT", "workshop": "EVENT",
    "station": "GPE", "site": "GPE",
    "department": "ORG", "institution": "ORG", "agency": "ORG", "team": "ORG",
    "researcher": "PERSON", "employee": "PERSON", "student": "PERSON",
}
# Conventional prefixes, used only when no kind word is present.
PREFIX_TYPES: dict[str, str] = {
    "INS": "ORG", "ORG": "ORG", "DEP": "ORG", "DEPT": "ORG", "FUN": "ORG",
    "RES": "PERSON", "EMP": "PERSON", "STU": "PERSON",
    "PRJ": "PRODUCT", "PROJ": "PRODUCT", "DS": "PRODUCT", "MOD": "PRODUCT", "PUB": "PRODUCT",
    "WP": "CONCEPT", "GR": "CONCEPT", "RSK": "CONCEPT", "TSK": "CONCEPT",
    "MT": "EVENT", "MTG": "EVENT", "EVT": "EVENT",
}
DEFAULT_RECORD_TYPE = "CONCEPT"

_KIND_BEFORE = re.compile(r"\b([A-Za-z]+)\s+$")
_NAME_WORD = r"[A-Z][\w&.'-]*"
_NAME_BEFORE_PAREN = re.compile(
    rf"({_NAME_WORD}(?:\s+(?:{_NAME_WORD}|of|for|the|and))*\s+{_NAME_WORD}|{_NAME_WORD})\s*\($")
_NAME_AFTER_COMMA = re.compile(
    rf"^,\s+((?:{_NAME_WORD})(?:\s+(?:{_NAME_WORD}|of|for|the|and))*),")
# "Department ID" or "sensor ID" names a column, not a thing in the world.
_ATTRIBUTE_NAME = re.compile(r"\b(?:IDs?|Id)$")
_CONTAINS = re.compile(r"^\s+(?:contains?|has|have|includes?|comprises?)\s+([a-z][a-z -]*[a-z])[.,;]")


def _owner_of(chunk, identifier: re.Match, entities: list[Entity]):
    """Return (start, end, existing entity or None) of the name an ID belongs to."""
    before, after = chunk.text[:identifier.start()], chunk.text[identifier.end():]
    if before.endswith("(") and after.startswith(")"):
        paren = identifier.start() - 1
        named = [e for e in entities if e.end_char <= paren and not chunk.text[e.end_char:paren].strip()]
        if named:
            owner = max(named, key=lambda e: e.end_char - e.start_char)
            return owner.start_char, owner.end_char, owner
        match = _NAME_BEFORE_PAREN.search(before)
        if match:
            return match.start(1), match.end(1), None
    match = _NAME_AFTER_COMMA.match(after)
    if match:
        start, end = identifier.end() + match.start(1), identifier.end() + match.end(1)
        named = [e for e in entities if e.start_char == start]
        return start, end, (named[0] if named else None)
    return None


def resolve_record_identifiers(chunks, entities_by_chunk: dict[str, list[Entity]]) -> dict[str, list[Entity]]:
    """Fold identifier mentions into the records they name."""
    records: dict[str, tuple[str, str]] = {}   # "INS-01" -> (canonical name, type)
    explicit: dict[tuple[str, int], tuple[int, int]] = {}  # (chunk, ID start) -> mention span

    def make(chunk, start: int, end: int, canonical: str, kind: str) -> Entity:
        return Entity(f"{chunk.chunk_id}:{start}:{end}", chunk.text[start:end], canonical, kind,
                      start, end, SOURCE, chunk.chunk_id, chunk.text)

    # Pass 1: learn what each identifier names from the sentences that define it.
    for chunk in chunks:
        entities = entities_by_chunk.get(chunk.chunk_id, [])
        for identifier in RECORD_ID.finditer(chunk.text):
            code, prefix = identifier.group(), identifier.group(1)
            if code in records:
                continue
            kind_word = _KIND_BEFORE.search(chunk.text[:identifier.start()])
            if kind_word and kind_word.group(1).lower() in KIND_TYPES:
                span = (kind_word.start(1), identifier.end())
                records[code] = (_canonical_name(chunk.text[span[0]:span[1]]),
                                 KIND_TYPES[kind_word.group(1).lower()])
                explicit[(chunk.chunk_id, identifier.start())] = span
                continue
            owner = _owner_of(chunk, identifier, entities)
            if owner is not None:
                start, end, entity = owner
                name = entity.canonical_name if entity else _canonical_name(chunk.text[start:end])
                fallback = entity.type if entity else DEFAULT_RECORD_TYPE
                records[code] = (name, PREFIX_TYPES.get(prefix, fallback))
    # An identifier nobody names is still a record, known only by its code.
    for chunk in chunks:
        for identifier in RECORD_ID.finditer(chunk.text):
            records.setdefault(identifier.group(), (
                identifier.group(), PREFIX_TYPES.get(identifier.group(1), DEFAULT_RECORD_TYPE)))

    record_type_by_name = {name: kind for name, kind in records.values()}

    # Pass 2: rewrite the mentions.
    resolved: dict[str, list[Entity]] = {}
    for chunk in chunks:
        entities = entities_by_chunk.get(chunk.chunk_id, [])
        mentions: list[Entity] = []
        for identifier in RECORD_ID.finditer(chunk.text):
            name, kind = records[identifier.group()]
            start, end = explicit.get((chunk.chunk_id, identifier.start()), identifier.span())
            # "grant GR-5521" keeps its kind word wherever it is repeated.
            kind_word = _KIND_BEFORE.search(chunk.text[:identifier.start()])
            if kind_word and _canonical_name(chunk.text[kind_word.start(1):identifier.end()]) == name:
                start = kind_word.start(1)
            mentions.append(make(chunk, start, end, name, kind))
            owner = _owner_of(chunk, identifier, entities)
            if owner is not None and owner[2] is None:
                mentions.append(make(chunk, owner[0], owner[1], name, kind))
        kept = [
            replace(entity, type=record_type_by_name.get(entity.canonical_name, entity.type))
            for entity in entities
            if not any(entity.start_char < m.end_char and entity.end_char > m.start_char for m in mentions)
            and not _ATTRIBUTE_NAME.search(entity.text)
        ]
        resolved[chunk.chunk_id] = sorted(mentions + kept, key=lambda e: e.start_char)
    return resolved


def _initials(phrase: str) -> str:
    return "".join(word[0] for word in re.split(r"[\s-]+", phrase) if word).upper()


def link_contained_records(chunks, entities_by_chunk, normalized: list[NormalizedTriple]) -> list[NormalizedTriple]:
    """Attach records to the entity said to contain their kind.

    "RiverWatch contains work packages. WP-03, Sensor Data Integration, ..."
    The prefix WP abbreviates "work packages", so WP-03 is one of them. The
    class-level edge to "work packages" is then replaced by the specific one.
    """
    linked: list[NormalizedTriple] = []
    replaced: set[tuple[str, str]] = set()
    for position, chunk in enumerate(chunks):
        for container in entities_by_chunk.get(chunk.chunk_id, []):
            match = _CONTAINS.match(chunk.text[container.end_char:])
            if match is None:
                continue
            prefix = _initials(match.group(1))
            for later in chunks[position + 1:position + 1 + CONTAINER_SENTENCE_SPAN]:
                for record in entities_by_chunk.get(later.chunk_id, []):
                    code = RECORD_ID.search(record.text)
                    if (record.source != SOURCE or code is None or code.group(1) != prefix
                            or record.canonical_name == container.canonical_name):
                        continue
                    linked.append(NormalizedTriple(
                        relation_type="CONTAINS", subject=container, object=record,
                        raw_predicate="contain", chunk_id=later.chunk_id, sent_idx=later.sent_idx,
                        confidence=CONTAINED_RECORD_CONFIDENCE,
                        evidence=f"{chunk.text} {later.text}", extractor=EXTRACTOR,
                    ))
                    replaced.add((container.canonical_name, _canonical_name(match.group(1)).casefold()))
    return [
        triple for triple in normalized
        if (triple.subject.canonical_name, triple.object.canonical_name.casefold()) not in replaced
    ] + linked
