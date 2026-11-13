"""Regression tests for record identifiers, checked on meridian_relational_data_story.pdf."""
from KnowledgeGraphApp.ExtractionPipeline import run_pipeline_nlp
from KnowledgeGraphApp.GraphStorage.GraphBuilder import build_resolved_edges

MERIDIAN_TEXT = (
    "The Meridian Research Consortium\n\n"
    "Meridian Research Consortium is a partnership of universities, laboratories, and funding "
    "agencies running a five-year programme to improve flood forecasting across western India. "
    "The programme is coordinated by Meridian Institute of Technology (INS-01), which contains "
    "departments identified by a Department ID, name, and campus location. The Centre for "
    "Environmental Computing (DEP-12) belongs to INS-01. Other partners are Coastal State "
    "University (INS-02) and the National Hydrology Laboratory (INS-03). "
    "Dr. Asha Mehta (RES-104) works in DEP-12 and leads the forecasting workstream. She also "
    "collaborates with INS-02. "
    "The Projects table contains RiverWatch (PRJ-220), recording its start date, planned "
    "completion date, budget, status, and lead institution. It receives funding through Grant "
    "GR-5521 from the Western Resilience Fund (FUN-08). RiverWatch contains work packages. "
    "WP-03, Sensor Data Integration, is assigned to DEP-12 and has a milestone date, "
    "deliverable, and completion status. "
    "Dataset DS-77 contains cleaned rainfall observations and records its owner, version, "
    "creation date, licence, and source stations. Model MOD-14 predicts flood risk. "
    "Meeting MT-31 stores its date, location, agenda, and minutes. "
    "Publication PUB-19 stores a title, date, DOI, and venue. "
    "Risk RSK-09 concerns delayed sensor maintenance."
)


def _graph(text: str):
    result = run_pipeline_nlp(text.encode("utf-8"), "story.txt", "txt")
    entities = {entity.name: entity for entity in result["resolved_entities"]}
    names = {entity.canonical_id: entity.name for entity in result["resolved_entities"]}
    edges = {
        (names[edge.source_id], edge.relation_type, names[edge.target_id])
        for edge in build_resolved_edges(result["normalized"], result["id_map"])
    }
    return entities, edges


def test_identifier_is_an_alias_of_the_record_it_names():
    entities, _ = _graph(MERIDIAN_TEXT)

    # No node for a bare prefix or a bare code that has a name.
    assert not {"INS", "DEP", "RES", "PRJ", "FUN", "WP", "GR", "DS", "MOD", "MT", "PUB", "RSK"} & set(entities)
    assert "INS-01" in entities["Meridian Institute Of Technology"].aliases
    assert "INS-02" in entities["Coastal State University"].aliases
    assert "INS-03" in entities["National Hydrology Laboratory"].aliases
    assert "DEP-12" in entities["Centre For Environmental Computing"].aliases
    assert "RES-104" in entities["Asha Mehta"].aliases
    assert "WP-03" in entities["Sensor Data Integration"].aliases  # "ID, Name," order


def test_record_types_and_schema_attributes():
    entities, _ = _graph(MERIDIAN_TEXT)

    assert entities["RiverWatch"].type == "PRODUCT"              # PRJ prefix: a project
    assert entities["Sensor Data Integration"].type == "CONCEPT"  # a work package, not an org
    assert entities["Grant GR-5521"].type == "CONCEPT"           # kind word + ID is the name
    assert entities["Dataset DS-77"].type == "PRODUCT"
    assert entities["Meeting MT-31"].type == "EVENT"
    assert entities["Asha Mehta"].type == "PERSON"
    # Column names and generic acronyms are not entities.
    assert not {"Department ID", "DOI", "Dataset", "a title", "title"} & set(entities)


def test_record_relations_are_named():
    _, edges = _graph(MERIDIAN_TEXT)

    assert {
        ("Centre For Environmental Computing", "PART_OF", "Meridian Institute Of Technology"),
        ("Asha Mehta", "EMPLOYED_BY", "Centre For Environmental Computing"),
        ("Asha Mehta", "COLLABORATED_WITH", "Coastal State University"),
        ("Asha Mehta", "LEADS", "forecasting workstream"),
        ("Western Resilience Fund", "FUNDED", "RiverWatch"),            # "It" is the project
        ("Grant GR-5521", "FUNDED", "RiverWatch"),
        ("Western Resilience Fund", "PROVIDED", "Grant GR-5521"),
        ("Sensor Data Integration", "ASSIGNED_TO", "Centre For Environmental Computing"),
        ("RiverWatch", "CONTAINS", "Sensor Data Integration"),          # WP = "work packages"
        ("Dataset DS-77", "CONTAINS", "cleaned rainfall observations"),
        ("Model MOD-14", "PREDICTS", "flood risk"),
        ("Risk RSK-09", "CONCERNS", "delayed sensor maintenance"),
        ("Meridian Institute Of Technology", "COORDINATES", "Meridian Research Consortium"),
        ("Coastal State University", "MEMBER_OF", "Meridian Research Consortium"),
        ("National Hydrology Laboratory", "MEMBER_OF", "Meridian Research Consortium"),
    } <= edges
    # The class-level "work packages" edge is replaced by the specific record.
    assert ("RiverWatch", "CONTAINS", "work packages") not in edges
    assert sum(1 for _, relation, _ in edges if relation == "RELATED_TO") <= 1


def test_unnamed_identifier_is_kept_under_its_code():
    entities, edges = _graph(
        "Status Report\n\nAcme Robotics is a company based in Pune. "
        "Priya Nair works in Acme Robotics. Ticket TSK-44 is assigned to Acme Robotics. "
        "The team later reviewed ZX-9 again."
    )

    assert entities["Ticket TSK-44"].type == "CONCEPT"
    assert "ZX-9" in entities
    assert ("Ticket TSK-44", "ASSIGNED_TO", "Acme Robotics") in edges
    assert ("Priya Nair", "EMPLOYED_BY", "Acme Robotics") in edges
