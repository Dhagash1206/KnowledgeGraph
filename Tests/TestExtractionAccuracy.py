"""Regression tests for the extraction accuracy fixes, checked on AiResearchReport.pdf."""
from pathlib import Path

import pytest

from KnowledgeGraphApp.ExtractionPipeline import run_pipeline_nlp
from KnowledgeGraphApp.GraphStorage.GraphBuilder import build_resolved_edges
from KnowledgeGraphApp.Ingestion.PdfCleaner import clean_pdf_text
from KnowledgeGraphApp.Ingestion.Preprocessor import preprocess

REPORT_TEXT = (
    "AI Research Collaboration Report\n\n"
    "On 15 March 2025, Dr. Ananya Sharma joined the Artificial Intelligence Research "
    "Institute (AIRI) in Bengaluru as a senior researcher. AIRI was founded by Professor "
    "Rahul Mehta in 2018 and focuses on natural language processing, computer vision, and "
    "robotics. Ananya Sharma previously worked at TechNova Solutions, a technology company "
    "headquartered in Mumbai. There, she collaborated with Dr. Vikram Patel on a project "
    "called DocSense, which automatically extracts entities and relationships from "
    "unstructured documents. Vikram Patel leads the NLP team at TechNova Solutions. In June "
    "2024, TechNova Solutions partnered with the Indian Institute of Science (IISc), located "
    "in Bengaluru, to develop better information extraction systems. Professor Rahul Mehta "
    "represented AIRI during the partnership negotiations. IISc provided the computing "
    "infrastructure, while TechNova Solutions contributed its document-processing software. "
    "AIRI funded the initial phase of the project with 50 lakh. "
    "The DocSense project uses named entity recognition to identify people and organizations. "
    "These extracted relationships are stored in Neo4j, where researchers can query the graph. "
    "Ananya Sharma will lead the next phase of the project. She will work with Vikram Patel "
    "and Rahul Mehta to evaluate the system on scientific papers and business reports."
)


def _graph(text: str):
    result = run_pipeline_nlp(text.encode("utf-8"), "report.txt", "txt")
    entities = {entity.name: entity.type for entity in result["resolved_entities"]}
    names = {entity.canonical_id: entity.name for entity in result["resolved_entities"]}
    edges = {
        (names[edge.source_id], edge.relation_type, names[edge.target_id])
        for edge in build_resolved_edges(result["normalized"], result["id_map"])
    }
    return entities, edges


def test_report_entities_are_typed_correctly():
    entities, _ = _graph(REPORT_TEXT)

    assert "AI Research Collaboration Report" not in entities  # title is not an entity
    assert "NLP" not in entities                               # a field, not an organization
    assert "Ananya" not in entities                            # part of a person's full name
    assert entities["Ananya Sharma"] == "PERSON"
    assert entities["DocSense"] == "PRODUCT"                   # "a project called DocSense"
    assert entities["NLP team"] == "ORG"
    assert entities["computing infrastructure"] == "CONCEPT"   # common noun phrase, not a product
    assert "PRODUCT" not in {
        entities.get("the next phase"), entities.get("better information extraction systems")}


def test_report_relations():
    _, edges = _graph(REPORT_TEXT)

    assert {
        ("Ananya Sharma", "MEMBER_OF", "Artificial Intelligence Research Institute"),
        ("Rahul Mehta", "REPRESENTED", "Artificial Intelligence Research Institute"),
        ("Vikram Patel", "EMPLOYED_BY", "TechNova Solutions"),
        ("Ananya Sharma", "WORKED_ON", "DocSense"),
        ("Vikram Patel", "WORKED_ON", "DocSense"),
        ("Ananya Sharma", "COLLABORATED_WITH", "Rahul Mehta"),
        ("TechNova Solutions", "PROVIDED", "document-processing software"),
        # "the project" is the product the document introduced as a project.
        ("Artificial Intelligence Research Institute", "FUNDED", "DocSense"),
        ("Ananya Sharma", "LEADS", "DocSense"),
        # An agentless passive is attributed to the product under discussion.
        ("DocSense", "STORES_DATA_IN", "Neo4j"),
    } <= edges
    # A person representing their own institute does not advise it.
    assert not any(relation == "ADVISER_TO" for _, relation, _ in edges)
    # "partnered ... to develop better systems" states a goal, not a result.
    assert not any(relation == "DEVELOPED" for _, relation, _ in edges)
    assert not any(relation == "RELATED_TO" for _, relation, _ in edges)


def test_law_firm_representing_a_company_is_still_an_adviser():
    text = ("Desai Rao Associates represented Meridian Analytics Ltd. in the sale. "
            "Meridian Analytics Ltd. is based in Pune.")
    _, edges = _graph(text)

    assert ("Desai Rao Associates", "ADVISER_TO", "Meridian Analytics Ltd") in edges


def test_pdf_title_stays_on_its_own_line():
    raw = "AI Research Collaboration Report\nOn 15 March 2025, Dr. Sharma joined the\ninstitute in Bengaluru."

    cleaned = clean_pdf_text(raw)

    assert cleaned.startswith("AI Research Collaboration Report\n\nOn 15 March 2025")
    # An ordinary wrapped first sentence is not mistaken for a title.
    wrapped = "the institute was founded in 2018 and\nit focuses on robotics."
    assert "\n" not in clean_pdf_text(wrapped)


def test_sentences_do_not_span_paragraph_breaks():
    text = "AI Research Collaboration Report\n\nDr. Ananya Sharma joined the institute in Bengaluru."

    document = preprocess(text.encode("utf-8"), "report.txt")

    assert [chunk.text for chunk in document.chunks][0] == "AI Research Collaboration Report"
    for chunk in document.chunks:
        assert document.raw_text[chunk.start_char:chunk.end_char] == chunk.text


def test_sentence_final_period_is_not_part_of_an_organization_name():
    text = ("Northstar Systems Ltd. is listed on the exchange under the ticker NSTAR. "
            "Northstar Systems Ltd. expects a loan from Union National Bank.")
    entities, _ = _graph(text)

    assert "Union National Bank" in entities
    result = run_pipeline_nlp(text.encode("utf-8"), "loan.txt", "txt")
    surfaces = {entity.text for entity in result["all_entities"]}
    assert "Union National Bank." not in surfaces
    assert "NSTAR." not in surfaces


def test_sample_report_pdf_has_no_mistyped_entities():
    path = Path("Docs/Samples/AiResearchReport.pdf")
    if not path.exists():
        pytest.skip("Sample PDF is kept locally and is not part of the repository")
    result = run_pipeline_nlp(path.read_bytes(), path.name, "pdf")
    entities = {entity.name: entity.type for entity in result["resolved_entities"]}

    assert entities == {
        "Ananya Sharma": "PERSON", "Rahul Mehta": "PERSON", "Vikram Patel": "PERSON",
        "Artificial Intelligence Research Institute": "ORG", "TechNova Solutions": "ORG",
        "Indian Institute Of Science": "ORG", "NLP team": "ORG",
        "Bengaluru": "GPE", "Mumbai": "GPE",
        "DocSense": "PRODUCT", "Neo4j": "PRODUCT", "named entity recognition": "CONCEPT",
        "computing infrastructure": "CONCEPT", "document-processing software": "CONCEPT",
    }


def test_definite_product_reference_needs_a_matching_introduction():
    text = ("Acme Analytics Ltd. sells a platform called DataHub in Pune. "
            "Acme Analytics Ltd. funded the first phase of the project.")
    _, edges = _graph(text)

    # DataHub was introduced as a platform, so "the project" is not DataHub.
    assert not any(relation == "FUNDED" for _, relation, _ in edges)


def test_reporting_verb_is_not_a_relation():
    text = ("Ananya Rao is the Chief Executive Officer of Northstar Systems Ltd. "
            "Rao described Meridian Analytics Ltd. as a strong fit.")
    _, edges = _graph(text)

    assert not any({source, target} == {"Ananya Rao", "Meridian Analytics Ltd"}
                   for source, _, target in edges)


def test_acquisition_pdf_firm_typing_and_future_role():
    path = Path("Docs/Samples/AcquisitionCaseStudy.pdf")
    if not path.exists():
        pytest.skip("Sample PDF is kept locally and is not part of the repository")
    result = run_pipeline_nlp(path.read_bytes(), path.name, "pdf")
    entities = {entity.name: entity.type for entity in result["resolved_entities"]}
    names = {entity.canonical_id: entity.name for entity in result["resolved_entities"]}
    edges = {(names[edge.source_id], edge.relation_type, names[edge.target_id])
             for edge in build_resolved_edges(result["normalized"], result["id_map"])}

    assert entities["Grant Thornton Bharat"] == "ORG"   # conducted due diligence: a firm
    assert ("Neel Shah", "LEADS", "Pocket Labs Pvt. Ltd") in edges   # "serve as its CEO"
    assert not any(relation == "RELATED_TO" for _, relation, _ in edges)


def test_product_named_by_kind_noun_uses_a_technique():
    _, edges = _graph(REPORT_TEXT)

    # "The DocSense project uses named entity recognition ..."
    assert ("DocSense", "USES", "named entity recognition") in edges


def test_elided_ownership_refers_to_the_company_just_named():
    text = ("Cedar Peak Ventures LLP owns 22% of Meridian Analytics Ltd. in Pune. "
            "Rohan Mehta owns 51%, and Kavya Iyer owns 17%.")
    _, edges = _graph(text)

    assert ("Rohan Mehta", "OWNS", "Meridian Analytics Ltd") in edges
    assert ("Kavya Iyer", "OWNS", "Meridian Analytics Ltd") in edges


def test_elided_ownership_needs_a_recent_antecedent():
    text = ("Rohan Mehta works at Meridian Analytics Ltd. in Pune. "
            "Rohan Mehta owns 51%.")
    _, edges = _graph(text)

    assert not any(relation == "OWNS" for _, relation, _ in edges)


def test_acquisition_pdf_context_relations():
    path = Path("Docs/Samples/AcquisitionCaseStudy.pdf")
    if not path.exists():
        pytest.skip("Sample PDF is kept locally and is not part of the repository")
    result = run_pipeline_nlp(path.read_bytes(), path.name, "pdf")
    names = {entity.canonical_id: entity.name for entity in result["resolved_entities"]}
    edges = {(names[edge.source_id], edge.relation_type, names[edge.target_id])
             for edge in build_resolved_edges(result["normalized"], result["id_map"])}

    assert {
        ("Northstar Systems Ltd", "REQUIRES_APPROVAL_FROM", "Competition Commission Of India"),
        ("Grant Thornton Bharat", "ASSESSED", "Meridian Analytics Pvt. Ltd"),
        ("Rohan Mehta", "OWNS", "Meridian Analytics Pvt. Ltd"),
        ("Kavya Iyer", "OWNS", "Meridian Analytics Pvt. Ltd"),
        ("Neel Shah", "OWNS", "Pocket Labs Pvt. Ltd"),
    } <= edges
    linked = {name for source, _, target in edges for name in (source, target)}
    assert set(names.values()) <= linked   # no isolated entities
