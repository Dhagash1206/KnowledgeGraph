"""Regression tests for the fixes listed in CodeReview.txt (H1-H5, M2, M4, M5, L1, L2)."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from KnowledgeGraphApp import ExtractionPipeline
from KnowledgeGraphApp.Extraction import ClassicalNews, StanfordClient
from KnowledgeGraphApp.Extraction.LexicalSvm import score
from KnowledgeGraphApp.GraphSearch.QueryEngine import QueryEngine
from KnowledgeGraphApp.GraphStorage.GraphBuilder import build_resolved_entities
from KnowledgeGraphApp.Ingestion.PdfCleaner import remove_headers_footers, strip_tables
from KnowledgeGraphApp.Settings import Settings, get_settings
from KnowledgeGraphApp.WebApi.App import app
from KnowledgeGraphApp.WebApi.Routes import GraphRoutes, SearchRoutes
from KnowledgeGraphApp.WebApi.Schemas import ResolvedEdgePayload, ResolvedEntityPayload

NEWS_TEXT = "Satya Nadella works for Microsoft. Microsoft is based in Seattle."


def _news_annotation(text: str) -> dict:
    """Fake Stanford output; sentence indexes deliberately disagree with spaCy's (H2)."""
    def mention(mention_id, entity_type, surface, sentence_index):
        start = text.index(surface) if mention_id != "m3" else text.rindex(surface)
        return {
            "mention_id": mention_id, "type": entity_type, "text": surface,
            "char_start": start, "char_end": start + len(surface),
            "mapped_text": surface, "sentence_index": sentence_index,
        }
    return {
        "sentences": [{"index": 0, "text": text, "char_start": 0, "char_end": len(text)}],
        "entities": [
            mention("m1", "PEOPLE", "Satya Nadella", 7),
            mention("m2", "ORGANIZATION", "Microsoft", 7),
            mention("m3", "ORGANIZATION", "Microsoft", 7),
            mention("m4", "LOCATION", "Seattle", 7),
        ],
        "relations": [
            {"label": "Work_For", "sentence_index": 0, "probabilities": {"Work_For": 0.9},
             "arg0": {"object_id": "m1"}, "arg1": {"object_id": "m2"}},
            {"label": "OrgBased_In", "sentence_index": 0, "probabilities": {"OrgBased_In": 0.8},
             "arg0": {"object_id": "m3"}, "arg1": {"object_id": "m4"}},
        ],
    }


@pytest.fixture
def classical_news_settings(monkeypatch, tmp_path):
    settings = get_settings()
    monkeypatch.setattr(settings, "relation_backend", "classical_news")
    monkeypatch.setattr(settings, "intermediate_graph_dir", str(tmp_path))
    fake_client = MagicMock()
    fake_client.is_available.return_value = True
    fake_client.annotate.side_effect = lambda text, req_id=None: _news_annotation(text)
    monkeypatch.setattr(ClassicalNews, "get_shared_stanford_client", lambda: fake_client)
    return settings


def test_classical_news_entities_use_graph_types_and_real_chunk_ids(classical_news_settings):
    """H1 + H2: ORG/GPE types pass schema validation; chunk ids name stored Chunk nodes."""
    result = ExtractionPipeline.run_pipeline_nlp(NEWS_TEXT.encode(), "news.txt", "txt", job_id="job-uuid")
    chunk_by_id = {chunk.chunk_id: chunk for chunk in result["chunks"]}
    doc_id = result["preprocessed"].doc_id

    assert len(chunk_by_id) == 2
    final_entities, entity_to_chunks = build_resolved_entities(result["all_entities"], result["id_map"])
    assert {entity.type for entity in final_entities} == {"PERSON", "ORG", "GPE"}
    for entity in final_entities:
        ResolvedEntityPayload(**entity.to_dict())

    for entity in result["all_entities"]:
        assert entity.chunk_id.startswith(doc_id + ":")
        assert entity.text in chunk_by_id[entity.chunk_id].text
    for chunk_ids in entity_to_chunks.values():
        assert set(chunk_ids) <= set(chunk_by_id)
    for edge in result["resolved_edges"]:
        ResolvedEdgePayload(**edge.to_dict())
        assert set(edge.chunk_ids) <= set(chunk_by_id)
    # Native types stay on mentions for the evaluation contract.
    assert {claim.relation for claim in result["v2_claims"]} == {"WORKS_FOR", "BASED_IN"}


@pytest.mark.asyncio
async def test_classical_news_write_persists_claims_and_entities(classical_news_settings, monkeypatch, tmp_path):
    """H1 + M2: the write phase validates, writes ORG/GPE entities and keeps v2 claims."""
    result = ExtractionPipeline.run_pipeline_nlp(NEWS_TEXT.encode(), "news.txt", "txt")
    fake_session = MagicMock()
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=False)
    fake_driver = MagicMock()
    fake_driver.session.return_value = fake_session
    fake_driver.close = AsyncMock()
    monkeypatch.setattr(ExtractionPipeline, "create_driver", lambda *args: fake_driver)
    write_graph = AsyncMock()
    monkeypatch.setattr(ExtractionPipeline.GraphBuilder, "write_document_graph_atomic", write_graph)

    summary = await ExtractionPipeline.run_pipeline_write(result)

    written_entities = write_graph.call_args.args[4]
    written_edges = write_graph.call_args.args[6]
    assert {entity.type for entity in written_entities} == {"PERSON", "ORG", "GPE"}
    assert len(written_edges) == 2
    assert summary["entities"] == summary["new_entities"] == 3
    import json
    payload = json.loads((tmp_path / f"{summary['doc_id']}.json").read_text())
    assert len(payload["claims"]) == 2 and len(payload["canonical_claims"]) == 2


def test_clear_graph_requires_api_key(monkeypatch):
    """H3: DELETE /api/graph/clear is rejected without the configured key."""
    clear_session = MagicMock()
    monkeypatch.setattr(GraphRoutes, "get_driver", AsyncMock(return_value=clear_session))
    app.dependency_overrides[get_settings] = lambda: SimpleNamespace(
        api_key="secret", neo4j_uri="", neo4j_user="", neo4j_password="")
    try:
        client = TestClient(app)
        assert client.delete("/api/graph/clear").status_code == 401
        assert client.delete("/api/graph/clear", headers={"X-API-Key": "wrong"}).status_code == 401
        clear_session.session.assert_not_called()
    finally:
        app.dependency_overrides.clear()


def test_search_and_path_validation_and_errors(monkeypatch):
    """M4: blank queries are 422, same-entity paths are 400, driver errors are 503."""
    failing_driver = AsyncMock(side_effect=ConnectionError("neo4j down"))
    monkeypatch.setattr(SearchRoutes, "get_driver", failing_driver)
    monkeypatch.setattr(GraphRoutes, "get_driver", failing_driver)
    client = TestClient(app)
    assert client.get("/api/search?q=").status_code == 422
    assert client.get("/api/search?q=%20%20").status_code == 422
    assert client.get("/api/search?q=Google").status_code == 503
    assert client.get("/api/graph/path?source=a&target=a").status_code == 400
    assert client.get("/api/graph/path?source=a&target=b").status_code == 503
    assert client.get("/api/graph/hubs").status_code == 503


def _sleeping_bridge(ready_first: bool):
    """Popen stand-in: a Python process that hangs (optionally after the ready line)."""
    import subprocess
    script = ("import sys, time\n"
              + ("print('{\"status\": \"ready\"}', flush=True)\n" if ready_first else "")
              + "time.sleep(60)\n")
    real_popen = subprocess.Popen

    def fake_popen(cmd, **kwargs):
        return real_popen([sys.executable, "-c", script], **kwargs)
    return fake_popen


@pytest.mark.parametrize("ready_first", [False, True])
def test_stanford_client_times_out_on_hanging_bridge(monkeypatch, ready_first):
    """H4: silent startup and silent requests both raise TimeoutError and kill the process."""
    monkeypatch.setattr(StanfordClient.subprocess, "Popen", _sleeping_bridge(ready_first))
    client = StanfordClient.StanfordClassicalClient(startup_timeout=1.0, request_timeout=1.0)
    monkeypatch.setattr(client, "is_available", lambda: True)
    with pytest.raises(TimeoutError):
        client.annotate("Hello world.")
    assert client.proc is None


@pytest.fixture
def isolated_config_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONFIG_YAML", raising=False)
    monkeypatch.delenv("RELATION_BACKEND", raising=False)
    (tmp_path / "Configuration").mkdir()
    return tmp_path


def test_dotenv_value_beats_yaml(isolated_config_dir):
    """H5: .env takes precedence over Settings.yaml."""
    (isolated_config_dir / ".env").write_text("RELATION_BACKEND=semeval_svm\n")
    (isolated_config_dir / "Configuration/Settings.yaml").write_text("relation_backend: rules\nwindow_size: 5\n")
    settings = Settings()
    assert settings.relation_backend == "semeval_svm"
    assert settings.window_size == 5


def test_invalid_yaml_value_raises(isolated_config_dir):
    """H5: YAML values are validated instead of silently accepted."""
    (isolated_config_dir / "Configuration/Settings.yaml").write_text("relation_backend: typo\n")
    with pytest.raises(ValidationError):
        Settings()


def test_yaml_cannot_set_secrets(isolated_config_dir):
    (isolated_config_dir / "Configuration/Settings.yaml").write_text("api_key: from-yaml\n")
    assert Settings().api_key == ""


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __aiter__(self):
        async def generator():
            for row in self._rows:
                yield SimpleNamespace(data=lambda row=row: row)
        return generator()


@pytest.mark.asyncio
async def test_visualization_respects_node_limit():
    """M5: node count never exceeds the requested limit."""
    edge_rows = [
        {"src_id": f"s{i}", "src_name": "S", "src_type": "ORG", "rel_type": "R",
         "confidence": 1.0, "evidence_count": 1, "tgt_id": f"t{i}", "tgt_name": "T", "tgt_type": "ORG"}
        for i in range(5)
    ]
    isolated_rows = [{"id": f"iso{i}", "name": "I", "type": "ORG"} for i in range(5)]
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.run = AsyncMock(side_effect=[_FakeResult(edge_rows), _FakeResult(isolated_rows)])
    driver = MagicMock()
    driver.session.return_value = session

    data = await QueryEngine(driver).get_visualization_data(limit=5)

    assert len(data["nodes"]) <= 5
    linked_ids = {link["source"] for link in data["links"]} | {link["target"] for link in data["links"]}
    assert linked_ids <= {node["id"] for node in data["nodes"]}


def test_lexical_svm_score_rejects_misaligned_input():
    """L1: misaligned lists fail before zip() truncates them."""
    with pytest.raises(ValueError):
        score(["Other", "Other"], ["Other"])


def test_pdf_cleaner_keeps_body_text_and_numeric_prose():
    """L2: repeated lines are removed only at page edges; number-heavy prose survives."""
    repeated = "Quarterly Report"
    pages = [f"{repeated}\nBody line {i}.\n{repeated}\nMore body {i}.\nPage {i}" for i in range(4)]
    cleaned = remove_headers_footers("\x0c".join(pages))
    first_page = cleaned.split("\x0c")[0].splitlines()
    assert first_page[0] != repeated
    assert repeated in first_page  # third line is body text, not a header
    prose = "Revenue rose 12% to $4.5 billion in 2023."
    assert strip_tables(prose) == prose
    assert strip_tables("12 34 56 78 90 12") == ""
