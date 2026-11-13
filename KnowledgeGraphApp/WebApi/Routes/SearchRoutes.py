"""Search routes — full-text entity search."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from KnowledgeGraphApp.WebApi.Schemas import SearchResult
from KnowledgeGraphApp.Settings import Settings, get_settings
from KnowledgeGraphApp.GraphStorage.Neo4jClient import get_driver
from KnowledgeGraphApp.GraphSearch.QueryEngine import QueryEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])


@router.get("", response_model=list[SearchResult])
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    doc_id: str | None = Query(None, min_length=1),
    settings: Settings = Depends(get_settings),
):
    """Full-text search over entity names and aliases."""
    query_text = q.strip()
    if not query_text:
        raise HTTPException(422, "Search query cannot be empty or whitespace only")
    try:
        driver = await get_driver(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        engine = QueryEngine(driver)
        results = await engine.search_entities(query_text, limit, doc_id=doc_id)
    except Exception as exc:
        logger.warning("Search failed for %r: %s", query_text, exc)
        raise HTTPException(503, f"Neo4j database unavailable: {exc}")
    return [SearchResult(**r) for r in results]
