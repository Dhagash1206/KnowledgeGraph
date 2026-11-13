"""FastAPI application — lifespan, CORS, routers, static UI."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from KnowledgeGraphApp.WebApi import JobStore as job_store
from KnowledgeGraphApp.WebApi.Routes import IngestRoutes as ingest, GraphRoutes as graph, SearchRoutes as search
from KnowledgeGraphApp.Settings import get_settings
from KnowledgeGraphApp.GraphStorage.Neo4jClient import get_driver, run_schema, close_driver


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + Neo4j schema. Shutdown: close driver."""
    settings = get_settings()

    # Init SQLite job store
    await job_store.init_db()

    # Init Neo4j schema (constraints + indexes) — idempotent IF NOT EXISTS
    try:
        driver = await get_driver(
            settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password
        )
        await run_schema(driver)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Neo4j schema init failed (Neo4j may not be ready yet): %s", exc
        )

    yield

    await close_driver()


app = FastAPI(
    title="Doc2Graph",
    description="Knowledge Graph Builder from Documents — Classical ML Edition",
    version="1.0.0",
    lifespan=lifespan,
)

# The bundled UI is same-origin and needs no CORS; other origins must be listed explicitly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers
app.include_router(ingest.router)
app.include_router(graph.router)
app.include_router(search.router)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# Serve the D3.js UI at /
app.mount("/", StaticFiles(directory="KnowledgeGraphApp/WebApi/Static", html=True), name="static")
