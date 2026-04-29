#!/usr/bin/env python3
"""
app.py
──────
Vector-mode FastAPI server.

Usage:
    python app.py
    # or:
    uvicorn app:create_app --factory --host 0.0.0.0 --port 8000 --reload

Environment variables:
    OLLAMA_BASE_URL  — default http://localhost:11434
    OLLAMA_MODEL     — default llama3.2
    PORT             — default 8000
    HOST             — default 0.0.0.0
    LOG_LEVEL        — default info

First-time setup:
    Start the server, then open http://localhost:8000/static/admin.html
    Use the Setup tab to connect to MySQL and train the knowledge base.
    No need to run train.py from the command line.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from embeddings.vector_store import VectorStore
from llm.model               import LocalLLM, LLMConfig
from api.routes              import router as qa_router
from api.setup_routes        import router as setup_router
from api.widget_routes       import router as widget_router

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── App factory ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Training job state (shared with setup_routes) ──────────────────────
    app.state.training_job = {
        "status":   "idle",
        "progress": 0,
        "log":      [],
        "error":    None,
        "result":   None,
    }
    app.state.db_session = None

    # ── Vector store ───────────────────────────────────────────────────────
    vs = VectorStore()
    try:
        vs.load()
        logger.info("Vector store ready: %d chunks", len(vs._chunks))
    except FileNotFoundError:
        logger.info(
            "No trained index found. "
            "Open /static/admin.html and use the Setup tab to train."
        )
    app.state.vector_store = vs

    # Pre-load the embedding model now so the first /ask request is instant
    # and HF Hub checks happen at startup rather than during a user query.
    try:
        _ = vs.model
    except Exception as exc:
        logger.warning("Could not pre-load embedding model: %s", exc)

    # ── LLM ───────────────────────────────────────────────────────────────
    llm_config = LLMConfig(
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    llm = LocalLLM(llm_config)
    try:
        llm.check_connection()
        logger.info("Ollama connection OK (model: %s)", llm_config.model)
    except RuntimeError as exc:
        logger.warning("Ollama check failed: %s", exc)
        logger.warning("The /ask endpoint will return 502 until Ollama is running.")
    app.state.llm = llm

    yield

    logger.info("Shutting down …")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Local AI Knowledge Base",
        description=(
            "Ask natural-language questions about your database. "
            "Powered by sentence-transformers + FAISS + local Ollama. "
            "Configure via /static/admin.html."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    frontend_dir = ROOT / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    app.include_router(qa_router)
    app.include_router(setup_router)
    app.include_router(widget_router)

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    logger.info(
        "Admin panel: http://localhost:%s/static/admin.html",
        os.getenv("PORT", "8000"),
    )
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
