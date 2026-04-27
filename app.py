#!/usr/bin/env python3
"""
app.py
──────
FastAPI server entry point.

Usage:
    python app.py
    # or with uvicorn directly:
    uvicorn app:create_app --factory --host 0.0.0.0 --port 8000 --reload

Environment variables (all optional):
    OLLAMA_BASE_URL  — default http://localhost:11434
    OLLAMA_MODEL     — default llama3.2
    PORT             — default 8000
    HOST             — default 0.0.0.0
    LOG_LEVEL        — default info
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
from api.routes              import router

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── App factory ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector store and warm-up the LLM on startup."""
    logger.info("Loading vector store …")
    vs = VectorStore()
    try:
        vs.load()
        logger.info("Vector store ready: %d chunks", len(vs._chunks))
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error("Run `python train.py ...` first, then restart the server.")

    logger.info("Initialising LLM wrapper …")
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

    # Attach to app state so routes can access them
    app.state.vector_store = vs
    app.state.llm          = llm

    yield   # ←  server is running

    logger.info("Shutting down …")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Local AI Knowledge Base",
        description=(
            "Ask natural-language questions about your MySQL database. "
            "Powered by sentence-transformers + FAISS + local Ollama LLM."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Allow any origin so the embeddable widget works on third-party pages
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Mount the frontend as static files at /static
    frontend_dir = ROOT / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    # Register API routes (prefix-free so /ask is at the root)
    app.include_router(router)

    return app


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
