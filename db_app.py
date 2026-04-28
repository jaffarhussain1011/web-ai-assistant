#!/usr/bin/env python3
"""
db_app.py
─────────
Direct-DB mode server — answers questions by querying the live MySQL database
via a Text-to-SQL pipeline. No training step required.

Usage:
    python db_app.py \
        --db_host=localhost \
        --db_user=root \
        --db_pass=1234 \
        --db_name=mydb

Or via environment variables:
    DB_HOST=localhost DB_USER=root DB_PASS=1234 DB_NAME=mydb python db_app.py

Endpoints (same widget works with apiUrl pointed here):
    POST /ask          — ask a question
    GET  /health       — status
    GET  /schema       — view DB schema as understood by the LLM
    GET  /cache/stats  — cache stats
    POST /cache/clear  — wipe cache
    POST /cache/invalidate — remove one cached answer
    GET  /docs         — Swagger UI
"""

from __future__ import annotations

import argparse
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

from db.direct_query      import DirectQueryExecutor
from llm.model            import LLMConfig
from llm.sql_agent        import SQLAgent
from cache.query_cache    import QueryCache
from api.direct_routes    import router

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
# Ensure sql_agent logs are always visible regardless of root level
logging.getLogger("llm.sql_agent").setLevel(logging.INFO)
logging.getLogger("db.direct_query").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct-DB Q&A server (Text-to-SQL)")
    p.add_argument("--db_host", default=os.getenv("DB_HOST", "localhost"))
    p.add_argument("--db_port", type=int, default=int(os.getenv("DB_PORT", 3306)))
    p.add_argument("--db_user", default=os.getenv("DB_USER", ""))
    p.add_argument("--db_pass", default=os.getenv("DB_PASS", ""))
    p.add_argument("--db_name", default=os.getenv("DB_NAME", ""))
    p.add_argument("--host",    default=os.getenv("HOST",    "0.0.0.0"))
    p.add_argument("--port",    type=int, default=int(os.getenv("PORT", 8000)))
    p.add_argument(
        "--cache_ttl", type=int, default=int(os.getenv("CACHE_TTL", 86400)),
        help="Cache TTL in seconds (default: 86400 = 24 hours)",
    )
    return p.parse_args()


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(args: argparse.Namespace) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ── Validate required DB args ──────────────────────────────────────
        if not args.db_user or not args.db_name:
            logger.error(
                "DB credentials required. Use --db_user / --db_pass / --db_name "
                "or set DB_USER / DB_PASS / DB_NAME environment variables."
            )
            sys.exit(1)

        # ── Connect to DB ──────────────────────────────────────────────────
        logger.info("Connecting to MySQL: %s@%s/%s", args.db_user, args.db_host, args.db_name)
        executor = DirectQueryExecutor(
            host=args.db_host,
            user=args.db_user,
            password=args.db_pass,
            database=args.db_name,
            port=args.db_port,
        )
        try:
            executor.connect()
        except ConnectionError as exc:
            logger.error("DB connection failed: %s", exc)
            sys.exit(1)

        # ── Pre-warm schema cache ──────────────────────────────────────────
        logger.info("Loading database schema …")
        schema = executor.get_schema()
        logger.info("Schema loaded:\n%s", schema)

        # ── Set up LLM + agent ─────────────────────────────────────────────
        llm_config = LLMConfig(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.0,   # deterministic SQL generation
        )
        agent = SQLAgent(executor=executor, config=llm_config)
        logger.info("SQL agent ready (model: %s)", llm_config.model)

        # ── Load query cache ───────────────────────────────────────────────
        cache = QueryCache(ttl_seconds=args.cache_ttl)
        cache.load()
        logger.info("Query cache ready (TTL: %ds)", args.cache_ttl)

        # Attach to app state
        app.state.sql_agent   = agent
        app.state.query_cache = cache

        yield   # ← server is running

        # ── Shutdown ───────────────────────────────────────────────────────
        logger.info("Saving cache and disconnecting …")
        cache.save()
        executor.disconnect()

    app = FastAPI(
        title="Local AI — Direct DB Mode",
        description=(
            "Ask natural-language questions about your live MySQL database. "
            "Uses Text-to-SQL via local Ollama. No training step required."
        ),
        version="1.0.0",
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

    app.include_router(router)
    return app


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    app  = create_app(args)
    logger.info(
        "Starting Direct-DB server on http://%s:%d", args.host, args.port
    )
    logger.info(
        "Widget config:  ChatWidget.init({ apiUrl: 'http://localhost:%d/ask' })",
        args.port,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
