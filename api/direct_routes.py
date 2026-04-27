"""
api/direct_routes.py
────────────────────
FastAPI routes for the direct DB mode (Text-to-SQL pipeline).

Endpoints:
  POST /ask          — ask a question (cache-first, then Text-to-SQL)
  GET  /health       — server + DB connection status
  GET  /schema       — dump the cached DB schema
  GET  /cache/stats  — cache hit stats
  POST /cache/clear  — wipe the cache
  POST /cache/invalidate — remove one specific entry
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question:   str  = Field(..., min_length=1, max_length=2000)
    skip_cache: bool = Field(False, description="Force re-run even if cached")


class AskResponse(BaseModel):
    answer:     str
    sql:        str           # the SQL query that was executed (empty = schema-only)
    from_cache: bool
    latency_ms: float
    row_count:  int


class InvalidateRequest(BaseModel):
    question: str = Field(..., description="Exact question text to remove from cache")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health", tags=["system"])
async def health(request: Request):
    """Check DB connectivity and whether the agent is ready."""
    agent = request.app.state.sql_agent
    cache = request.app.state.query_cache

    db_ok = False
    try:
        agent.executor._ensure_connected()
        db_ok = True
    except Exception:
        pass

    return {
        "status":         "ok" if db_ok else "db_unavailable",
        "db_connected":   db_ok,
        "model":          agent.config.model,
        "cache_entries":  cache.stats()["total_entries"],
    }


@router.post("/ask", response_model=AskResponse, tags=["qa"])
async def ask(body: AskRequest, request: Request) -> AskResponse:
    """
    Answer a natural-language question about the live database.

    Flow:
      1. Normalise + hash the question
      2. Check cache → return instantly if hit
      3. Call SQLAgent (LLM writes SQL → execute → LLM explains result)
      4. Store result in cache
      5. Return answer
    """
    agent = request.app.state.sql_agent
    cache = request.app.state.query_cache

    # ── Cache lookup ──────────────────────────────────────────────────────
    if not body.skip_cache:
        entry = cache.get(body.question)
        if entry:
            logger.info("Cache HIT (hits=%d): %s", entry.hits, body.question[:60])
            return AskResponse(
                answer=entry.answer,
                sql=entry.sql,
                from_cache=True,
                latency_ms=entry.latency_ms,   # original generation time
                row_count=0,
            )

    # ── Live query ────────────────────────────────────────────────────────
    logger.info("Cache MISS — running SQL agent: %s", body.question[:80])
    try:
        result = agent.ask(body.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Store in cache (async-safe: cache uses a threading.Lock internally)
    cache.set(
        body.question,
        answer=result.answer,
        sql=result.sql,
        latency_ms=result.latency_ms,
    )
    # Persist to disk in background (best-effort)
    try:
        cache.save()
    except Exception as exc:
        logger.warning("Cache save failed: %s", exc)

    return AskResponse(
        answer=result.answer,
        sql=result.sql,
        from_cache=False,
        latency_ms=result.latency_ms,
        row_count=result.row_count,
    )


@router.get("/schema", tags=["debug"])
async def get_schema(request: Request, refresh: bool = False):
    """Return the cached database schema string. Add ?refresh=true to reload."""
    agent = request.app.state.sql_agent
    try:
        schema = agent.executor.get_schema(force_refresh=refresh)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"schema": schema}


@router.get("/cache/stats", tags=["cache"])
async def cache_stats(request: Request):
    """Return cache statistics: total entries, total hits, top cached questions."""
    return request.app.state.query_cache.stats()


@router.post("/cache/clear", tags=["cache"])
async def cache_clear(request: Request):
    """Wipe the entire cache (in-memory + disk)."""
    count = request.app.state.query_cache.clear()
    return {"cleared": count, "message": f"Removed {count} cache entries."}


@router.post("/cache/invalidate", tags=["cache"])
async def cache_invalidate(body: InvalidateRequest, request: Request):
    """Remove a specific question from the cache so it gets re-answered."""
    existed = request.app.state.query_cache.invalidate(body.question)
    request.app.state.query_cache.save()
    return {
        "removed": existed,
        "message": "Entry removed." if existed else "Entry not found in cache.",
    }
