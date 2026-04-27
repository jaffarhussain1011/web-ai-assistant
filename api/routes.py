"""
api/routes.py
─────────────
FastAPI router for the Q&A endpoints.

Pipeline for POST /ask:
  1. Receive question
  2. Embed question via sentence-transformers
  3. Retrieve top-k chunks from FAISS
  4. Send context + question to local LLM (Ollama)
  5. Return answer
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User's question")
    top_k:    int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")


class AskResponse(BaseModel):
    answer:        str
    sources_count: int        # how many chunks were used as context
    latency_ms:    float      # end-to-end time in milliseconds


class HealthResponse(BaseModel):
    status:        str
    index_loaded:  bool
    chunks_count:  int


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request) -> HealthResponse:
    """Check if the vector store is loaded and the service is ready."""
    vs  = request.app.state.vector_store
    return HealthResponse(
        status="ok",
        index_loaded=vs.is_loaded,
        chunks_count=len(vs._chunks),
    )


@router.post("/ask", response_model=AskResponse, tags=["qa"])
async def ask(body: AskRequest, request: Request) -> AskResponse:
    """
    Answer a natural-language question about the trained database.

    The question is matched against embedded knowledge chunks; the best
    matches are passed as context to the local LLM, which generates an answer.
    """
    vs  = request.app.state.vector_store
    llm = request.app.state.llm

    if not vs.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Vector store not loaded. Run `python train.py` first.",
        )

    t0 = time.perf_counter()

    # 1. Retrieve relevant chunks
    try:
        chunks = vs.search(body.question, top_k=body.top_k)
    except Exception as exc:
        logger.exception("Vector search failed")
        raise HTTPException(status_code=500, detail=f"Search error: {exc}") from exc

    if not chunks:
        return AskResponse(
            answer="I couldn't find relevant information in the knowledge base.",
            sources_count=0,
            latency_ms=0.0,
        )

    # 2. Generate answer
    try:
        answer = llm.ask(body.question, chunks)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return AskResponse(
        answer=answer,
        sources_count=len(chunks),
        latency_ms=round(elapsed_ms, 1),
    )


@router.get("/chunks", tags=["debug"])
async def list_chunks(request: Request, limit: int = 20, offset: int = 0):
    """
    Debug endpoint: browse the raw text chunks stored in the vector index.
    Useful during development / verification.
    """
    vs = request.app.state.vector_store
    if not vs.is_loaded:
        raise HTTPException(status_code=503, detail="Vector store not loaded.")
    total  = len(vs._chunks)
    sliced = vs._chunks[offset: offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "chunks": sliced}
