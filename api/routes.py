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
import random
import re
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Small-talk / greeting handling ────────────────────────────────────────────

_SMALLTALK: list[tuple[re.Pattern, list[str]]] = [
    (
        re.compile(r"^\s*(hi+|hello+|hey+|howdy|hiya|yo+|sup)\W*$", re.I),
        [
            "Hi there! How can I help you today?",
            "Hello! What would you like to know?",
            "Hey! Go ahead and ask me anything about the data.",
        ],
    ),
    (
        re.compile(r"\bgood\s*(morning|afternoon|evening|day)\b", re.I),
        [
            "Good day! What can I help you with?",
            "Hi! Hope you're having a great day. What would you like to know?",
        ],
    ),
    (
        re.compile(r"\bhow are (you|u)\b|\bhow('?re| are) (you|u) doing\b|\bhow.*going\b", re.I),
        [
            "I'm doing great, thanks for asking! Ready to answer your questions.",
            "All good here — what can I help you with?",
        ],
    ),
    (
        re.compile(r"\bwhat'?s up\b", re.I),
        [
            "Not much! Just here to help. What do you need?",
            "Hey! Ask me anything about the data.",
        ],
    ),
    (
        re.compile(r"\b(thank(s| you)|thx|cheers)\b", re.I),
        [
            "You're welcome! Let me know if there's anything else.",
            "Happy to help! Feel free to ask more questions.",
            "Anytime!",
        ],
    ),
    (
        re.compile(r"\b(bye+|goodbye|see (ya|you)|cya|take care)\b", re.I),
        [
            "Goodbye! Come back if you have more questions.",
            "See you later! Have a great day.",
        ],
    ),
    (
        re.compile(r"\bwho are you\b|\bwhat are you\b|\bintroduce yourself\b", re.I),
        [
            "I'm your AI assistant, here to help you explore and understand your data. "
            "Try asking something like 'How many users are there?' or 'Tell me about the products table.'",
        ],
    ),
    (
        re.compile(r"^\s*help\s*\??\s*$", re.I),
        [
            "Sure! Ask me anything about your database — for example: "
            "'How many active users are there?' or 'What products do we have?'",
        ],
    ),
]


def _smalltalk_reply(text: str) -> str | None:
    """Return a canned reply if *text* is small-talk, else None."""
    for pattern, replies in _SMALLTALK:
        if pattern.search(text):
            return random.choice(replies)
    return None


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

    # Short-circuit for greetings / small-talk — no vector search or LLM needed
    reply = _smalltalk_reply(body.question)
    if reply:
        return AskResponse(answer=reply, sources_count=0, latency_ms=0.0)

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
