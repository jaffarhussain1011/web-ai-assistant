# System Architecture — Local AI Knowledge Base

> This document explains the "why" behind every major technology choice.
> Use it as a reference when demoing or explaining the system.

---

## The Core Problem: LLMs Have a Memory Limit

A local LLM like llama3.2 (3B) can only "see" roughly **4,000–8,000 tokens** at a time — that's its **context window**. One token ≈ ~4 characters.

If your database has 50,000 rows across 20 tables, that's potentially **millions of characters** of data. It physically cannot fit inside the LLM's context window in one shot.

So you can't just say: *"Here is my entire database. Now answer questions."*

---

## What Each Piece Does

```
Your DB rows
     │
     ▼
[sentence-transformers]  ← converts text → numbers (embeddings)
     │
     ▼
[FAISS index]            ← stores & searches those numbers fast
     │
     ▼  (only top 5 relevant chunks)
[Ollama / local LLM]     ← reads a small, focused context → answers
```

### sentence-transformers (all-MiniLM-L6-v2)
Converts text into a list of 384 numbers called an **embedding** — a mathematical fingerprint of the *meaning* of that text.

- "Users with admin role" and "administrators in the system" produce similar numbers even though they share no words.
- 22 MB model, runs on CPU in milliseconds.
- Purpose-built for semantic similarity — much better and faster at this than an LLM.

### FAISS (Facebook AI Similarity Search)
Stores all those number-fingerprints and answers one question instantly: *"Which stored chunks are mathematically closest to this query's fingerprint?"*

- Searches millions of vectors in under a millisecond.
- Runs entirely in memory — no network, no server.
- Understands nothing about the content — it's a pure math index.

### Ollama / Local LLM (llama3.2 / mistral / phi3)
Receives only the top-5 most relevant text chunks (≈ 800 words) plus the user's question.

- The only component that actually "understands" language and meaning.
- Generates a natural, conversational answer grounded in the retrieved context.
- Runs fully on-device — no data leaves the machine.

---

## Why Not Just Use the LLM for Everything?

| | Pure LLM only | This system (RAG) |
|---|---|---|
| DB size limit | Context window (~8K tokens) | Unlimited — only top-K chunks sent |
| Query speed | Slow — LLM reads all data every query | Fast — FAISS finds relevant chunks in <1ms |
| Accuracy | Hallucinates without real data | Grounded in actual DB rows |
| DB access at query time | Required (or answers go stale) | Not required — fully offline after training |
| Cost per query | High (more tokens = slower generation) | Low (only 5 chunks in context) |

The LLM is **powerful but narrow** — it can only see a small window at once.
FAISS is **fast but blind** — it can search everything instantly but understands nothing.
sentence-transformers **bridges** them by translating language into numbers both systems can work with.

---

## The Pattern: RAG (Retrieval-Augmented Generation)

This architecture has a name: **RAG**. It is the standard industry approach for grounding LLMs in private or large-scale data.

**Two phases:**

**Training (one-time):**
1. Extract rows from database → format as natural-language text documents
2. Chunk documents into overlapping windows (~400 chars)
3. Embed each chunk with sentence-transformers → 384-dimensional vector
4. Store all vectors in a FAISS index on disk

**Query (every user question):**
1. Embed the user's question with the same model → one vector
2. FAISS finds the top-K most similar vectors (cosine similarity)
3. Return the corresponding text chunks as context
4. Pass context + question to the LLM → natural language answer

---

## The Alternative We Moved Away From: Text-to-SQL

`db_app.py` implements a different approach:
1. Send the DB schema to the LLM
2. LLM writes a SQL query
3. Execute SQL against the live DB
4. Send results back to LLM → natural answer

**Advantages:** always fresh data, no training step.

**Why we moved away from it:**
- LLM frequently writes wrong SQL (wrong column names, wrong WHERE clauses, hallucinated table relationships)
- Requires live DB access at every query (security concern)
- Schema alone can overflow the context window on large databases
- Concrete bug hit: LLM guessed `metadata LIKE '%super_admin%'` because it had no sample values — the query returned zero rows

---

## Data Flow Diagram (Full)

```
┌─────────────────────────────────────────────────────┐
│                   TRAINING (one-time)                │
│                                                     │
│  MySQL DB ──► DBExtractor ──► NL Documents          │
│                                    │                │
│                               chunk_text()          │
│                                    │                │
│                           SentenceTransformer       │
│                           (all-MiniLM-L6-v2)        │
│                                    │                │
│                           384-dim float32 vectors   │
│                                    │                │
│                           FAISS IndexFlatIP          │
│                                    │                │
│                          data/vectors/faiss.index   │
│                          data/vectors/chunks.json   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                  QUERY (every request)              │
│                                                     │
│  User question                                      │
│       │                                             │
│  SentenceTransformer → query vector                 │
│       │                                             │
│  FAISS.search(query_vec, top_k=5)                   │
│       │                                             │
│  Top-5 text chunks (≈ 800 words)                    │
│       │                                             │
│  Ollama /api/generate                               │
│  (system prompt + context + question)               │
│       │                                             │
│  Natural language answer → user                     │
└─────────────────────────────────────────────────────┘
```

---

## Technology Choices at a Glance

| Component | Technology | Why |
|---|---|---|
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | 22 MB, CPU-fast, state-of-the-art semantic similarity |
| Vector index | FAISS (Facebook AI Similarity Search) | Sub-millisecond search, in-memory, no server needed |
| LLM | Ollama (llama3.2 / mistral / phi3) | Fully local, no API key, no data leaves the machine |
| API | FastAPI + Uvicorn | Async, fast, auto-generates OpenAPI docs |
| Frontend | Vanilla JS widget | Zero dependencies, embeddable anywhere |
| DB layer | Abstract provider pattern | MySQL now; PostgreSQL, SQLite, SQL Server pluggable |

---

## Key Design Decisions

- **Offline after training**: The database is only accessed during training. After that the server runs with no DB connection and no internet.
- **Hot-reload**: Training saves a new FAISS index and the server loads it live — no restart needed.
- **Password never on disk**: DB credentials live only in `app.state.db_session` (RAM) and are never serialised.
- **Small model friendly**: The system prompt uses a concrete GOOD/BAD example (few-shot) because 3B models respond better to examples than abstract rules.
