"""
api/setup_routes.py
────────────────────
Admin setup endpoints — connect to DB, select tables, run training.

Endpoints:
    POST /setup/connect   — test connection, return table list + row counts
    POST /setup/train     — start background training for selected tables
    GET  /setup/status    — training progress (idle / running / done / error)
    GET  /setup/config    — persisted config (provider, host, tables, last trained)

Training flow:
    1. POST /setup/connect  → stores session credentials in app.state (memory only)
    2. POST /setup/train    → starts threading.Thread, returns immediately
    3. GET  /setup/status   → poll every 2 s for progress + log lines
    4. On completion        → app.state.vector_store hot-reloaded; config saved to disk

Security note:
    The database password is NEVER written to disk. It lives in app.state.db_session
    for the lifetime of the server process only. If the server restarts the admin
    must re-enter credentials, or set DB_PASS as an environment variable.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/setup", tags=["setup"])

ROOT          = Path(__file__).parent.parent
CONFIG_FILE   = ROOT / "data" / "setup_config.json"
VECTORS_DIR   = ROOT / "data" / "vectors"
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ConnectRequest(BaseModel):
    provider: str = "mysql"
    host: str     = "localhost"
    port: int     = 3306
    user: str
    password: str
    database: str


class TrainRequest(BaseModel):
    tables: list[str]


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(data: dict) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    safe = {k: v for k, v in data.items() if k != "password"}
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/connect")
async def connect(body: ConnectRequest, request: Request):
    """
    Test database credentials and return the full table list with row counts.
    On success, stores the session (including password) in app.state for training.
    """
    from db.providers import get_provider

    try:
        provider = get_provider(
            body.provider,
            host=body.host,
            port=body.port,
            user=body.user,
            password=body.password,
            database=body.database,
        )
        provider.connect()
        tables = provider.list_tables()
        provider.disconnect()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Store credentials in memory only — never serialised to disk
    request.app.state.db_session = {
        "provider": body.provider,
        "host":     body.host,
        "port":     body.port,
        "user":     body.user,
        "password": body.password,
        "database": body.database,
    }

    return {
        "ok":       True,
        "database": body.database,
        "provider": body.provider,
        "tables":   [{"name": t.name, "row_count": t.row_count} for t in tables],
    }


@router.post("/train")
async def train(body: TrainRequest, request: Request):
    """Start training in a background thread. Poll /setup/status for progress."""
    job = request.app.state.training_job

    if job["status"] == "running":
        raise HTTPException(status_code=409, detail="Training already in progress.")

    session: dict | None = getattr(request.app.state, "db_session", None)
    if not session:
        raise HTTPException(
            status_code=400,
            detail="No active DB session. Call POST /setup/connect first.",
        )

    if not body.tables:
        raise HTTPException(status_code=400, detail="No tables selected.")

    # Reset job state before launching thread
    job.update({
        "status":   "running",
        "progress": 0,
        "log":      [],
        "error":    None,
        "result":   None,
    })

    thread = threading.Thread(
        target=_run_training,
        args=(request.app, dict(session), list(body.tables)),
        daemon=True,
    )
    thread.start()

    return {"ok": True, "message": f"Training started for {len(body.tables)} table(s)."}


@router.get("/status")
async def status(request: Request):
    """Return the current training job state."""
    return request.app.state.training_job


@router.get("/config")
async def config(request: Request):
    """Return the persisted setup config (no password), plus live KB stats."""
    cfg = load_config()

    vs = getattr(request.app.state, "vector_store", None)
    if vs and getattr(vs, "is_loaded", False):
        cfg["chunks_loaded"] = len(vs._chunks)

    return cfg


# ── Background training pipeline ──────────────────────────────────────────────

def _run_training(app: Any, session: dict, tables: list[str]) -> None:
    """
    Full training pipeline running in a background thread.

    Steps:
      1. Connect to DB via provider
      2. Extract selected tables
      3. Save raw schema JSON
      4. Convert to NL documents
      5. Backup existing FAISS index
      6. Build new FAISS index
      7. Hot-reload vector store in running app
      8. Save setup config to disk
    """
    from db.providers   import get_provider
    from db.extractor   import table_to_documents, build_catalog_document, tables_to_json
    from embeddings.vector_store import VectorStore

    job = app.state.training_job

    def log(msg: str, progress: int | None = None) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        job["log"].append(f"[{ts}] {msg}")
        if progress is not None:
            job["progress"] = progress
        logger.info("[training] %s", msg)

    try:
        # ── Step 1: Connect ───────────────────────────────────────────────
        log("Connecting to database …", 5)
        provider = get_provider(
            session["provider"],
            host=session["host"],
            port=session["port"],
            user=session["user"],
            password=session["password"],
            database=session["database"],
        )
        provider.connect()
        log(f"Connected to '{session['database']}' on {session['host']}", 10)

        # ── Step 2: Extract tables ────────────────────────────────────────
        log(f"Extracting {len(tables)} table(s): {', '.join(tables)} …", 15)
        extracted = provider.extract(include=tables)
        provider.disconnect()
        log(f"Extracted {len(extracted)} table(s).", 30)

        if not extracted:
            raise RuntimeError("No tables extracted. Check your table selection.")

        for t in extracted:
            mode = "full" if t.full_extract else f"sample/{t.row_count}"
            log(f"  {t.name}: {t.row_count} rows ({mode})")

        # ── Step 3: Save raw schema JSON ──────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_file = RAW_DIR / f"schema_{timestamp}.json"
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(tables_to_json(extracted), f, ensure_ascii=False, indent=2)
        log(f"Schema saved → {raw_file.name}", 40)

        # ── Step 4: Build NL documents ────────────────────────────────────
        log("Converting tables to natural-language documents …", 50)
        catalog_doc = build_catalog_document(extracted)
        documents: list[str] = [catalog_doc]
        for table in extracted:
            docs = table_to_documents(table)
            documents.extend(docs)
            log(f"  {table.name} → {len(docs)} document(s)")
        log(f"Total documents: {len(documents)}", 60)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        proc_file = PROCESSED_DIR / f"documents_{timestamp}.json"
        with open(proc_file, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        # ── Step 5: Backup existing index ─────────────────────────────────
        index_file = VECTORS_DIR / "faiss.index"
        if index_file.exists():
            backup_dir = VECTORS_DIR / f"backup_{timestamp}"
            shutil.copytree(
                VECTORS_DIR, backup_dir,
                ignore=shutil.ignore_patterns("backup_*"),
            )
            log(f"Previous index backed up → {backup_dir.name}", 65)

        # ── Step 6: Embed + build FAISS index ─────────────────────────────
        log("Generating embeddings — this may take a few minutes …", 70)
        vs = VectorStore()
        vs.build(documents)
        vs.save()
        log(f"FAISS index built: {len(vs._chunks)} chunks", 90)

        # ── Step 7: Hot-reload vector store in running app ────────────────
        log("Reloading vector store in server …", 95)
        app.state.vector_store.load()
        log(f"Vector store live: {len(app.state.vector_store._chunks)} chunks", 98)

        # ── Step 8: Persist config (no password) ──────────────────────────
        cfg = {
            "provider":        session["provider"],
            "host":            session["host"],
            "port":            session["port"],
            "user":            session["user"],
            "database":        session["database"],
            "included_tables": tables,
            "last_trained":    datetime.now().isoformat(),
            "chunks":          len(vs._chunks),
            "total_tables":    len(extracted),
        }
        save_config(cfg)

        result = {"tables": len(extracted), "chunks": len(vs._chunks)}
        job.update({"status": "done", "progress": 100, "result": result})
        log(f"Training complete! {result['tables']} tables · {result['chunks']} chunks.", 100)

    except Exception as exc:
        logger.exception("Training failed")
        job.update({"status": "error", "error": str(exc)})
        log(f"ERROR: {exc}")
