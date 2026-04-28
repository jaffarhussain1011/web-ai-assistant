#!/usr/bin/env python3
"""
train.py
────────
CLI training entry point. Connects to a database, extracts selected tables,
builds a FAISS vector index, and saves it to data/vectors/.

── First-time training ────────────────────────────────────────────────────────
    python train.py --db_user=root --db_pass=1234 --db_name=mydb

── Include only specific tables (whitelist) ───────────────────────────────────
    python train.py --db_user=root --db_pass=1234 --db_name=mydb \\
        --tables products,orders,categories

── Exclude sensitive tables (blacklist) ───────────────────────────────────────
    python train.py --db_user=root --db_pass=1234 --db_name=mydb \\
        --exclude users,sessions,audit_log,payments

── Retrain after DB changes ───────────────────────────────────────────────────
    python train.py --db_user=root --db_pass=1234 --db_name=mydb --retrain

── List available tables without training ─────────────────────────────────────
    python train.py --db_user=root --db_pass=1234 --db_name=mydb --list_tables

Tip: use the admin panel at /static/admin.html for a GUI alternative.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from db.providers            import get_provider
from db.extractor            import tables_to_json
from embeddings.vector_store import VectorStore

RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
VECTORS_DIR   = ROOT / "data" / "vectors"


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the local Q&A system from a database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    db = parser.add_argument_group("Database connection")
    db.add_argument("--provider",  default="mysql",       help="DB provider: mysql (default)")
    db.add_argument("--db_host",   default="localhost",   help="DB host (default: localhost)")
    db.add_argument("--db_port",   type=int, default=3306, help="DB port (default: 3306)")
    db.add_argument("--db_user",   required=True,         help="DB username")
    db.add_argument("--db_pass",   required=True,         help="DB password")
    db.add_argument("--db_name",   required=True,         help="Database name")

    filt = parser.add_argument_group("Table filtering (mutually exclusive)")
    excl = filt.add_mutually_exclusive_group()
    excl.add_argument(
        "--tables", metavar="TABLE[,TABLE...]",
        help="Comma-separated whitelist of tables to include.",
    )
    excl.add_argument(
        "--exclude", metavar="TABLE[,TABLE...]",
        help="Comma-separated blacklist of tables to skip.",
    )

    parser.add_argument("--retrain",    action="store_true",
                        help="Rebuild index even if one exists (backs up old first).")
    parser.add_argument("--list_tables", action="store_true",
                        help="Print available table names and exit (no training).")
    parser.add_argument("--model",      default="all-MiniLM-L6-v2",
                        help="Sentence-transformer model (default: all-MiniLM-L6-v2)")
    parser.add_argument("--sample_rows",       type=int, default=50)
    parser.add_argument("--full_extract_limit", type=int, default=500)

    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_list(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [t.strip() for t in raw.split(",") if t.strip()]


def _backup_index() -> Path | None:
    index_file = VECTORS_DIR / "faiss.index"
    if not index_file.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = VECTORS_DIR / f"backup_{ts}"
    shutil.copytree(VECTORS_DIR, backup, ignore=shutil.ignore_patterns("backup_*"))
    logger.info("Old index backed up → %s", backup)
    return backup


def _prev_table_names() -> set[str]:
    files = sorted(RAW_DIR.glob("schema_*.json"))
    if not files:
        return set()
    with open(files[-1], encoding="utf-8") as f:
        return {e["table"] for e in json.load(f)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    include_tables = _parse_list(args.tables)
    exclude_tables = _parse_list(args.exclude)

    provider = get_provider(
        args.provider,
        host=args.db_host,
        port=args.db_port,
        user=args.db_user,
        password=args.db_pass,
        database=args.db_name,
        sample_limit=args.sample_rows,
        full_extract_limit=args.full_extract_limit,
    )
    provider.connect()

    # ── --list_tables ──────────────────────────────────────────────────────
    if args.list_tables:
        tables = provider.list_tables()
        provider.disconnect()
        print(f"\nDatabase '{args.db_name}' — {len(tables)} table(s):\n")
        for t in tables:
            print(f"  {t.name:<40} {t.row_count:>8} rows")
        print()
        return

    # ── Guard: already trained? ────────────────────────────────────────────
    index_path = VECTORS_DIR / "faiss.index"
    is_retrain = index_path.exists()

    if is_retrain and not args.retrain:
        logger.warning(
            "Trained index already exists at %s.\n"
            "  Add --retrain to rebuild, or open the admin panel.",
            index_path,
        )
        provider.disconnect()
        sys.exit(0)

    backup_path = None
    prev_tables: set[str] = set()
    if is_retrain:
        logger.info("RETRAIN — backing up existing index …")
        backup_path = _backup_index()
        prev_tables = _prev_table_names()

    # ── Extract ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 — Extracting from %s@%s/%s", args.db_user, args.db_host, args.db_name)

    try:
        tables = provider.extract(include=include_tables, exclude=exclude_tables)
    finally:
        provider.disconnect()

    if not tables:
        logger.error("No tables extracted. Check your --tables / --exclude filter.")
        sys.exit(1)

    logger.info("Extracted %d table(s).", len(tables))

    if is_retrain and prev_tables:
        new_set   = {t.name for t in tables}
        added     = new_set - prev_tables
        removed   = prev_tables - new_set
        unchanged = new_set & prev_tables
        if added:
            logger.info("  NEW     (+%d): %s", len(added),   ", ".join(sorted(added)))
        if removed:
            logger.info("  REMOVED (-%d): %s", len(removed), ", ".join(sorted(removed)))
        logger.info("  Unchanged (%d): %s", len(unchanged), ", ".join(sorted(unchanged)))

    # ── Save raw schema ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 — Saving raw schema …")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_file = RAW_DIR / f"schema_{ts}.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(tables_to_json(tables), f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", raw_file)

    # ── Build documents ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 — Converting to natural-language documents …")

    catalog_doc = provider.build_catalog_document(tables)
    documents: list[str] = [catalog_doc]
    logger.info("  %-30s → 1 doc (master catalog)", "[catalog]")

    for table in tables:
        docs = provider.table_to_documents(table)
        documents.extend(docs)
        note = "(full)" if table.full_extract else f"(sample/{table.row_count})"
        logger.info("  %-30s → %d doc(s) %s", table.name, len(docs), note)

    logger.info("Total documents: %d", len(documents))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / f"documents_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    with open(PROCESSED_DIR / f"knowledge_{ts}.txt", "w", encoding="utf-8") as f:
        for i, doc in enumerate(documents, 1):
            f.write(f"--- Document {i} ---\n{doc}\n\n")

    # ── Build FAISS index ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 — Generating embeddings and building FAISS index …")
    vs = VectorStore(model_name=args.model)
    vs.build(documents)
    vs.save()

    # ── Done ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("%s complete!", "Retraining" if is_retrain else "Training")
    logger.info("  Tables    : %d", len(tables))
    logger.info("  Documents : %d", len(documents))
    logger.info("  Chunks    : %d", len(vs._chunks))
    if backup_path:
        logger.info("  Backup    : %s", backup_path)
    logger.info("")
    logger.info("Start the API server with:  python app.py")


if __name__ == "__main__":
    main()
