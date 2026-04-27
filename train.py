#!/usr/bin/env python3
"""
train.py
────────
Training / retraining command.

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
    python train.py ... --retrain --tables products,orders  # retrain a subset

    The old index is backed up before being replaced so you can roll back:
        data/vectors/backup_<timestamp>/

── List available tables without training ─────────────────────────────────────
    python train.py --db_user=root --db_pass=1234 --db_name=mydb --list_tables
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from db.extractor             import (
    DBExtractor, table_to_documents, tables_to_json, build_catalog_document,
)
from embeddings.vector_store  import VectorStore

RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
VECTORS_DIR   = ROOT / "data" / "vectors"


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train (or retrain) the local Q&A system from a MySQL database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # DB connection
    db = parser.add_argument_group("Database connection")
    db.add_argument("--db_host", default="localhost", help="MySQL host (default: localhost)")
    db.add_argument("--db_port", type=int, default=3306, help="MySQL port (default: 3306)")
    db.add_argument("--db_user", required=True, help="MySQL username")
    db.add_argument("--db_pass", required=True, help="MySQL password")
    db.add_argument("--db_name", required=True, help="Database name")

    # Table filtering
    filt = parser.add_argument_group("Table filtering (mutually exclusive)")
    excl = filt.add_mutually_exclusive_group()
    excl.add_argument(
        "--tables",
        metavar="TABLE[,TABLE...]",
        help=(
            "Comma-separated whitelist of tables to include. "
            "Everything else is skipped. "
            "Example: --tables products,orders,categories"
        ),
    )
    excl.add_argument(
        "--exclude",
        metavar="TABLE[,TABLE...]",
        help=(
            "Comma-separated blacklist of tables to skip. "
            "Useful for hiding sensitive tables (users, payments, audit_log, etc.). "
            "Example: --exclude users,sessions,payments"
        ),
    )

    # Behaviour flags
    parser.add_argument(
        "--retrain", action="store_true",
        help="Rebuild the index even if one already exists. Old index is backed up first.",
    )
    parser.add_argument(
        "--list_tables", action="store_true",
        help="Connect to the DB, print all available table names, then exit (no training).",
    )
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="Sentence-transformer embedding model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--sample_rows", type=int, default=50,
        help=(
            "Max rows to show in the schema overview document per table "
            "(default: 50). Does not affect full-extract tables."
        ),
    )
    parser.add_argument(
        "--full_extract_limit", type=int, default=500,
        help=(
            "Tables with <= this many rows will have ALL rows embedded as "
            "individual documents, making exact-value lookups possible "
            "(e.g. 'do we have a team named X?'). Default: 500. "
            "Set to 0 to disable full extraction."
        ),
    )

    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_table_list(raw: str | None) -> list[str] | None:
    """Split 'a,b, c' → ['a', 'b', 'c'], or return None if input is None."""
    if not raw:
        return None
    return [t.strip() for t in raw.split(",") if t.strip()]


def _backup_existing_index() -> Path | None:
    """
    Copy data/vectors/ → data/vectors/backup_<timestamp>/ before overwriting.
    Returns the backup path, or None if nothing existed yet.
    """
    index_file = VECTORS_DIR / "faiss.index"
    if not index_file.exists():
        return None

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir  = VECTORS_DIR / f"backup_{timestamp}"
    shutil.copytree(VECTORS_DIR, backup_dir,
                    ignore=shutil.ignore_patterns("backup_*"))
    logger.info("Old index backed up → %s", backup_dir)
    return backup_dir


def _load_previous_table_names() -> set[str]:
    """
    Read the most recent raw schema JSON and return the set of table names
    that were trained last time. Used to show a diff during retrain.
    """
    raw_files = sorted(RAW_DIR.glob("schema_*.json"))
    if not raw_files:
        return set()
    with open(raw_files[-1], encoding="utf-8") as f:
        data = json.load(f)
    return {entry["table"] for entry in data}


def _print_retrain_diff(prev_tables: set[str], new_tables: list[str]) -> None:
    """Log which tables are new, removed, or unchanged since last training."""
    new_set  = set(new_tables)
    added    = new_set - prev_tables
    removed  = prev_tables - new_set
    unchanged = new_set & prev_tables

    logger.info("── Retrain diff vs previous run ──────────────────────────")
    if added:
        logger.info("  NEW tables      (+%d): %s", len(added),    ", ".join(sorted(added)))
    if removed:
        logger.info("  REMOVED tables  (-%d): %s", len(removed),  ", ".join(sorted(removed)))
    logger.info("  Unchanged tables  (%d): %s", len(unchanged), ", ".join(sorted(unchanged)))
    logger.info("──────────────────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    include_tables = _parse_table_list(args.tables)
    exclude_tables = _parse_table_list(args.exclude)

    # ── Connect to DB ──────────────────────────────────────────────────────
    extractor = DBExtractor(
        host=args.db_host,
        user=args.db_user,
        password=args.db_pass,
        database=args.db_name,
        port=args.db_port,
        sample_limit=args.sample_rows,
        full_extract_limit=args.full_extract_limit,
    )
    extractor.connect()

    # ── --list_tables: just print and exit ─────────────────────────────────
    if args.list_tables:
        all_tables = extractor.list_tables()
        extractor.disconnect()
        print(f"\nDatabase '{args.db_name}' contains {len(all_tables)} table(s):\n")
        for t in all_tables:
            print(f"  {t}")
        print(
            "\nTip: use --tables to whitelist or --exclude to blacklist.\n"
            "Example: python train.py ... --exclude users,sessions,payments\n"
        )
        return

    # ── Guard: already trained? ────────────────────────────────────────────
    index_path = VECTORS_DIR / "faiss.index"
    is_retrain = index_path.exists()

    if is_retrain and not args.retrain:
        logger.warning(
            "A trained index already exists at %s.\n"
            "  • To retrain with updated data:  add --retrain\n"
            "  • To list available tables:      add --list_tables",
            index_path,
        )
        extractor.disconnect()
        sys.exit(0)

    # ── Retrain: backup old index first ────────────────────────────────────
    backup_path  = None
    prev_tables  = set()
    if is_retrain:
        logger.info("=" * 60)
        logger.info("RETRAIN mode — backing up existing index …")
        backup_path = _backup_existing_index()
        prev_tables = _load_previous_table_names()

    # ── Step 1: Validate table filter against actual DB ────────────────────
    if include_tables or exclude_tables:
        logger.info("=" * 60)
        logger.info("STEP 0 — Validating table filter …")
        all_db_tables = extractor.list_tables()
        all_lower     = {t.lower() for t in all_db_tables}

        if include_tables:
            unknown = [t for t in include_tables if t.lower() not in all_lower]
            if unknown:
                logger.error(
                    "Unknown table(s) in --tables: %s\n"
                    "Available: %s", ", ".join(unknown), ", ".join(all_db_tables)
                )
                extractor.disconnect()
                sys.exit(1)
            logger.info("Whitelist: including only → %s", ", ".join(include_tables))

        if exclude_tables:
            unknown = [t for t in exclude_tables if t.lower() not in all_lower]
            if unknown:
                logger.warning(
                    "Some --exclude names not found in DB (will be ignored): %s",
                    ", ".join(unknown),
                )
            logger.info("Blacklist: excluding → %s", ", ".join(exclude_tables))

    # ── Step 2: Extract from MySQL ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(
        "STEP 1 — Connecting to MySQL: %s@%s/%s",
        args.db_user, args.db_host, args.db_name,
    )

    try:
        tables = extractor.extract(include=include_tables, exclude=exclude_tables)
    finally:
        extractor.disconnect()   # DB connection closed — never used again

    if not tables:
        logger.error("No tables extracted. Check your --tables / --exclude filter.")
        sys.exit(1)

    logger.info("Extracted %d table(s).", len(tables))

    # Show diff only when retraining
    if is_retrain and prev_tables:
        _print_retrain_diff(prev_tables, [t.name for t in tables])

    # ── Step 3: Save raw JSON ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 — Saving raw schema JSON …")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_file  = RAW_DIR / f"schema_{timestamp}.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(tables_to_json(tables), f, ensure_ascii=False, indent=2)
    logger.info("Raw schema saved: %s", raw_file)

    # ── Step 4: Convert to natural-language documents ─────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 — Converting schema to natural-language documents …")

    # Master catalog always goes first — guarantees "how many tables?" is answered
    catalog_doc = build_catalog_document(tables)
    documents: list[str] = [catalog_doc]
    logger.info("  %-30s → 1 document (master catalog)", "[catalog]")

    for table in tables:
        docs = table_to_documents(table)
        documents.extend(docs)
        row_note = "(full)" if table.full_extract else f"(sample/{table.row_count})"
        logger.info("  %-30s → %d document(s) %s", table.name, len(docs), row_note)

    logger.info("Total documents: %d", len(documents))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / f"documents_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    with open(PROCESSED_DIR / f"knowledge_{timestamp}.txt", "w", encoding="utf-8") as f:
        for i, doc in enumerate(documents, 1):
            f.write(f"--- Document {i} ---\n{doc}\n\n")

    # ── Step 5: Embed + build FAISS index ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 — Generating embeddings and building FAISS index …")
    vs = VectorStore(model_name=args.model)
    vs.build(documents)
    vs.save()

    # ── Done ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    action = "Retraining" if is_retrain else "Training"
    logger.info("%s complete!", action)
    logger.info("  Tables processed : %d", len(tables))
    logger.info("  Documents created: %d", len(documents))
    logger.info("  Index chunks     : %d", len(vs._chunks))
    if backup_path:
        logger.info("  Previous index   : %s  (kept as backup)", backup_path)
    logger.info("")
    logger.info("Start the API server with:  python app.py")


if __name__ == "__main__":
    main()
