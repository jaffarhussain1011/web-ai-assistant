"""
db/extractor.py
───────────────
Connects to MySQL once during training.
Extracts schema metadata + sample rows and converts everything
into human-readable natural-language documents.
After this module runs, the rest of the system never touches the DB again.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import mysql.connector
from mysql.connector import Error as MySQLError

logger = logging.getLogger(__name__)

# Column name patterns that likely contain human-readable content worth embedding
_CONTENT_COL_RE = re.compile(
    r"(name|title|description|content|body|text|summary|detail|note|"
    r"message|label|rule|policy|terms|condition|remark|comment|value|"
    r"slug|tag|category|type|status|role|code|reason|subject|info)",
    re.IGNORECASE,
)


@dataclass
class ColumnMeta:
    name: str
    data_type: str
    nullable: bool
    default: str | None
    key: str          # PRI / MUL / UNI / ""
    extra: str        # auto_increment, etc.


@dataclass
class TableMeta:
    name: str
    columns: list[ColumnMeta] = field(default_factory=list)
    primary_keys: list[str] = field(default_factory=list)
    foreign_keys: list[dict] = field(default_factory=list)
    sample_rows: list[dict] = field(default_factory=list)
    all_rows: list[dict] = field(default_factory=list)    # populated when full extract
    row_count: int = 0
    full_extract: bool = False   # True when all_rows contains every row


class DBExtractor:
    """Extracts schema and data from a MySQL database."""

    # Default: show 50 sample rows in the schema overview document
    DEFAULT_SAMPLE_LIMIT = 50

    # Tables with <= this many rows will have ALL rows embedded individually
    DEFAULT_FULL_EXTRACT_LIMIT = 500

    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        database: str,
        port: int = 3306,
        sample_limit: int = DEFAULT_SAMPLE_LIMIT,
        full_extract_limit: int = DEFAULT_FULL_EXTRACT_LIMIT,
    ):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.sample_limit = sample_limit
        self.full_extract_limit = full_extract_limit
        self._conn: Any = None

    # ── Connection ──────────────────────────────────────────────────────────

    def connect(self) -> None:
        try:
            self._conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port,
                connection_timeout=10,
            )
            logger.info("Connected to MySQL: %s@%s/%s", self.user, self.host, self.database)
        except MySQLError as exc:
            raise ConnectionError(f"Cannot connect to MySQL: {exc}") from exc

    def disconnect(self) -> None:
        if self._conn and self._conn.is_connected():
            self._conn.close()
            logger.info("MySQL connection closed.")

    # ── Schema extraction ────────────────────────────────────────────────────

    def _get_table_names(self) -> list[str]:
        cur = self._conn.cursor()
        cur.execute("SHOW TABLES")
        return [row[0] for row in cur.fetchall()]

    def _get_columns(self, table: str) -> list[ColumnMeta]:
        cur = self._conn.cursor(dictionary=True)
        cur.execute(f"SHOW COLUMNS FROM `{table}`")
        cols = []
        for row in cur.fetchall():
            cols.append(ColumnMeta(
                name=row["Field"],
                data_type=row["Type"],
                nullable=row["Null"] == "YES",
                default=row["Default"],
                key=row["Key"],
                extra=row["Extra"],
            ))
        return cols

    def _get_foreign_keys(self, table: str) -> list[dict]:
        cur = self._conn.cursor(dictionary=True)
        query = """
            SELECT
                kcu.COLUMN_NAME          AS column_name,
                kcu.REFERENCED_TABLE_NAME AS ref_table,
                kcu.REFERENCED_COLUMN_NAME AS ref_column
            FROM information_schema.KEY_COLUMN_USAGE kcu
            JOIN information_schema.TABLE_CONSTRAINTS tc
                ON kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
               AND kcu.TABLE_SCHEMA    = tc.TABLE_SCHEMA
               AND kcu.TABLE_NAME      = tc.TABLE_NAME
            WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
              AND kcu.TABLE_SCHEMA   = %s
              AND kcu.TABLE_NAME     = %s
        """
        cur.execute(query, (self.database, table))
        return cur.fetchall()

    def _get_row_count(self, table: str) -> int:
        cur = self._conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM `{table}`")
        return cur.fetchone()[0]

    @staticmethod
    def _safe_row(row: dict) -> dict:
        """Convert non-JSON-serialisable values (dates, decimals, bytes) to strings."""
        return {
            k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
            for k, v in row.items()
        }

    def _get_rows(self, table: str, limit: int | None = None) -> list[dict]:
        cur = self._conn.cursor(dictionary=True)
        if limit is not None:
            cur.execute(f"SELECT * FROM `{table}` LIMIT {limit}")
        else:
            cur.execute(f"SELECT * FROM `{table}`")
        return [self._safe_row(r) for r in cur.fetchall()]

    # ── Public API ───────────────────────────────────────────────────────────

    def list_tables(self) -> list[str]:
        """Return all table names in the database (no filtering)."""
        return self._get_table_names()

    def extract(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[TableMeta]:
        """
        Return metadata + samples for tables in the database.

        Args:
            include: If given, process ONLY these tables (whitelist).
                     Names are case-insensitive.
            exclude: If given, skip these tables (blacklist).
                     Applied after `include` filtering.
                     Names are case-insensitive.

        Examples:
            extractor.extract()                          # all tables
            extractor.extract(include=["products","orders"])  # only these two
            extractor.extract(exclude=["users","sessions"])   # everything except these
        """
        all_tables = self._get_table_names()

        # Normalise filter lists to lowercase for case-insensitive comparison
        include_set = {t.lower() for t in include} if include else None
        exclude_set = {t.lower() for t in exclude} if exclude else set()

        selected: list[str] = []
        skipped:  list[str] = []

        for tname in all_tables:
            tname_lower = tname.lower()
            if include_set is not None and tname_lower not in include_set:
                skipped.append(tname)
                continue
            if tname_lower in exclude_set:
                skipped.append(tname)
                continue
            selected.append(tname)

        if skipped:
            logger.info("Skipped tables (%d): %s", len(skipped), ", ".join(skipped))

        tables = []
        for tname in selected:
            cols = self._get_columns(tname)
            fks  = self._get_foreign_keys(tname)
            cnt  = self._get_row_count(tname)
            pks  = [c.name for c in cols if c.key == "PRI"]

            # Decide whether to do a full extract (all rows) or just sample rows.
            # Full extract is done for small tables so every value gets embedded
            # and questions like "do we have a team named X?" can be answered.
            do_full = cnt <= self.full_extract_limit

            if do_full:
                logger.info(
                    "Extracting table: %-30s  (%d rows — full extract)", tname, cnt
                )
                all_rows     = self._get_rows(tname)               # every row
                sample_rows  = all_rows[: self.sample_limit]        # subset for overview doc
            else:
                logger.info(
                    "Extracting table: %-30s  (%d rows — sample only, "
                    "exceeds full-extract limit of %d)",
                    tname, cnt, self.full_extract_limit,
                )
                sample_rows  = self._get_rows(tname, limit=self.sample_limit)
                all_rows     = []

            tables.append(TableMeta(
                name=tname,
                columns=cols,
                primary_keys=pks,
                foreign_keys=fks,
                sample_rows=sample_rows,
                all_rows=all_rows,
                row_count=cnt,
                full_extract=do_full,
            ))
        return tables


# ── Natural-language document builder ────────────────────────────────────────

def _row_to_sentence(table_name: str, row: dict) -> str:
    """
    Convert a single DB row into a natural-language sentence.

    Example:
        policies row {id:1, rule:"Only adults allowed", category:"admission"}
        →  "In table 'policies': rule is 'Only adults allowed',
            category is 'admission', id is 1."

    This makes individual values (team names, policy text, etc.) searchable
    by the vector store so questions like "do we have a team named X?" work.
    """
    parts = []
    for col, val in row.items():
        if val is None:
            continue
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ("none", "null"):
            continue
        parts.append(f"{col} is {val_str!r}")
    if not parts:
        return ""
    return f"In table '{table_name}': " + ", ".join(parts) + "."


def _identify_content_columns(table: TableMeta) -> list[str]:
    """
    Return column names that look like they carry human-readable content
    (names, descriptions, rules, policies, etc.) rather than IDs or timestamps.
    These columns get extra attention in row-level documents.
    """
    content_cols = []
    for col in table.columns:
        # Skip pure numeric / date / binary columns
        dtype_lower = col.data_type.lower()
        if any(t in dtype_lower for t in ("int", "float", "double", "decimal",
                                           "date", "time", "blob", "binary")):
            continue
        if _CONTENT_COL_RE.search(col.name):
            content_cols.append(col.name)
    return content_cols


def table_to_documents(table: TableMeta) -> list[str]:
    """
    Convert a TableMeta into a list of natural-language text chunks.

    Document types produced:
      1. Schema overview  — columns, types, row count
      2. Relationships    — foreign key links to other tables
      3. Sample overview  — a compact block showing sample rows
      4. Per-row docs     — one sentence per row (when full_extract=True)
                           These are the KEY addition that makes the system
                           answer questions about actual values/content.
      5. Content summary  — deduplicated list of values in content columns
      6. FAQ              — quick-answer facts
    """
    docs: list[str] = []

    # ── 1. Schema overview ────────────────────────────────────────────────
    col_descs = []
    for col in table.columns:
        nullable = "nullable" if col.nullable else "required"
        pk_tag   = " (primary key)" if col.key == "PRI" else ""
        uk_tag   = " (unique)"      if col.key == "UNI" else ""
        ai_tag   = " auto-increment" if "auto_increment" in col.extra else ""
        col_descs.append(
            f"  - {col.name} ({col.data_type}, {nullable}{pk_tag}{uk_tag}{ai_tag})"
        )

    schema_doc = (
        f"Table '{table.name}' has {table.row_count} rows "
        f"and {len(table.columns)} columns:\n"
        + "\n".join(col_descs)
    )
    if table.primary_keys:
        schema_doc += f"\nPrimary key(s): {', '.join(table.primary_keys)}"
    docs.append(schema_doc)

    # ── 2. Relationships ──────────────────────────────────────────────────
    if table.foreign_keys:
        rel_lines = [
            f"  - Column '{fk['column_name']}' references "
            f"'{fk['ref_table']}.{fk['ref_column']}'"
            for fk in table.foreign_keys
        ]
        docs.append(
            f"Table '{table.name}' is related to other tables via foreign keys:\n"
            + "\n".join(rel_lines)
        )

    # ── 3. Sample overview block ───────────────────────────────────────────
    rows_to_show = table.all_rows if table.full_extract else table.sample_rows
    if rows_to_show:
        lines = []
        for i, row in enumerate(rows_to_show, 1):
            pairs = ", ".join(f"{k}={v!r}" for k, v in row.items())
            lines.append(f"  Row {i}: {pairs}")
        label = "All data" if table.full_extract else "Sample data"
        docs.append(
            f"{label} from table '{table.name}' "
            f"({len(rows_to_show)} of {table.row_count} rows):\n"
            + "\n".join(lines)
        )

    # ── 4. Per-row documents (full extract only) ──────────────────────────
    #
    # This is the most important addition.
    # Each row becomes its own searchable document, so a question like
    # "do we have a team named SuperAdmins?" will retrieve the exact row
    # that contains that name and the LLM can give a definitive answer.
    #
    if table.full_extract and table.all_rows:
        for row in table.all_rows:
            sentence = _row_to_sentence(table.name, row)
            if sentence:
                docs.append(sentence)

    # ── 5. Content-column value summary ───────────────────────────────────
    #
    # Deduplicated list of values in "content" columns (name, title, rule, …).
    # Makes it easy for the LLM to answer "what are all the teams?" or
    # "list all policies" even without matching a specific row.
    #
    content_cols = _identify_content_columns(table)
    source_rows  = table.all_rows or table.sample_rows

    if content_cols and source_rows:
        for col_name in content_cols:
            values = []
            seen: set[str] = set()
            for row in source_rows:
                val = row.get(col_name)
                if val is not None:
                    v = str(val).strip()
                    if v and v not in seen:
                        seen.add(v)
                        values.append(v)
            if values:
                docs.append(
                    f"All values of '{col_name}' in table '{table.name}': "
                    + ", ".join(f"'{v}'" for v in values) + "."
                )

    # ── 6. FAQ ─────────────────────────────────────────────────────────────
    col_names = [c.name for c in table.columns]
    docs.append(
        f"Frequently asked about '{table.name}': "
        f"What columns does it have? Columns: {', '.join(col_names)}. "
        f"How many records? {table.row_count} rows total."
    )

    return docs


def build_catalog_document(tables: list[TableMeta]) -> str:
    """
    Build a single master catalog document listing ALL tables and their
    row counts. This ensures "how many tables are there?" is always
    answered correctly from one authoritative chunk.
    """
    lines = [
        f"The database contains {len(tables)} table(s):",
    ]
    for t in tables:
        lines.append(f"  - '{t.name}' ({t.row_count} rows)")
    lines.append(
        f"\nTable names: {', '.join(t.name for t in tables)}."
    )
    return "\n".join(lines)


def tables_to_json(tables: list[TableMeta]) -> list[dict]:
    """Serialise extracted table metadata to plain dicts (for raw JSON save)."""
    result = []
    for t in tables:
        result.append({
            "table": t.name,
            "row_count": t.row_count,
            "columns": [
                {
                    "name": c.name,
                    "type": c.data_type,
                    "nullable": c.nullable,
                    "key": c.key,
                    "extra": c.extra,
                }
                for c in t.columns
            ],
            "primary_keys": t.primary_keys,
            "foreign_keys": t.foreign_keys,
            "sample_rows": t.sample_rows,
        })
    return result
