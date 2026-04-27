"""
db/direct_query.py
──────────────────
Safe, read-only SQL executor for the direct-DB mode.

Key safety rules:
  - Only SELECT statements are allowed (DML/DDL are rejected before execution)
  - Results are capped at MAX_ROWS to prevent flooding the LLM context
  - Connection is kept alive with reconnect-on-failure
  - Schema is loaded once and cached in memory
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import mysql.connector
from mysql.connector import Error as MySQLError

logger = logging.getLogger(__name__)

MAX_ROWS = 200   # cap result sets sent to LLM

# Reject anything that isn't a SELECT (case-insensitive, strips leading comments)
_SELECT_ONLY = re.compile(r"^\s*(--|/\*.*?\*/\s*)*\s*SELECT\b", re.IGNORECASE | re.DOTALL)
_DANGEROUS   = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|EXEC|CALL)\b",
    re.IGNORECASE,
)


@dataclass
class QueryResult:
    columns: list[str]
    rows:    list[dict]
    row_count: int          # actual rows returned (capped at MAX_ROWS)
    total_count: int | None # None if unknown


class DirectQueryExecutor:
    """
    Maintains a persistent MySQL connection and executes read-only queries.

    Usage:
        executor = DirectQueryExecutor(host, user, password, database)
        executor.connect()
        schema = executor.get_schema()
        result = executor.execute("SELECT COUNT(*) FROM users")
        executor.disconnect()
    """

    def __init__(self, host: str, user: str, password: str, database: str, port: int = 3306):
        self.host     = host
        self.user     = user
        self.password = password
        self.database = database
        self.port     = port
        self._conn: Any = None
        self._schema_cache: str | None = None

    # ── Connection ────────────────────────────────────────────────────────

    def connect(self) -> None:
        try:
            self._conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port,
                connection_timeout=10,
                autocommit=True,
            )
            logger.info("DirectQuery connected: %s@%s/%s", self.user, self.host, self.database)
        except MySQLError as exc:
            raise ConnectionError(f"Cannot connect to MySQL: {exc}") from exc

    def disconnect(self) -> None:
        if self._conn and self._conn.is_connected():
            self._conn.close()

    def _ensure_connected(self) -> None:
        """Reconnect if the connection dropped (e.g. server timeout)."""
        try:
            if self._conn and self._conn.is_connected():
                return
        except Exception:
            pass
        logger.info("Reconnecting to MySQL …")
        self.connect()

    # ── Schema ────────────────────────────────────────────────────────────

    def get_schema(self, force_refresh: bool = False) -> str:
        """
        Return a compact text description of the database schema.
        Cached in memory after the first call — schema rarely changes at runtime.

        Format:
            TABLE users (id INT PK, name VARCHAR(120), email VARCHAR(255) UNIQUE, ...)
            TABLE orders (id INT PK, user_id INT FK→users.id, ...)
        """
        if self._schema_cache and not force_refresh:
            return self._schema_cache

        self._ensure_connected()
        cur = self._conn.cursor()
        cur.execute("SHOW TABLES")
        tables = [row[0] for row in cur.fetchall()]

        lines = [f"Database: {self.database}", f"Tables ({len(tables)}):"]

        for tname in tables:
            cur2 = self._conn.cursor(dictionary=True)
            cur2.execute(f"SHOW COLUMNS FROM `{tname}`")
            cols = cur2.fetchall()

            # Foreign keys
            cur3 = self._conn.cursor(dictionary=True)
            cur3.execute("""
                SELECT kcu.COLUMN_NAME, kcu.REFERENCED_TABLE_NAME, kcu.REFERENCED_COLUMN_NAME
                FROM information_schema.KEY_COLUMN_USAGE kcu
                JOIN information_schema.TABLE_CONSTRAINTS tc
                  ON kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
                 AND kcu.TABLE_SCHEMA    = tc.TABLE_SCHEMA
                 AND kcu.TABLE_NAME      = tc.TABLE_NAME
                WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
                  AND kcu.TABLE_SCHEMA   = %s
                  AND kcu.TABLE_NAME     = %s
            """, (self.database, tname))
            fk_map = {
                row["COLUMN_NAME"]: f"FK→{row['REFERENCED_TABLE_NAME']}.{row['REFERENCED_COLUMN_NAME']}"
                for row in cur3.fetchall()
            }

            # Row count
            cur4 = self._conn.cursor()
            cur4.execute(f"SELECT COUNT(*) FROM `{tname}`")
            row_count = cur4.fetchone()[0]

            col_parts = []
            for col in cols:
                tag = ""
                if col["Key"] == "PRI":
                    tag = " PK"
                elif col["Key"] == "UNI":
                    tag = " UNIQUE"
                if col["Field"] in fk_map:
                    tag += f" {fk_map[col['Field']]}"
                col_parts.append(f"{col['Field']} {col['Type']}{tag}")

            lines.append(
                f"  TABLE {tname} ({row_count} rows): "
                + ", ".join(col_parts)
            )

        self._schema_cache = "\n".join(lines)
        return self._schema_cache

    def invalidate_schema_cache(self) -> None:
        self._schema_cache = None

    # ── Query execution ───────────────────────────────────────────────────

    @staticmethod
    def _validate_sql(sql: str) -> None:
        """
        Raise ValueError if the SQL is not a safe SELECT statement.
        This is a defence-in-depth check — the LLM prompt already instructs
        it to only write SELECT queries.
        """
        if not _SELECT_ONLY.match(sql):
            raise ValueError(
                f"Only SELECT queries are allowed. Received: {sql[:100]!r}"
            )
        if _DANGEROUS.search(sql):
            raise ValueError(
                f"Dangerous keyword detected in SQL: {sql[:100]!r}"
            )

    def execute(self, sql: str) -> QueryResult:
        """
        Validate and execute a SELECT query. Returns up to MAX_ROWS rows.
        Raises ValueError for unsafe SQL, RuntimeError for DB errors.
        """
        self._validate_sql(sql)
        self._ensure_connected()

        try:
            cur = self._conn.cursor(dictionary=True)
            cur.execute(sql)
            all_rows = cur.fetchmany(MAX_ROWS + 1)   # fetch one extra to detect truncation
        except MySQLError as exc:
            raise RuntimeError(f"SQL execution error: {exc}") from exc

        truncated  = len(all_rows) > MAX_ROWS
        rows       = all_rows[:MAX_ROWS]
        columns    = list(rows[0].keys()) if rows else []

        # Serialise non-JSON types
        safe_rows = []
        for row in rows:
            safe_rows.append({
                k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                for k, v in row.items()
            })

        return QueryResult(
            columns=columns,
            rows=safe_rows,
            row_count=len(safe_rows),
            total_count=None if truncated else len(safe_rows),
        )
