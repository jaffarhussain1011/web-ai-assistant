"""
db/providers/mysql.py
─────────────────────
MySQL implementation of DatabaseProvider.
Delegates to DBExtractor for all extraction logic.
"""

from __future__ import annotations

from .base import DatabaseProvider, TableInfo
from db.extractor import (
    DBExtractor,
    table_to_documents as _to_docs,
    build_catalog_document as _catalog,
)


class MySQLProvider(DatabaseProvider):
    """
    MySQL provider — wraps DBExtractor.

    Usage:
        p = MySQLProvider(host="localhost", user="root",
                          password="...", database="mydb")
        p.connect()
        tables = p.list_tables()
        data   = p.extract(include=["users", "products"])
        p.disconnect()
    """

    def __init__(
        self,
        host: str = "localhost",
        user: str = "",
        password: str = "",
        database: str = "",
        port: int = 3306,
        sample_limit: int = 50,
        full_extract_limit: int = 500,
    ):
        self._extractor = DBExtractor(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
            sample_limit=sample_limit,
            full_extract_limit=full_extract_limit,
        )

    def connect(self) -> None:
        self._extractor.connect()

    def disconnect(self) -> None:
        self._extractor.disconnect()

    def list_tables(self) -> list[TableInfo]:
        return [
            TableInfo(name=name, row_count=count)
            for name, count in self._extractor.list_tables_with_counts()
        ]

    def extract(self, include=None, exclude=None):
        return self._extractor.extract(include=include, exclude=exclude)

    def table_to_documents(self, table) -> list[str]:
        return _to_docs(table)

    def build_catalog_document(self, tables) -> str:
        return _catalog(tables)
