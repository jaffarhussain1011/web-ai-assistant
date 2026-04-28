"""
db/providers/base.py
────────────────────
Abstract base class for all database providers.

To add a new provider (e.g. PostgreSQL):
  1. Create db/providers/postgres.py
  2. Subclass DatabaseProvider and implement all abstract methods
  3. Register it in db/providers/__init__.py  get_provider()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TableInfo:
    """Lightweight table descriptor returned by list_tables()."""
    name: str
    row_count: int


class DatabaseProvider(ABC):
    """
    Common interface every database provider must implement.

    Lifecycle:
        provider = get_provider("mysql", host=..., user=..., ...)
        provider.connect()
        tables = provider.list_tables()          # → list[TableInfo]
        data   = provider.extract(include=[...]) # → list[TableMeta]
        docs   = provider.table_to_documents(data[0])
        provider.disconnect()
    """

    @abstractmethod
    def connect(self) -> None:
        """Open the database connection."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the database connection."""

    @abstractmethod
    def list_tables(self) -> list[TableInfo]:
        """Return all tables with row counts — no data extraction."""

    @abstractmethod
    def extract(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list:
        """
        Extract schema + data for selected tables.
        Returns list[TableMeta] (see db.extractor).

        Args:
            include: Whitelist — process only these tables.
            exclude: Blacklist — skip these tables (applied after include).
        """

    @abstractmethod
    def table_to_documents(self, table) -> list[str]:
        """Convert a single TableMeta into natural-language text chunks."""

    @abstractmethod
    def build_catalog_document(self, tables: list) -> str:
        """Build the master catalog document listing all extracted tables."""
