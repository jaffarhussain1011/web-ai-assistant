"""
db/providers
────────────
Database provider factory.

Usage:
    from db.providers import get_provider

    provider = get_provider("mysql", host="localhost", user="root",
                             password="secret", database="mydb")
    provider.connect()
    tables = provider.list_tables()   # list[TableInfo]
    data   = provider.extract(include=["users", "orders"])
    provider.disconnect()

Adding a new provider:
    1. Create db/providers/postgres.py  implementing DatabaseProvider
    2. Add "postgres" to SUPPORTED_PROVIDERS
    3. Add the elif branch in get_provider()
"""

from .base import DatabaseProvider, TableInfo

SUPPORTED_PROVIDERS: list[str] = ["mysql"]


def get_provider(provider_type: str, **kwargs) -> DatabaseProvider:
    """
    Instantiate the right provider.

    Args:
        provider_type: "mysql" (more coming).
        **kwargs: forwarded to the provider constructor
                  (host, port, user, password, database, ...).
    """
    pt = provider_type.lower().strip()

    if pt == "mysql":
        from .mysql import MySQLProvider
        return MySQLProvider(**kwargs)

    raise ValueError(
        f"Unknown provider {provider_type!r}. "
        f"Supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )


__all__ = ["DatabaseProvider", "TableInfo", "get_provider", "SUPPORTED_PROVIDERS"]
