"""
cache/query_cache.py
────────────────────
Two-layer cache for Q&A answers:

  Layer 1 — In-memory dict  (microsecond lookup, lost on restart)
  Layer 2 — JSON file on disk (survives restarts, loaded into memory at startup)

Cache key  = normalised question string (lowercase, stripped punctuation/spaces)
Cache value = {answer, sql, latency_ms, hits, created_at, expires_at}

TTL is configurable (default 24 hours). Expired entries are evicted lazily
(on next read) and also during periodic cleanup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 60 * 60 * 24   # 24 hours
CACHE_FILE = Path(__file__).parent.parent / "data" / "cache" / "query_cache.json"


# ── Cache entry ───────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    answer:     str
    sql:        str          # the SQL that was run (empty for schema-only answers)
    latency_ms: float        # original generation time
    hits:       int          # how many times this entry was served from cache
    created_at: float        # unix timestamp
    expires_at: float        # unix timestamp


# ── Normalisation ─────────────────────────────────────────────────────────────

_PUNCT = re.compile(r"[^\w\s]")

def _normalise(question: str) -> str:
    """
    Reduce a question to a stable cache key.
    "How many Users???" and "how many users" → same key.
    """
    q = question.lower().strip()
    q = _PUNCT.sub("", q)                 # remove punctuation
    q = re.sub(r"\s+", " ", q).strip()   # collapse whitespace
    return q

def _cache_key(question: str) -> str:
    norm = _normalise(question)
    # Use MD5 as the dict key (short, consistent)
    return hashlib.md5(norm.encode()).hexdigest()


# ── QueryCache ────────────────────────────────────────────────────────────────

class QueryCache:
    """
    Thread-safe, persistent two-layer cache.

    Usage:
        cache = QueryCache()
        cache.load()

        entry = cache.get("how many users?")
        if entry:
            return entry.answer

        # ... generate answer ...
        cache.set("how many users?", answer="5 users", sql="SELECT COUNT(*) FROM users", latency_ms=1200)
        cache.save()
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS, cache_file: Path = CACHE_FILE):
        self.ttl_seconds = ttl_seconds
        self.cache_file  = cache_file
        self._store: dict[str, CacheEntry] = {}
        self._lock  = Lock()
        # Track original question text alongside the key for /cache endpoint
        self._questions: dict[str, str] = {}

    # ── Load / Save ───────────────────────────────────────────────────────

    def load(self) -> None:
        """Load persisted cache from disk into memory."""
        if not self.cache_file.exists():
            logger.info("No cache file yet at %s — starting fresh.", self.cache_file)
            return
        try:
            with open(self.cache_file, encoding="utf-8") as f:
                raw = json.load(f)
            now = time.time()
            loaded = 0
            for key, data in raw.get("entries", {}).items():
                entry = CacheEntry(**data)
                if entry.expires_at > now:      # skip expired
                    self._store[key] = entry
                    loaded += 1
            self._questions = raw.get("questions", {})
            logger.info("Cache loaded: %d live entries from %s", loaded, self.cache_file)
        except Exception as exc:
            logger.warning("Could not load cache: %s — starting empty.", exc)

    def save(self) -> None:
        """Persist the current in-memory cache to disk."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "entries":   {k: asdict(v) for k, v in self._store.items()},
                "questions": self._questions,
            }
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("Could not save cache: %s", exc)

    # ── Get / Set ─────────────────────────────────────────────────────────

    def get(self, question: str) -> CacheEntry | None:
        """Return a live cache entry or None if missing/expired."""
        key = _cache_key(question)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at < time.time():
                # Lazy eviction
                del self._store[key]
                self._questions.pop(key, None)
                logger.debug("Cache entry expired and evicted: %s", question[:60])
                return None
            entry.hits += 1
            return entry

    def set(
        self,
        question: str,
        answer: str,
        sql: str = "",
        latency_ms: float = 0.0,
    ) -> None:
        """Store a new answer in the cache."""
        key  = _cache_key(question)
        now  = time.time()
        entry = CacheEntry(
            answer=answer,
            sql=sql,
            latency_ms=latency_ms,
            hits=0,
            created_at=now,
            expires_at=now + self.ttl_seconds,
        )
        with self._lock:
            self._store[key] = entry
            self._questions[key] = question[:200]   # store truncated original

    def invalidate(self, question: str) -> bool:
        """Remove a specific question from the cache. Returns True if it existed."""
        key = _cache_key(question)
        with self._lock:
            existed = key in self._store
            self._store.pop(key, None)
            self._questions.pop(key, None)
        return existed

    def clear(self) -> int:
        """Wipe the entire cache. Returns how many entries were removed."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._questions.clear()
        self.save()
        return count

    def evict_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._store.items() if v.expires_at < now]
            for k in expired:
                del self._store[k]
                self._questions.pop(k, None)
        return len(expired)

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            total  = len(self._store)
            hits   = sum(e.hits for e in self._store.values())
            recent = sorted(
                [
                    {
                        "question":   self._questions.get(k, "?"),
                        "hits":       v.hits,
                        "created_at": v.created_at,
                        "expires_in": max(0, v.expires_at - time.time()),
                    }
                    for k, v in self._store.items()
                ],
                key=lambda x: x["hits"],
                reverse=True,
            )[:20]
        return {"total_entries": total, "total_cache_hits": hits, "top_entries": recent}
