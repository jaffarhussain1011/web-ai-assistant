"""
llm/sql_agent.py
────────────────
Text-to-SQL agent using a local Ollama LLM.

Two-step pipeline per question:

  Step 1 — SQL generation
      Input : database schema (text) + user question
      Output: a single SELECT SQL query

  Step 2 — Answer generation
      Input : original question + SQL that was run + query result rows
      Output: a natural-language answer

Both steps use the same local Ollama model. No external API calls.

If Step 1 produces invalid/unsafe SQL the agent falls back to answering
the question from the schema alone (schema-only mode).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import requests

from db.direct_query import DirectQueryExecutor, QueryResult
from llm.model       import LLMConfig, OLLAMA_BASE, DEFAULT_MODEL

logger = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────────────

_SQL_SYSTEM = """\
You are an expert MySQL query writer.
Given a database schema and a user question, write ONE valid SELECT query that answers the question.

Rules:
- Output ONLY the SQL query — no explanation, no markdown, no extra text.
- Always use SELECT. Never use INSERT, UPDATE, DELETE, DROP, or any other DML/DDL.
- If the question cannot be answered with SQL (e.g. it asks for opinions), output: NO_SQL
- Prefer simple queries. Use JOINs only when needed.
- Limit results to 100 rows unless the question asks for a specific count or all rows.
- Use backtick-quoted identifiers: `table_name`.`column_name`
"""

_SQL_USER_TMPL = """\
Schema:
{schema}

Question: {question}

SQL:"""


_ANSWER_SYSTEM = """\
You are a helpful data analyst assistant.
You will be given:
  1. A user question
  2. The SQL query that was executed
  3. The query result

Your job is to answer the question in clear, natural language based on the result.

Rules:
- Be concise and direct.
- If the result is empty, say so clearly.
- If the result has many rows, summarise the key findings.
- Do not repeat the SQL back to the user unless they asked for it.
- Do not invent data — only use what is in the result.
"""

_ANSWER_USER_TMPL = """\
Question: {question}

SQL executed:
{sql}

Result ({row_count} row(s)):
{result_text}

Answer:"""

_SCHEMA_ONLY_SYSTEM = """\
You are a helpful database assistant.
Answer the user's question using ONLY the database schema provided.
If you cannot answer from the schema, say so clearly.
"""

_SCHEMA_ONLY_USER_TMPL = """\
Schema:
{schema}

Question: {question}

Answer:"""


# ── SQL extraction ────────────────────────────────────────────────────────────

_CODE_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_INLINE_CODE = re.compile(r"`([^`]+)`")


def _extract_sql(raw: str) -> str | None:
    """
    Pull the SQL out of an LLM response that may contain markdown code fences,
    explanations, or be a bare query. Returns None if nothing usable found.
    """
    raw = raw.strip()

    if raw.upper() == "NO_SQL":
        return None

    # Try fenced code block first
    m = _CODE_BLOCK.search(raw)
    if m:
        return m.group(1).strip()

    # Try inline code
    m = _INLINE_CODE.search(raw)
    if m:
        return m.group(1).strip()

    # Bare response — take the first SELECT statement
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    sql_lines = []
    in_select = False
    for line in lines:
        if re.match(r"^\s*SELECT\b", line, re.IGNORECASE):
            in_select = True
        if in_select:
            sql_lines.append(line)
    if sql_lines:
        return " ".join(sql_lines)

    return None


def _result_to_text(result: QueryResult, max_rows: int = 30) -> str:
    """Format query results as a compact readable string for the LLM."""
    if not result.rows:
        return "(no rows returned)"

    rows = result.rows[:max_rows]
    lines = []

    # Single-column, single-row → just the value
    if len(result.columns) == 1 and len(rows) == 1:
        val = list(rows[0].values())[0]
        return str(val)

    # Otherwise: key=value pairs per row
    for row in rows:
        pairs = ", ".join(f"{k}={v!r}" for k, v in row.items())
        lines.append(pairs)

    if result.row_count > max_rows:
        lines.append(f"... ({result.row_count - max_rows} more rows not shown)")

    return "\n".join(lines)


# ── SQLAgent ─────────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    answer:     str
    sql:        str       # empty string if no SQL was used
    from_cache: bool
    latency_ms: float
    row_count:  int       # 0 if no SQL ran or cache hit


class SQLAgent:
    """
    Orchestrates the Text-to-SQL pipeline.

    Usage:
        agent = SQLAgent(executor, config)
        result = agent.ask("how many users are there?")
        print(result.answer)
    """

    def __init__(
        self,
        executor: DirectQueryExecutor,
        config: LLMConfig | None = None,
    ):
        self.executor = executor
        self.config   = config or LLMConfig()

    # ── Ollama call ───────────────────────────────────────────────────────

    def _call_ollama(self, system: str, user_prompt: str) -> str:
        """Send a prompt to Ollama and return the response text."""
        payload = {
            "model":      self.config.model,
            "system":     system,
            "prompt":     user_prompt,
            "stream":     False,
            "keep_alive": self.config.keep_alive,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
            },
        }
        prompt_preview = user_prompt[:120].replace("\n", " ")
        logger.info(
            "[Ollama] POST /api/generate  model=%s  timeout=%ds  prompt=%.120s...",
            self.config.model, self.config.request_timeout, prompt_preview,
        )
        try:
            resp = requests.post(
                f"{self.config.ollama_url}/api/generate",
                json=payload,
                timeout=self.config.request_timeout,
            )
            resp.raise_for_status()
            response_text = resp.json().get("response", "").strip()
            logger.info("[Ollama] Response received (%d chars)", len(response_text))
            return response_text
        except requests.Timeout:
            raise RuntimeError(
                f"Ollama request timed out after {self.config.request_timeout}s. "
                "The model may still be loading. Retry in a moment, or set "
                "OLLAMA_TIMEOUT=600 for large/slow models."
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama error: {exc}") from exc

    # ── Pipeline ──────────────────────────────────────────────────────────

    def ask(self, question: str) -> AgentResult:
        """
        Full pipeline: question → SQL → execute → natural-language answer.
        Falls back to schema-only answer if SQL generation fails.
        """
        t0 = time.perf_counter()
        schema = self.executor.get_schema()

        _sep = "─" * 60

        # ── Step 1: Generate SQL ──────────────────────────────────────────
        logger.info("%s", _sep)
        logger.info("[STEP 1] Generating SQL for question: %s", question)
        logger.info("[STEP 1] Sending schema (%d chars) + question to LLM ...", len(schema))

        t1 = time.perf_counter()
        sql_prompt       = _SQL_USER_TMPL.format(schema=schema, question=question)
        raw_sql_response = self._call_ollama(_SQL_SYSTEM, sql_prompt)
        t1_ms            = (time.perf_counter() - t1) * 1000

        logger.info("[STEP 1] LLM responded in %.0f ms", t1_ms)
        logger.info("[STEP 1] Raw LLM output:\n%s", raw_sql_response)

        sql = _extract_sql(raw_sql_response)
        logger.info("[STEP 1] Extracted SQL: %s", sql if sql else "None (NO_SQL or unparseable)")

        # ── Step 2a: Execute SQL + generate answer ────────────────────────
        if sql:
            logger.info("%s", _sep)
            logger.info("[STEP 2] Executing SQL on database:")
            logger.info("[STEP 2]   %s", sql)

            try:
                t2      = time.perf_counter()
                result  = self.executor.execute(sql)
                t2_ms   = (time.perf_counter() - t2) * 1000

                logger.info(
                    "[STEP 2] Query returned %d row(s) in %.0f ms",
                    result.row_count, t2_ms,
                )
                if result.rows:
                    preview = str(result.rows[:3])
                    logger.info("[STEP 2] Result preview (first 3 rows): %s", preview[:300])
                else:
                    logger.info("[STEP 2] Result: (empty — no rows matched)")

                result_text = _result_to_text(result)

                logger.info("%s", _sep)
                logger.info("[STEP 3] Generating natural-language answer ...")
                t3 = time.perf_counter()
                answer_prompt = _ANSWER_USER_TMPL.format(
                    question=question,
                    sql=sql,
                    row_count=result.row_count,
                    result_text=result_text,
                )
                answer = self._call_ollama(_ANSWER_SYSTEM, answer_prompt)
                t3_ms  = (time.perf_counter() - t3) * 1000

                logger.info("[STEP 3] Answer generated in %.0f ms", t3_ms)
                logger.info("[STEP 3] Answer: %s", answer[:300])
                row_count = result.row_count

            except ValueError as exc:
                logger.warning("%s", _sep)
                logger.warning("[STEP 2] SQL REJECTED (safety check): %s", exc)
                logger.warning("[STEP 2] Falling back to schema-only answer.")
                answer    = self._schema_only_answer(question, schema)
                sql       = f"[REJECTED] {sql}"
                row_count = 0

            except RuntimeError as exc:
                logger.warning("%s", _sep)
                logger.warning("[STEP 2] SQL EXECUTION FAILED: %s", exc)
                logger.warning("[STEP 2] Failed SQL was: %s", sql)
                logger.warning("[STEP 2] Falling back to schema-only answer.")
                answer    = self._schema_only_answer(question, schema)
                sql       = f"[FAILED] {sql}"
                row_count = 0

        # ── Step 2b: Schema-only answer (no SQL) ──────────────────────────
        else:
            logger.info("%s", _sep)
            logger.info("[STEP 2] LLM returned NO_SQL — answering from schema only.")
            answer    = self._schema_only_answer(question, schema)
            row_count = 0

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info("%s", _sep)
        logger.info("[DONE] Total latency: %.0f ms | rows: %d | from_cache: False", total_ms, row_count)
        logger.info("%s", _sep)

        return AgentResult(
            answer=answer,
            sql=sql,
            from_cache=False,
            latency_ms=round(total_ms, 1),
            row_count=row_count,
        )

    def _schema_only_answer(self, question: str, schema: str) -> str:
        """Answer from schema alone when SQL cannot be used."""
        logger.info("[SCHEMA-ONLY] Generating answer from schema (no SQL executed).")
        prompt = _SCHEMA_ONLY_USER_TMPL.format(schema=schema, question=question)
        return self._call_ollama(_SCHEMA_ONLY_SYSTEM, prompt)
