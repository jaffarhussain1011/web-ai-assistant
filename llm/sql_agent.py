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
            "model":  self.config.model,
            "system": system,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
            },
        }
        try:
            resp = requests.post(
                f"{self.config.ollama_url}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.Timeout:
            raise RuntimeError("Ollama request timed out.")
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

        # ── Step 1: Generate SQL ──────────────────────────────────────────
        sql_prompt = _SQL_USER_TMPL.format(schema=schema, question=question)
        logger.info("Generating SQL for: %s", question[:80])
        raw_sql_response = self._call_ollama(_SQL_SYSTEM, sql_prompt)
        logger.debug("Raw LLM SQL response: %s", raw_sql_response[:200])

        sql = _extract_sql(raw_sql_response)

        # ── Step 2a: Execute SQL + generate answer ────────────────────────
        if sql:
            try:
                logger.info("Executing SQL: %s", sql[:120])
                result = self.executor.execute(sql)
                result_text = _result_to_text(result)

                answer_prompt = _ANSWER_USER_TMPL.format(
                    question=question,
                    sql=sql,
                    row_count=result.row_count,
                    result_text=result_text,
                )
                logger.info("Generating natural-language answer …")
                answer = self._call_ollama(_ANSWER_SYSTEM, answer_prompt)
                row_count = result.row_count

            except (ValueError, RuntimeError) as exc:
                # SQL was unsafe or failed — fall back to schema-only
                logger.warning("SQL execution failed (%s), falling back to schema-only.", exc)
                answer    = self._schema_only_answer(question, schema)
                sql       = ""
                row_count = 0

        # ── Step 2b: Schema-only answer (no SQL) ──────────────────────────
        else:
            logger.info("LLM returned NO_SQL — answering from schema only.")
            answer    = self._schema_only_answer(question, schema)
            sql       = ""
            row_count = 0

        latency_ms = (time.perf_counter() - t0) * 1000
        return AgentResult(
            answer=answer,
            sql=sql,
            from_cache=False,
            latency_ms=round(latency_ms, 1),
            row_count=row_count,
        )

    def _schema_only_answer(self, question: str, schema: str) -> str:
        """Answer from schema alone when SQL cannot be used."""
        prompt = _SCHEMA_ONLY_USER_TMPL.format(schema=schema, question=question)
        return self._call_ollama(_SCHEMA_ONLY_SYSTEM, prompt)
