"""
llm/model.py
────────────
Thin wrapper around a locally running Ollama instance.

Ollama exposes a REST API on localhost:11434.
The selected model (default: llama3.2 or mistral) is used for all generation.

No external API calls are made. Everything runs on-device.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Iterator

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL  = os.getenv("OLLAMA_MODEL",    "llama3.2")   # or "mistral", "phi3", etc.

# System prompt — strict grounding to prevent hallucination
SYSTEM_PROMPT = """\
You are a precise database assistant. Your ONLY job is to answer questions \
about the database using the context passages provided below.

STRICT RULES — follow all of them:
1. Answer ONLY from the provided context. Do not use outside knowledge.
2. If the context contains the answer, state it clearly and directly.
3. If the context does NOT contain the answer, reply with exactly:
   "I don't have that information in the knowledge base."
   Do not guess, invent, or speculate.
4. For yes/no questions (e.g. "do we have a team named X?"):
   - Answer "Yes" or "No" based on whether the value appears in the context.
   - If you see a row or value matching the question, answer "Yes" and quote it.
5. For listing questions (e.g. "what are all the policies?"):
   - List every relevant item found in the context.
6. Never mention LinkedIn, external websites, or anything unrelated to the database.
7. Be concise. Do not pad the answer.
"""

# Maximum characters of context to send to the model (prevents token overflow)
MAX_CONTEXT_CHARS = 6000   # increased to fit more row-level docs


@dataclass
class LLMConfig:
    model:        str   = DEFAULT_MODEL
    temperature:  float = 0.1     # low temp → factual, deterministic answers
    num_predict:  int   = 512     # max tokens to generate
    top_p:        float = 0.9
    ollama_url:   str   = OLLAMA_BASE
    # Timeout for a single Ollama generate call (seconds).
    # First call after a cold start can be slow — 300s is safe for llama3.2.
    # Increase further if using a large model (7B+) on CPU.
    request_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", 300))
    # keep_alive tells Ollama how long to keep the model in RAM after a call.
    # "10m" prevents eviction when another model is also loaded.
    keep_alive:   str   = os.getenv("OLLAMA_KEEP_ALIVE", "10m")


class LocalLLM:
    """
    Wraps the Ollama /api/generate endpoint.

    Usage:
        llm = LocalLLM()
        llm.check_connection()   # raises if Ollama isn't running
        answer = llm.ask("How many users are there?", context_chunks)
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    # ── Connectivity ──────────────────────────────────────────────────────

    def check_connection(self) -> None:
        """Raise RuntimeError if Ollama is not reachable or model not found."""
        try:
            resp = requests.get(f"{self.config.ollama_url}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.config.ollama_url}. "
                "Make sure Ollama is running (`ollama serve`)."
            ) from exc

        models = [m["name"] for m in resp.json().get("models", [])]
        # Accept both "llama3.2" and "llama3.2:latest"
        base_names = [m.split(":")[0] for m in models]
        model_base = self.config.model.split(":")[0]
        if model_base not in base_names:
            logger.warning(
                "Model '%s' not found in Ollama. Available: %s. "
                "Pull it with: ollama pull %s",
                self.config.model, models, self.config.model
            )

    # ── Prompt building ───────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(question: str, context_chunks: list[str]) -> str:
        """Assemble the full prompt from context chunks + question."""
        # Concatenate chunks, truncate to avoid blowing token budget
        context = "\n\n---\n\n".join(context_chunks)
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n[...context truncated...]"

        prompt = (
            f"Context information:\n"
            f"===================\n"
            f"{context}\n"
            f"===================\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        return prompt

    # ── Generation ────────────────────────────────────────────────────────

    def ask(self, question: str, context_chunks: list[str]) -> str:
        """
        Send question + context to Ollama and return the complete answer string.
        Uses the non-streaming endpoint for simplicity.
        """
        prompt = self._build_prompt(question, context_chunks)

        payload = {
            "model":      self.config.model,
            "system":     SYSTEM_PROMPT,
            "prompt":     prompt,
            "stream":     False,
            "keep_alive": self.config.keep_alive,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
                "top_p":       self.config.top_p,
            },
        }

        try:
            resp = requests.post(
                f"{self.config.ollama_url}/api/generate",
                json=payload,
                timeout=self.config.request_timeout,
            )
            resp.raise_for_status()
        except requests.Timeout:
            raise RuntimeError(
                f"Ollama request timed out after {self.config.request_timeout}s. "
                "The model may still be loading into RAM. "
                f"Try increasing OLLAMA_TIMEOUT (current: {self.config.request_timeout}s) "
                "or wait a moment and retry."
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        data = resp.json()
        return data.get("response", "").strip()

    def ask_stream(self, question: str, context_chunks: list[str]) -> Iterator[str]:
        """
        Streaming version — yields text tokens as they arrive.
        Useful for future SSE/WebSocket support.
        """
        prompt = self._build_prompt(question, context_chunks)

        payload = {
            "model":  self.config.model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
            },
        }

        with requests.post(
            f"{self.config.ollama_url}/api/generate",
            json=payload,
            stream=True,
            timeout=self.config.request_timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get("response", "")
                    if chunk.get("done"):
                        break
