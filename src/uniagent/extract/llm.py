"""LLM client for OpenAI-compatible endpoints (Groq, OpenAI, vLLM) and Ollama's native API.

Features that matter for evaluation, not just demos:
- a persistent response cache keyed on the full request, so re-running a benchmark is free and
  bit-for-bit reproducible, and an ablation only pays for the calls that actually changed;
- bounded concurrency plus retry with exponential backoff that honours `Retry-After` (429s);
- token + latency accounting per call, so reports can show cost alongside accuracy.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False


class ResponseCache:
    def __init__(self, path: Path | None):
        self._lock = threading.Lock()
        self._conn = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, body TEXT NOT NULL,"
                " created REAL NOT NULL)"
            )
            self._conn.commit()

    def get(self, key: str) -> dict | None:
        if self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute("SELECT body FROM responses WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, body: dict) -> None:
        if self._conn is None:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?,?,?)",
                (key, json.dumps(body), time.time()),
            )
            self._conn.commit()


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_s: float = 120.0,
        max_concurrency: int = 2,
        cache_path: Path | None = None,
        num_ctx: int = 8192,
        max_retries: int = 6,
        think: bool | str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.native_ollama = self.base_url.endswith(":11434") or self.base_url.endswith("/api")
        if self.base_url.endswith(":11434/v1"):
            # Ollama's OpenAI shim ignores num_ctx and silently truncates long prompts to the
            # default context; the native API lets us set it, so prefer it.
            self.base_url = self.base_url[: -len("/v1")]
            self.native_ollama = True
        self.api_key = api_key
        self.num_ctx = num_ctx
        self.max_retries = max_retries
        # Ollama: False disables qwen3-style thinking; 'low'/'medium'/'high' sets gpt-oss effort.
        self.think = think
        self._sem = asyncio.Semaphore(max_concurrency)
        self._cache = ResponseCache(cache_path)
        self._http = httpx.AsyncClient(timeout=timeout_s)
        self.stats = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}

    async def aclose(self) -> None:
        await self._http.aclose()

    def _key(self, messages: list[dict], temperature: float, seed: int, max_tokens: int) -> str:
        payload = json.dumps(
            [self.model, messages, round(temperature, 4), seed, max_tokens, self.think],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        seed: int = 0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        key = self._key(messages, temperature, seed, max_tokens) + ("j" if json_mode else "")
        if (hit := self._cache.get(key)) is not None:
            self.stats["cache_hits"] += 1
            return LLMResponse(
                hit["text"], hit.get("pt", 0), hit.get("ct", 0), hit.get("ms", 0.0), cached=True
            )

        async with self._sem:
            t0 = time.monotonic()
            body = await self._request_with_retries(
                messages, temperature, seed, max_tokens, json_mode
            )
            ms = (time.monotonic() - t0) * 1000
        text, pt, ct = self._parse(body)
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += pt
        self.stats["completion_tokens"] += ct
        self._cache.put(key, {"text": text, "pt": pt, "ct": ct, "ms": ms})
        return LLMResponse(text, pt, ct, ms)

    async def _request_with_retries(self, messages, temperature, seed, max_tokens, json_mode):
        if self.native_ollama:
            url = f"{self.base_url}/api/chat"
            payload: dict = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": temperature,
                    "seed": seed,
                    "num_ctx": self.num_ctx,
                    "num_predict": max_tokens,
                },
            }
            if json_mode:
                payload["format"] = "json"
            if self.think is not None:
                payload["think"] = self.think
            headers = {}
        else:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "seed": seed,
                "max_tokens": max_tokens,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            headers = {"Authorization": f"Bearer {self.api_key}"}

        delay = 2.0
        last_err = ""
        for attempt in range(self.max_retries):
            try:
                resp = await self._http.post(url, json=payload, headers=headers)
            except httpx.TransportError as exc:
                last_err = f"transport error: {exc}"
            else:
                if resp.status_code == 200:
                    return resp.json()
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in (408, 409, 425, 429, 500, 502, 503, 504):
                    raise LLMError(last_err)
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    with contextlib.suppress(ValueError):
                        delay = max(delay, float(retry_after))
            if attempt < self.max_retries - 1:
                await asyncio.sleep(delay + random.uniform(0, delay / 2))
                delay = min(delay * 2, 60.0)
        raise LLMError(f"gave up after {self.max_retries} attempts: {last_err}")

    def _parse(self, body: dict) -> tuple[str, int, int]:
        if self.native_ollama:
            text = body.get("message", {}).get("content", "")
            return text, body.get("prompt_eval_count", 0), body.get("eval_count", 0)
        text = body["choices"][0]["message"].get("content") or ""
        usage = body.get("usage") or {}
        return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
