"""Settings (env prefix UNIAGENT_) and the university registry (config/universities.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from uniagent.schema import PageType


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UNIAGENT_", env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:8b"
    llm_api_key: str = Field(default="", repr=False)
    llm_max_concurrency: int = 2
    llm_cache_path: Path = Path(".cache/llm.sqlite")

    data_dir: Path = Path("data")
    user_agent: str = (
        "Mozilla/5.0 (compatible; uniagent/2.0; +https://github.com/saitejacodes/universityagent)"
    )
    min_domain_interval_s: float = 2.0  # polite gap between requests to the same host
    fetch_timeout_s: float = 30.0
    use_browser_fallback: bool = True  # render JS-only pages with a headless browser

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "universities.db"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if not s.llm_api_key:
        # backwards compatible with v1's .env
        import os

        from dotenv import dotenv_values

        env = {**dotenv_values(".env"), **os.environ}
        s.llm_api_key = env.get("GROQ_API_KEY", "") or ""
    return s


@dataclass(frozen=True)
class University:
    id: str
    name: str
    country: str
    currency: str
    pages: dict[PageType, str]
    render: frozenset[PageType] = frozenset()  # pages that need JavaScript rendering


def load_universities(path: str | Path = "config/universities.yaml") -> list[University]:
    raw = yaml.safe_load(Path(path).read_text())
    out = []
    for u in raw["universities"]:
        # a page is either "url" or {"url": ..., "render": true} for JavaScript-built pages
        pages, render = {}, set()
        for k, v in (u.get("pages") or {}).items():
            if not v:
                continue
            if isinstance(v, dict):
                pages[PageType(k)] = v["url"]
                if v.get("render"):
                    render.add(PageType(k))
            else:
                pages[PageType(k)] = v
        out.append(
            University(u["id"], u["name"], u["country"], u["currency"], pages, frozenset(render))
        )
    return out
