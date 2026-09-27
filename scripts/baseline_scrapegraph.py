"""Off-the-shelf baseline: ScrapeGraphAI's SmartScraperGraph on the *same* stored snapshots, with
the *same* local model and the *same* field definitions, scored by the same evaluator.

ScrapeGraphAI has its own dependency tree (LangChain, Playwright), so it runs in a separate venv:

    uv venv --python 3.12 ../baselines/.venv-sg
    uv pip install --python ../baselines/.venv-sg/bin/python scrapegraphai rapidfuzz \
        pydantic-settings pyyaml markdownify lxml
    SCRAPEGRAPHAI_TELEMETRY_ENABLED=false PYTHONPATH=src \
        ../baselines/.venv-sg/bin/python scripts/baseline_scrapegraph.py --model qwen3:8b

Output: runs/scrapegraph-<model>/predictions.json (same format as `uniagent extract`).
ScrapeGraphAI returns no evidence and no confidence; every non-null answer gets confidence 0.9,
i.e. it is trusted the way v1 trusted its answers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SCRAPEGRAPHAI_TELEMETRY_ENABLED", "false")

from scrapegraphai.graphs import SmartScraperGraph  # noqa: E402

from uniagent.config import load_universities  # noqa: E402
from uniagent.extract.normalize import normalize  # noqa: E402
from uniagent.schema import FIELDS, PageType, fields_for_page  # noqa: E402

FORMATS = {
    "year": "integer year",
    "int": "integer",
    "percent": "number (percent)",
    "money": '{"amount": number, "currency": ISO code, "period": "year|month|week|term"}',
    "date": '"Month Day" as written',
    "text": "short string",
    "choice": "public or private",
    "list": "list of names",
}


def prompt_for(fields, uni, model: str = "") -> str:
    lines = [
        f"You are extracting facts about {uni.name} ({uni.country}). "
        "Use only this page. If the page does not state a fact, use null.",
        "Return a JSON object with exactly these keys:",
    ]
    for f in fields:
        lines.append(f'- "{f.name}" ({FORMATS[f.kind.value]}): {f.instruction}')
    if model.startswith("qwen3"):
        lines.append("/no_think")  # same non-thinking mode our pipeline uses (think=false)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    config = {
        "llm": {
            "model": f"ollama/{args.model}",
            "base_url": "http://localhost:11434",
            "temperature": 0,
            "format": "json",
            "model_tokens": 8192,
        },
        "verbose": False,
        "headless": True,
    }
    unis = load_universities()
    if args.only:
        unis = [u for u in unis if u.id in set(args.only.split(","))]
    snap_root = Path("data/snapshots")
    by_key: dict[tuple[str, str], dict] = {}
    for u in unis:
        for page in PageType:
            html_path = snap_root / u.id / f"{page.value}.html"
            fields = fields_for_page(page)
            if not html_path.exists() or not fields:
                continue
            graph = SmartScraperGraph(
                prompt=prompt_for(fields, u, args.model),
                source=html_path.read_text(),
                config=config,
            )
            try:
                result = graph.run() or {}
            except Exception as exc:  # the baseline gets no special error handling
                print(f"{u.id}/{page.value}: error {exc}")
                result = {}
            if isinstance(result, dict) and "content" in result and len(result) == 1:
                result = result["content"]
            for f in fields:
                raw = result.get(f.name) if isinstance(result, dict) else None
                value = normalize(f, raw, u.currency)
                key = (u.id, f.name)
                # first page (in schema order) with an answer wins
                if value is not None and (key not in by_key or by_key[key]["value"] is None):
                    by_key[key] = {
                        "university": u.id,
                        "field": f.name,
                        "value": value,
                        "raw_value": raw,
                        "confidence": 0.9,
                        "status": "accepted",
                        "page_type": page.value,
                    }
                by_key.setdefault(
                    key,
                    {
                        "university": u.id,
                        "field": f.name,
                        "value": None,
                        "confidence": 0.0,
                        "status": "not_found",
                        "page_type": page.value,
                    },
                )
            print(f"{u.id}/{page.value}: {json.dumps(result, default=str)[:160]}")

    rows = []
    for u in unis:
        for f in FIELDS:
            rows.append(
                by_key.get(
                    (u.id, f.name),
                    {
                        "university": u.id,
                        "field": f.name,
                        "value": None,
                        "confidence": 0.0,
                        "status": "no_snapshot",
                    },
                )
            )
    out = Path("runs") / f"scrapegraph-{args.model.replace(':', '_')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.json").write_text(json.dumps(rows, indent=1, default=str))
    (out / "config.json").write_text(
        json.dumps(
            {"model": f"scrapegraphai/{args.model}", "system": "ScrapeGraphAI SmartScraperGraph"}
        )
    )
    print(f"wrote {out}/predictions.json")


if __name__ == "__main__":
    main()
