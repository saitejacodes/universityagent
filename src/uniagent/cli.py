"""uniagent CLI: crawl, extract, eval, show."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer

from uniagent.config import get_settings, load_universities

app = typer.Typer(add_completion=False, help="Evidence-grounded university data extraction.")

PRESETS = {
    # v1 behaviour: first 4,000 characters of each page, no evidence verification
    "v1": dict(retrieval=False, grounding=False),
    # + per-field BM25 retrieval over the whole page
    "retrieval": dict(retrieval=True, grounding=False),
    # + drop answers whose quote is not on the page (full system)
    "full": dict(retrieval=True, grounding=True),
}


def _unis(only: str):
    unis = load_universities()
    if only:
        wanted = set(only.split(","))
        unis = [u for u in unis if u.id in wanted]
    return unis


@app.command()
def crawl(only: str = typer.Option("", help="Comma-separated university ids")):
    """Fetch all configured pages into versioned snapshots (polite, conditional GETs)."""
    from uniagent.pipeline import crawl as do_crawl

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    report = asyncio.run(do_crawl(_unis(only), get_settings()))
    by = {}
    for r in report:
        by[r["outcome"]] = by.get(r["outcome"], 0) + 1
    Path("data").mkdir(exist_ok=True)
    Path("data/crawl_report.json").write_text(json.dumps(report, indent=1))
    typer.echo(f"{len(report)} pages: {by}  (details: data/crawl_report.json)")


@app.command()
def reclean(only: str = typer.Option("", help="Comma-separated university ids")):
    """Re-run the HTML cleaner over stored snapshots (after a cleaner fix). No network."""
    from uniagent.fetch.snapshots import SnapshotStore

    store = SnapshotStore(get_settings().snapshots_dir)
    changed = 0
    for u in _unis(only):
        for page in store.all_for(u.id):
            before = store.get(u.id, page)
            after = store.reclean(u.id, page)
            if after and before and after.sha != before.sha:
                changed += 1
                typer.echo(f"{u.id}/{page}: {before.sha} -> {after.sha}")
    typer.echo(f"{changed} snapshot(s) changed")


@app.command()
def extract(
    preset: str = typer.Option("full", help="v1 | retrieval | full"),
    run_name: str = typer.Option("", help="Defaults to <preset>-<model>"),
    model: str = typer.Option("", help="Override UNIAGENT_LLM_MODEL"),
    base_url: str = typer.Option("", help="Override UNIAGENT_LLM_BASE_URL"),
    only: str = typer.Option("", help="Comma-separated university ids"),
    concurrency: int = typer.Option(2),
    think: str = typer.Option("", help="Ollama think setting: false | low | medium | high"),
):
    """Extract all fields from stored snapshots into runs/<name>/predictions.json."""
    from uniagent.extract.extractor import ExtractorOptions
    from uniagent.extract.llm import LLMClient
    from uniagent.pipeline import extract as do_extract

    s = get_settings()
    model = model or s.llm_model
    think_val: bool | str | None = None
    if think:
        think_val = False if think.lower() == "false" else think
    llm = LLMClient(
        base_url or s.llm_base_url,
        model,
        s.llm_api_key,
        180.0,
        s.llm_max_concurrency,
        s.llm_cache_path,
        num_ctx=8192,
        think=think_val,
    )
    name = run_name or f"{preset}-{model.replace(':', '_').replace('/', '_')}"

    async def go():
        try:
            return await do_extract(
                _unis(only),
                s,
                llm,
                ExtractorOptions(**PRESETS[preset]),
                name,
                Path("runs") / name,
                concurrency,
            )
        finally:
            await llm.aclose()

    out = asyncio.run(go())
    typer.echo(f"wrote {out}/predictions.json  llm={llm.stats}")


@app.command("eval")
def eval_cmd(
    runs: list[Path] = typer.Argument(..., help="Run folders, in ablation order"),
    gold: Path = typer.Option(Path("data/gold/labels.jsonl")),
    split: str = typer.Option("test", help="test | dev | all"),
    out: Path = typer.Option(Path("EVAL_REPORT.md")),
):
    """Score runs against gold labels and write a Markdown report."""
    from uniagent.eval.report import build_report

    out.write_text(build_report(runs, gold, split))
    typer.echo(f"wrote {out}")


@app.command()
def show(university: str):
    """Print the current stored facts (with evidence) for one university."""
    from uniagent.store.db import Store

    store = Store(get_settings().db_path)
    for row in store.current(university):
        typer.echo(
            f"{row['field']:24} {json.dumps(row['value'])[:50]:52} "
            f'conf={row["confidence"]:.2f}  "{(row["quote"] or "")[:70]}"'
        )


if __name__ == "__main__":
    app()
