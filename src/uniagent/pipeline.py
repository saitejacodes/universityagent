"""Orchestration: crawl -> snapshot -> extract -> verify -> merge -> store.

    crawl()    fetch every configured page (conditional GET), store versioned snapshots
    extract()  read snapshots only (never the live web), extract + verify fields, write a run

Crawling and extraction are separate on purpose: extraction runs are reproducible against a
fixed snapshot set, and a re-crawl that finds no changed pages makes the next extraction free
(identical prompts are served from the LLM response cache).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from uniagent.config import Settings, University
from uniagent.extract.extractor import Extraction, Extractor, ExtractorOptions, merge_field
from uniagent.extract.llm import LLMClient
from uniagent.fetch.fetcher import PoliteFetcher
from uniagent.fetch.snapshots import SnapshotStore
from uniagent.schema import FIELDS, PageType, fields_for_page
from uniagent.store.db import Store

log = logging.getLogger(__name__)


async def crawl(unis: list[University], settings: Settings) -> list[dict]:
    fetcher = PoliteFetcher(
        settings.user_agent,
        settings.min_domain_interval_s,
        settings.fetch_timeout_s,
        settings.use_browser_fallback,
    )
    snaps = SnapshotStore(settings.snapshots_dir)
    store = Store(settings.db_path)
    report: list[dict] = []

    async def one_university(u: University) -> None:
        by_url: dict[str, list[PageType]] = {}
        for page, url in u.pages.items():
            by_url.setdefault(url, []).append(page)
        for url, pages in by_url.items():  # sequential within a university (same host)
            prev = snaps.get(u.id, pages[0].value)
            needs_render = any(p in u.render for p in pages)
            res = await fetcher.fetch(
                url,
                # rendered pages are re-rendered every time; a 304 would skip the JS
                etag=prev.etag if prev and not needs_render else None,
                last_modified=prev.last_modified if prev and not needs_render else None,
                render=needs_render,
            )
            for page in pages:
                row = {
                    "university": u.id,
                    "page_type": page.value,
                    "url": url,
                    "outcome": res.outcome,
                    "status": res.status,
                    "method": res.method,
                    "chars": len(res.markdown),
                    "error": res.error,
                }
                if res.outcome in ("ok", "not_modified"):
                    snap = snaps.save(u.id, page.value, res)
                    store.add_snapshot(
                        u.id, page.value, snap.sha, url, snap.fetched_at, snap.method
                    )
                    row.update(sha=snap.sha, changed=snap.changed)
                report.append(row)
                log.info(
                    "%s/%s %s (%s, %d chars)",
                    u.id,
                    page.value,
                    res.outcome,
                    res.method,
                    len(res.markdown),
                )

    # different universities = different hosts, so they can proceed in parallel
    await asyncio.gather(*(one_university(u) for u in unis))
    store.close()
    return report


async def extract(
    unis: list[University],
    settings: Settings,
    llm: LLMClient,
    options: ExtractorOptions,
    run_id: str,
    out_dir: Path,
    concurrency: int = 2,
) -> Path:
    snaps = SnapshotStore(settings.snapshots_dir)
    extractor = Extractor(llm, options)
    sem = asyncio.Semaphore(concurrency)
    all_rows: list[dict] = []
    page_rows: list[dict] = []

    async def page_job(u: University, page: PageType) -> list[Extraction]:
        snap = snaps.get(u.id, page.value)
        fields = fields_for_page(page)
        if snap is None or not fields:
            return []
        async with sem:
            return await extractor.extract_page(
                university=u.id,
                uni_name=u.name,
                country=u.country,
                currency=u.currency,
                page_type=page.value,
                url=snap.final_url,
                snapshot_sha=snap.sha,
                markdown=snap.markdown(settings.snapshots_dir),
                fields=fields,
            )

    jobs = [(u, p) for u in unis for p in PageType]
    results = await asyncio.gather(*(page_job(u, p) for u, p in jobs))

    by_key: dict[tuple[str, str], list[Extraction]] = {}
    for res in results:
        for ex in res:
            by_key.setdefault((ex.university, ex.field), []).append(ex)
            page_rows.append(ex.to_dict())
    for u in unis:
        for f in FIELDS:
            cands = by_key.get((u.id, f.name), [])
            # primary page first so merge prefers it on ties
            cands.sort(
                key=lambda e: (
                    f.pages.index(PageType(e.page_type)) if PageType(e.page_type) in f.pages else 9
                )
            )
            best = merge_field(cands)
            if best is None:
                all_rows.append(
                    {
                        "university": u.id,
                        "field": f.name,
                        "value": None,
                        "status": "no_snapshot",
                        "confidence": 0.0,
                    }
                )
            else:
                all_rows.append(best.to_dict())

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(json.dumps(all_rows, indent=1, default=str))
    (out_dir / "page_extractions.json").write_text(json.dumps(page_rows, indent=1, default=str))
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model": llm.model,
                "options": asdict(options),
                "universities": [u.id for u in unis],
                "llm_stats": llm.stats,
            },
            indent=1,
        )
    )

    store = Store(settings.db_path)
    store.add_run(
        run_id, datetime.now(UTC).isoformat(timespec="seconds"), llm.model, asdict(options)
    )
    store.add_facts(run_id, all_rows)
    store.close()
    return out_dir
