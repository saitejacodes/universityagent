"""Versioned page snapshots.

Every fetched page is stored as raw HTML + cleaned Markdown + metadata under
    data/snapshots/<university>/<page_type>.{html,md,json}
and when the content changes, the previous Markdown is kept under history/ with its hash.
Extraction and evaluation read snapshots, never the live web, which makes runs reproducible:
gold labels are written against a specific snapshot hash, and a changed page shows up as a
changed hash instead of a silently different answer.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from uniagent.fetch.fetcher import FetchResult


@dataclass
class Snapshot:
    university: str
    page_type: str
    url: str
    final_url: str
    sha: str
    fetched_at: str
    method: str
    status: int
    etag: str | None = None
    last_modified: str | None = None
    changed: bool = False

    def markdown(self, root: Path) -> str:
        return (root / self.university / f"{self.page_type}.md").read_text()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class SnapshotStore:
    def __init__(self, root: Path):
        self.root = root

    def _paths(self, uni: str, page: str) -> tuple[Path, Path, Path]:
        d = self.root / uni
        return d / f"{page}.html", d / f"{page}.md", d / f"{page}.json"

    def get(self, uni: str, page: str) -> Snapshot | None:
        _, md, meta = self._paths(uni, page)
        if not meta.exists() or not md.exists():
            return None
        return Snapshot(**json.loads(meta.read_text()))

    def all_for(self, uni: str) -> dict[str, Snapshot]:
        d = self.root / uni
        out = {}
        for meta in sorted(d.glob("*.json")) if d.is_dir() else []:
            snap = self.get(uni, meta.stem)
            if snap:
                out[meta.stem] = snap
        return out

    def save(self, uni: str, page: str, res: FetchResult) -> Snapshot:
        html_p, md_p, meta_p = self._paths(uni, page)
        md_p.parent.mkdir(parents=True, exist_ok=True)
        prev = self.get(uni, page)
        now = datetime.now(UTC).isoformat(timespec="seconds")

        if res.outcome == "not_modified" and prev:
            prev.fetched_at, prev.changed = now, False
            meta_p.write_text(json.dumps(asdict(prev), indent=1))
            return prev

        sha = _sha(res.markdown)
        changed = prev is not None and prev.sha != sha
        if changed:
            hist = md_p.parent / "history"
            hist.mkdir(exist_ok=True)
            shutil.copy(md_p, hist / f"{page}-{prev.sha}.md")
        html_p.write_text(res.html)
        md_p.write_text(res.markdown)
        snap = Snapshot(
            uni,
            page,
            res.url,
            res.final_url,
            sha,
            now,
            res.method,
            res.status,
            res.etag,
            res.last_modified,
            changed,
        )
        meta_p.write_text(json.dumps(asdict(snap), indent=1))
        return snap

    def reclean(self, uni: str, page: str) -> Snapshot | None:
        """Re-run the HTML cleaner on the stored HTML (no network). Used after a cleaner fix so
        extraction and gold labels see the improved text; the old Markdown goes to history/."""
        from uniagent.fetch.clean import html_to_markdown

        html_p, _, _ = self._paths(uni, page)
        prev = self.get(uni, page)
        if prev is None or not html_p.exists():
            return None
        res = FetchResult(
            prev.url,
            prev.final_url,
            prev.status,
            html=html_p.read_text(),
            method=prev.method,
            etag=prev.etag,
            last_modified=prev.last_modified,
        )
        res.markdown = html_to_markdown(res.html, prev.final_url)
        if len(res.markdown) < len(prev.markdown(self.root)) * 0.5:
            return prev  # never let a re-clean silently lose most of a page
        snap = self.save(uni, page, res)
        snap.fetched_at = prev.fetched_at  # content age is unchanged
        return snap
