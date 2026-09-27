"""Merge URL-discovery results into config/universities.yaml.

    python scripts/build_config.py data/discovery/*.json

Existing entries in the YAML are kept; a discovery-file entry with the same id replaces it
(later files win, e.g. a university re-probed after a cleaner fix). Only pages whose probe
outcome was `ok` (fetchable by our polite fetcher and containing the target fact) are kept.
Universities with too few usable pages are listed under `excluded` with the reason, so coverage
is reported honestly instead of silently dropped. With no arguments it does nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

MIN_PAGES = 3
CONFIG = Path("config/universities.yaml")
HEADER = (
    "# Official pages per university. Every URL was verified with scripts/probe_url.py:\n"
    "# fetchable by the pipeline's own polite fetcher (robots.txt obeyed, no bot-wall evasion)\n"
    "# and containing the target fact. Add a university by adding a block; no code changes.\n"
)


def main(paths: list[str]) -> None:
    if not paths:
        sys.exit(__doc__)
    current = yaml.safe_load(CONFIG.read_text()) if CONFIG.exists() else {}
    unis = {u["id"]: u for u in current.get("universities") or []}
    excluded = {e["id"]: e for e in current.get("excluded") or []}

    for path in paths:
        for u in json.loads(Path(path).read_text()):
            pages = {
                k: ({"url": v["url"], "render": True} if v.get("render") else v["url"])
                for k, v in u["pages"].items()
                if v.get("url") and v.get("probe") == "ok"
            }
            unis.pop(u["id"], None)
            excluded.pop(u["id"], None)
            if len({str(x) for x in pages.values()}) < MIN_PAGES and len(pages) < MIN_PAGES + 1:
                excluded[u["id"]] = {
                    "id": u["id"],
                    "name": u["name"],
                    "usable_pages": len(pages),
                    "probe_outcomes": sorted(
                        {v.get("probe") for v in u["pages"].values() if v.get("probe")}
                    ),
                    "notes": (u.get("notes") or "")[:300],
                }
            else:
                unis[u["id"]] = {
                    "id": u["id"],
                    "name": u["name"],
                    "country": u["country"],
                    "currency": u["currency"],
                    "pages": pages,
                }

    if not unis:
        sys.exit("refusing to write a config with no universities")
    body = yaml.safe_dump(
        {"universities": list(unis.values()), "excluded": list(excluded.values())},
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )
    CONFIG.write_text(HEADER + body)
    print(f"{len(unis)} universities configured, {len(excluded)} excluded ({', '.join(excluded)})")


if __name__ == "__main__":
    main(sys.argv[1:])
