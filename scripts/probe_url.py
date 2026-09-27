"""Check that a URL is fetchable by the pipeline's own fetcher and contains the expected fact.

    python scripts/probe_url.py URL [REGEX ...] [--render]   (--render: headless browser)

Prints HTTP status, outcome (ok / blocked / robots_disallowed / js_only / error), size of the
cleaned Markdown, and each regex hit with surrounding context. Used when curating
config/universities.yaml so every configured page is known to work with the real pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys

from uniagent.config import Settings
from uniagent.fetch.fetcher import PoliteFetcher


async def main(url: str, patterns: list[str], render: bool = False) -> None:
    logging.disable(logging.INFO)
    f = PoliteFetcher(
        Settings().user_agent, min_interval_s=1.0, timeout_s=30, browser_fallback=render
    )
    res = await f.fetch(url, render=render)
    print(
        f"status={res.status} outcome={res.outcome} md_chars={len(res.markdown)} "
        f"final_url={res.final_url} error={res.error}"
    )
    for pat in patterns:
        hits = [m.start() for m in re.finditer(pat, res.markdown, re.IGNORECASE)]
        print(f"  /{pat}/ -> {len(hits)} hit(s)")
        for h in hits[:3]:
            ctx = res.markdown[max(0, h - 100) : h + 120].replace("\n", " ")
            print(f"     …{ctx}…")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    args = [a for a in sys.argv[1:] if a != "--render"]
    asyncio.run(main(args[0], args[1:], render="--render" in sys.argv))
