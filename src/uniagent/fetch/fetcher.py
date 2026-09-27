"""Polite fetching with Scrapling.

- robots.txt is fetched once per host and obeyed, including Crawl-delay;
- requests to the same host are serialised with a minimum gap, while different hosts proceed
  in parallel (v1 slept 2 s between *every* request, even across universities);
- conditional requests (If-None-Match / If-Modified-Since) make re-crawls cheap: a 304 means
  the stored snapshot is still current;
- pages whose static HTML has almost no text (JS-rendered) are re-fetched with Scrapling's
  DynamicFetcher (a normal headless Chromium). We deliberately do NOT use Scrapling's
  anti-bot/stealth features: a 403 or challenge page is recorded as `blocked`, not bypassed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from scrapling.fetchers import AsyncFetcher

from uniagent.fetch.clean import html_to_markdown

log = logging.getLogger(__name__)

_MIN_TEXT_CHARS = 400  # below this the static HTML is probably a JS shell
_CHALLENGE_MARKERS = (
    "just a moment",
    "attention required",
    "access denied",
    "verify you are human",
    "enable javascript and cookies",
)


@dataclass
class FetchResult:
    url: str
    final_url: str
    status: int
    html: str = ""
    markdown: str = ""
    method: str = "http"  # http | browser | not_modified
    outcome: str = "ok"  # ok | not_modified | blocked | robots_disallowed | error | js_only
    etag: str | None = None
    last_modified: str | None = None
    error: str | None = None
    elapsed_ms: float = 0.0


class PoliteFetcher:
    def __init__(
        self,
        user_agent: str,
        min_interval_s: float = 2.0,
        timeout_s: float = 30.0,
        browser_fallback: bool = True,
    ):
        self.ua = user_agent
        # robots.txt groups are matched on the product token, not the full header value
        self.robots_token = "uniagent"
        self.min_interval = min_interval_s
        self.timeout = timeout_s
        self.browser_fallback = browser_fallback
        self._robots: dict[str, RobotFileParser | None] = {}
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._last_hit: dict[str, float] = {}
        self._browser_sem = asyncio.Semaphore(1)

    def _headers(self, extra: dict | None = None) -> dict:
        h = {"User-Agent": self.ua, "Accept-Language": "en;q=0.9"}
        h.update(extra or {})
        return h

    async def _throttle(self, host: str, delay: float) -> None:
        lock = self._host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = self._last_hit.get(host, 0) + delay - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_hit[host] = time.monotonic()

    async def _robots_for(self, url: str) -> RobotFileParser | None:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base in self._robots:
            return self._robots[base]
        rp: RobotFileParser | None = RobotFileParser()
        try:
            await self._throttle(p.netloc, self.min_interval)
            resp = await AsyncFetcher.get(
                f"{base}/robots.txt",
                headers=self._headers(),
                timeout=self.timeout,
                follow_redirects=True,
                stealthy_headers=False,
                impersonate=None,
            )
            # RFC 9309 §2.3.1: 2xx -> parse; 4xx ("unavailable") -> no restrictions;
            # 5xx / unreachable -> assume complete disallow.
            if resp.status == 200:
                rp.parse(resp.body.decode("utf-8", errors="replace").splitlines())
            elif 400 <= resp.status < 500:
                rp = None
            else:
                rp.disallow_all = True
        except Exception as exc:
            log.warning("robots.txt unreachable for %s: %s", base, exc)
            rp.disallow_all = True
        self._robots[base] = rp
        return rp

    async def fetch(
        self,
        url: str,
        etag: str | None = None,
        last_modified: str | None = None,
        render: bool = False,
    ) -> FetchResult:
        """Fetch one page. `render=True` also renders it in a headless browser (for pages whose
        facts are filled in by JavaScript, e.g. fee calculators) and keeps the richer text."""
        t0 = time.monotonic()
        host = urlparse(url).netloc
        rp = await self._robots_for(url)
        if rp is not None and not rp.can_fetch(self.robots_token, url):
            return FetchResult(url, url, 0, outcome="robots_disallowed")
        delay = max(self.min_interval, float(rp.crawl_delay(self.robots_token) or 0) if rp else 0)

        cond = {}
        if etag:
            cond["If-None-Match"] = etag
        if last_modified:
            cond["If-Modified-Since"] = last_modified

        await self._throttle(host, delay)
        try:
            resp = await AsyncFetcher.get(
                url,
                headers=self._headers(cond),
                timeout=self.timeout,
                follow_redirects=True,
                stealthy_headers=False,
                impersonate=None,
                retries=2,
                retry_delay=3,
            )
        except Exception as exc:
            return FetchResult(
                url,
                url,
                0,
                outcome="error",
                error=str(exc),
                elapsed_ms=(time.monotonic() - t0) * 1000,
            )

        headers = {k.lower(): v for k, v in (resp.headers or {}).items()}
        res = FetchResult(
            url,
            str(resp.url or url),
            resp.status,
            etag=headers.get("etag"),
            last_modified=headers.get("last-modified"),
        )
        if resp.status == 304:
            res.outcome, res.method = "not_modified", "not_modified"
            return res
        html = resp.body.decode(resp.encoding or "utf-8", errors="replace") if resp.body else ""
        if resp.status in (401, 403, 429) or _looks_like_challenge(html):
            res.outcome, res.error = "blocked", f"HTTP {resp.status}"
            return res
        if resp.status >= 400:
            res.outcome, res.error = "error", f"HTTP {resp.status}"
            return res

        res.html, res.markdown = html, html_to_markdown(html, res.final_url)
        if render or len(res.markdown) < _MIN_TEXT_CHARS:
            if self.browser_fallback:
                await self._render(res, host, delay)
            else:
                res.outcome = "js_only"
        res.elapsed_ms = (time.monotonic() - t0) * 1000
        return res

    async def _render(self, res: FetchResult, host: str, delay: float) -> None:
        try:
            from scrapling.fetchers import DynamicFetcher
        except ImportError:
            res.outcome = "js_only"
            return
        async with self._browser_sem:
            await self._throttle(host, delay)
            try:
                page = await DynamicFetcher.async_fetch(
                    res.url,
                    headless=True,
                    network_idle=True,
                    timeout=int(self.timeout * 1000),
                    useragent=self.ua,
                    disable_resources=True,
                    block_ads=True,
                )
            except Exception as exc:
                res.outcome, res.error = "js_only", f"browser render failed: {exc}"
                return
        html = page.body.decode("utf-8", errors="replace") if page.body else ""
        if _looks_like_challenge(html):
            res.outcome, res.error = "blocked", "challenge page"
            return
        md = html_to_markdown(html, res.final_url)
        if len(md) > len(res.markdown):
            res.html, res.markdown, res.method = html, md, "browser"
        if len(res.markdown) < _MIN_TEXT_CHARS:
            res.outcome = "js_only"


def _looks_like_challenge(html: str) -> bool:
    head = html[:5000].lower()
    return len(html) < 20000 and any(m in head for m in _CHALLENGE_MARKERS)
