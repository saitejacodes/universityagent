import asyncio
import aiohttp
import logging
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Polite bot headers — required to avoid 403s from university web servers.
# The User-Agent string identifies us as a research bot, not a browser scraper,
# which is more honest and often gets better treatment from rate limiters.
HEADERS = {
    "User-Agent": "UniversityResearchBot/1.0 (academic; contact@example.com)"
}

# Exponential backoff: wait 2s, then 4s, then 8s before giving up.
# Universities use Cloudflare / Akamai WAFs that throttle burst traffic.
# Three attempts covers transient 5xx errors without hammering the server.
RETRY_DELAYS = [2, 4, 8]


class Scraper:
    def __init__(self) -> None:
        # We cache robots.txt per domain (not per URL) because robots.txt is a
        # domain-wide policy. Re-fetching it for every individual URL on the same
        # domain wastes one HTTP round-trip per page — that's 6+ extra requests
        # per university run that add latency and load without any benefit.
        self._robots_cache: dict[str, RobotFileParser] = {}

        # A single shared aiohttp session is reused across all fetches.
        # aiohttp's own docs explicitly warn: "Don't create a session per request."
        # Each new ClientSession tears down and recreates the underlying TCP connection
        # pool + SSL handshake. For 6 pages × 3 universities that would be 18 redundant
        # SSL handshakes instead of one persistent connection per host.
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return the shared session, lazily creating it on first call."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=HEADERS)
        return self._session

    async def close(self) -> None:
        """
        Cleanly shut down the shared session.
        Must be called in a finally block in main() so the event loop
        doesn't emit 'Unclosed client session' warnings on exit.
        """
        if self._session and not self._session.closed:
            await self._session.close()

    def _can_fetch(self, url: str) -> bool:
        """
        Check robots.txt before fetching any URL.
        This is not just courtesy — many university sites explicitly disallow
        automated crawling of fee/admissions pages. Respecting robots.txt avoids
        IP bans that would break the entire pipeline mid-run.
        """
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots_cache:
            rp = RobotFileParser()
            rp.set_url(f"{base}/robots.txt")
            try:
                rp.read()
            except Exception:
                # If robots.txt is unreachable (network error, 404), allow by default.
                # Blocking on an unreachable robots.txt would silently skip valid pages.
                rp.allow_all = True
            self._robots_cache[base] = rp
        return self._robots_cache[base].can_fetch(HEADERS["User-Agent"], url)

    async def fetch(self, url: str) -> str | None:
        """Fetch a URL with retry + backoff. Returns cleaned plain text or None."""
        if not self._can_fetch(url):
            logger.warning(f"robots.txt disallows: {url}")
            return None

        session = await self._get_session()

        for attempt, delay in enumerate(RETRY_DELAYS, 1):
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        soup = BeautifulSoup(html, "lxml")
                        # Strip script/style/nav/footer tags before extracting text.
                        # These contain boilerplate (cookie banners, menus, tracking JS)
                        # that consumes token budget without containing useful data.
                        for tag in soup(["script", "style", "nav", "footer"]):
                            tag.decompose()
                        # Cap at 8,000 chars. llama3-70b-8192 has an 8,192-token context.
                        # The extraction prompt template itself uses ~400 tokens,
                        # leaving ~1,400 tokens for the JSON response. 8,000 chars ≈ 2,000
                        # tokens — the extractor trims this further to 6,000 chars before
                        # sending, giving comfortable headroom for the reply.
                        return soup.get_text(separator=" ", strip=True)[:8000]
                    elif resp.status in (403, 429):
                        logger.warning(f"Rate limited {resp.status}, waiting 30s")
                        await asyncio.sleep(30)
                    else:
                        logger.warning(f"HTTP {resp.status} for {url}")
            except Exception as e:
                logger.error(f"Attempt {attempt} failed for {url}: {e}")
            await asyncio.sleep(delay)

        logger.error(f"All retries failed for {url}")
        return None

    async def fetch_all(self, pages: dict[str, str]) -> dict[str, str | None]:
        """
        Fetch all pages for a university. Returns label → text dict.
        Sequential (not concurrent) to honour polite crawling etiquette —
        concurrent fetches to the same host look like a DDoS to WAFs.
        """
        results: dict[str, str | None] = {}
        for label, url in pages.items():
            logger.info(f"Fetching [{label}]: {url}")
            results[label] = await self.fetch(url)
            await asyncio.sleep(2)  # 2-second gap between requests per host
        return results
