"""Fetcher policy tests with a fake transport (no network)."""

from dataclasses import dataclass, field

import pytest

from uniagent.fetch import fetcher as F

PAGE = (
    "<html><body><h1>Fees</h1>"
    + "<p>International tuition is $60,000 per year.</p>" * 20
    + "</body></html>"
)
CHALLENGE = "<html><head><title>Just a moment...</title></head><body>Enable JavaScript and cookies to continue</body></html>"


@dataclass
class FakeResp:
    status: int
    text: str = ""
    headers: dict = field(default_factory=dict)
    url: str = ""
    encoding: str = "utf-8"

    @property
    def body(self) -> bytes:
        return self.text.encode()


def install(monkeypatch, routes: dict[str, FakeResp]) -> list[str]:
    seen: list[str] = []

    async def fake_get(url, **kw):
        seen.append(url)
        for suffix, resp in routes.items():
            if url.endswith(suffix):
                resp.url = url
                return resp
        return FakeResp(404)

    monkeypatch.setattr(F.AsyncFetcher, "get", staticmethod(fake_get))
    return seen


def make() -> F.PoliteFetcher:
    return F.PoliteFetcher(
        "Mozilla/5.0 (compatible; uniagent/2.0)", min_interval_s=0, browser_fallback=False
    )


async def test_robots_disallow_is_obeyed(monkeypatch):
    seen = install(
        monkeypatch,
        {
            "/robots.txt": FakeResp(200, "User-agent: *\nDisallow: /fees"),
            "/fees": FakeResp(200, PAGE),
        },
    )
    res = await make().fetch("https://u.edu/fees")
    assert res.outcome == "robots_disallowed"
    assert "https://u.edu/fees" not in seen  # never requested


async def test_robots_rules_for_our_token(monkeypatch):
    install(
        monkeypatch,
        {
            "/robots.txt": FakeResp(200, "User-agent: uniagent\nDisallow: /\n"),
            "/fees": FakeResp(200, PAGE),
        },
    )
    assert (await make().fetch("https://u.edu/fees")).outcome == "robots_disallowed"


@pytest.mark.parametrize("status,expected", [(404, "ok"), (503, "robots_disallowed")])
async def test_rfc9309_status_semantics(monkeypatch, status, expected):
    install(monkeypatch, {"/robots.txt": FakeResp(status), "/fees": FakeResp(200, PAGE)})
    assert (await make().fetch("https://u.edu/fees")).outcome == expected


async def test_challenge_pages_are_recorded_not_bypassed(monkeypatch):
    install(monkeypatch, {"/robots.txt": FakeResp(404), "/fees": FakeResp(403, CHALLENGE)})
    res = await make().fetch("https://u.edu/fees")
    assert res.outcome == "blocked" and res.markdown == ""


async def test_not_modified(monkeypatch):
    install(monkeypatch, {"/robots.txt": FakeResp(404), "/fees": FakeResp(304)})
    res = await make().fetch("https://u.edu/fees", etag='"abc"')
    assert res.outcome == "not_modified"


async def test_ok_page_is_cleaned(monkeypatch):
    install(monkeypatch, {"/robots.txt": FakeResp(404), "/fees": FakeResp(200, PAGE)})
    res = await make().fetch("https://u.edu/fees")
    assert res.outcome == "ok" and "$60,000" in res.markdown and "<p>" not in res.markdown


def test_cleaner_keeps_content_nested_inside_nav_by_broken_markup():
    from uniagent.fetch.clean import html_to_markdown

    html = (
        "<html><body><nav role='navigation'><ul><li>Home</li></ul>"
        "<main><h1>Living costs</h1><p>For 2026/27 we estimate £1,200 per month.</p>"
        + "<p>More detail about budgets.</p>" * 30
        + "</main></nav><footer>© University</footer></body></html>"
    )
    md = html_to_markdown(html)
    assert "£1,200 per month" in md and "© University" not in md


def test_cleaner_still_drops_ordinary_navigation():
    from uniagent.fetch.clean import html_to_markdown

    html = (
        "<html><body><nav><a>Home</a><a>Study</a><a>Research</a></nav>"
        "<div><p>Tuition is $60,000 per year.</p></div></body></html>"
    )
    md = html_to_markdown(html)
    assert "$60,000" in md and "Research" not in md


def test_cleaner_keeps_collapsed_accordion_content():
    from uniagent.fetch.clean import html_to_markdown

    html = (
        "<html><body><h2>Key dates</h2><button aria-expanded='false'>International</button>"
        "<div class='accordion-panel' aria-hidden='true' hidden>"
        "<p>Semester 1: 30 November of the previous year</p></div>"
        "<div aria-hidden='true'><p>40,531 Total students</p></div></body></html>"
    )
    md = html_to_markdown(html)
    assert "30 November" in md and "40,531" in md
