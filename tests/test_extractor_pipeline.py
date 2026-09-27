"""End-to-end extraction on a stored snapshot with a scripted LLM (no network)."""

import json

from uniagent.config import Settings, University
from uniagent.extract.extractor import Extractor, ExtractorOptions
from uniagent.extract.llm import LLMResponse
from uniagent.fetch.fetcher import FetchResult
from uniagent.fetch.snapshots import SnapshotStore
from uniagent.pipeline import extract
from uniagent.schema import FIELD_BY_NAME as F
from uniagent.schema import PageType

FACTS_MD = """# Facts
MIT was founded in 1861 in Boston and moved to Cambridge in 1916.
Enrollment: 11,816 students (4,561 undergraduates, 7,255 graduate students).
MIT is a private research university.
"""


class FakeLLM:
    model = "fake"

    def __init__(self, reply: dict):
        self.reply = reply
        self.stats = {"calls": 0}

    async def chat(self, messages, **kw):
        self.stats["calls"] += 1
        return LLMResponse(json.dumps(self.reply))

    async def aclose(self):
        pass


REPLY = {
    "founding_year": {"value": 1861, "quote": "MIT was founded in 1861 in Boston"},
    "city": {"value": "Cambridge", "quote": "moved to Cambridge in 1916"},
    "institution_type": {"value": "private", "quote": "MIT is a private research university."},
    # hallucinated: this sentence is not on the page
    "total_enrollment": {"value": 12000, "quote": "MIT enrolls about 12,000 students."},
}


async def test_extractor_verifies_evidence():
    ex = Extractor(FakeLLM(REPLY))
    fields = [F[n] for n in ("founding_year", "city", "institution_type", "total_enrollment")]
    out = {
        e.field: e
        for e in await ex.extract_page(
            university="mit",
            uni_name="MIT",
            country="USA",
            currency="USD",
            page_type="facts",
            url="u",
            snapshot_sha="s",
            markdown=FACTS_MD,
            fields=fields,
        )
    }
    assert out["founding_year"].value == 1861 and out["founding_year"].status == "accepted"
    assert out["founding_year"].confidence == 0.9
    assert out["total_enrollment"].value is None
    assert out["total_enrollment"].status == "rejected_ungrounded"


async def test_without_grounding_the_hallucination_survives():
    ex = Extractor(FakeLLM(REPLY), ExtractorOptions(grounding=False))
    out = {
        e.field: e
        for e in await ex.extract_page(
            university="mit",
            uni_name="MIT",
            country="USA",
            currency="USD",
            page_type="facts",
            url="u",
            snapshot_sha="s",
            markdown=FACTS_MD,
            fields=[F["total_enrollment"]],
        )
    }
    assert out["total_enrollment"].value == 12000


async def test_pipeline_extract_from_snapshots(tmp_path):
    settings = Settings(data_dir=tmp_path / "data")
    SnapshotStore(settings.snapshots_dir).save(
        "mit",
        "facts",
        FetchResult(
            "https://facts.mit.edu", "https://facts.mit.edu", 200, html="<html/>", markdown=FACTS_MD
        ),
    )
    uni = University("mit", "MIT", "USA", "USD", {PageType.FACTS: "https://facts.mit.edu"})
    out = await extract(
        [uni], settings, FakeLLM(REPLY), ExtractorOptions(), "t", tmp_path / "runs" / "t"
    )
    preds = {p["field"]: p for p in json.loads((out / "predictions.json").read_text())}
    assert preds["founding_year"]["value"] == 1861
    assert preds["acceptance_rate"]["status"] == "no_snapshot"
    from uniagent.store.db import Store

    facts = {r["field"]: r for r in Store(settings.db_path).current("mit")}
    assert facts["founding_year"]["quote"].startswith("MIT was founded")


def test_snapshot_history_on_change(tmp_path):
    store = SnapshotStore(tmp_path)
    a = store.save("u", "facts", FetchResult("x", "x", 200, markdown="version one"))
    b = store.save("u", "facts", FetchResult("x", "x", 200, markdown="version two"))
    assert a.sha != b.sha and b.changed
    assert (tmp_path / "u" / "history" / f"facts-{a.sha}.md").read_text() == "version one"
    c = store.save("u", "facts", FetchResult("x", "x", 304, outcome="not_modified"))
    assert c.sha == b.sha and not c.changed


async def test_total_row_is_flagged_for_living_cost():
    md = "| Housing | $14,090 |\n| Food | $8,104 |\n| **Total** | $92,760 |"
    reply = {
        "living_cost": {
            "value": {"amount": 92760, "currency": "USD", "period": "year"},
            "quote": "| **Total** | $92,760 |",
        }
    }
    out = await Extractor(FakeLLM(reply)).extract_page(
        university="mit",
        uni_name="MIT",
        country="USA",
        currency="USD",
        page_type="living_costs",
        url="u",
        snapshot_sha="s",
        markdown=md,
        fields=[F["living_cost"]],
    )
    assert out[0].status == "flagged" and out[0].confidence == 0.4


async def test_choice_without_support_in_quote_is_rejected():
    md = "Imperial College London is a leading university for public engagement."
    reply = {"institution_type": {"value": "public", "quote": "Imperial College London"}}
    out = await Extractor(FakeLLM(reply)).extract_page(
        university="imperial",
        uni_name="Imperial",
        country="UK",
        currency="GBP",
        page_type="facts",
        url="u",
        snapshot_sha="s",
        markdown=md,
        fields=[F["institution_type"]],
    )
    assert out[0].value is None and out[0].status == "rejected_unsupported"


def test_public_must_describe_the_institution():
    from uniagent.extract.grounding import value_supported

    assert value_supported(F["institution_type"], "public", "a public research university")
    assert value_supported(F["institution_type"], "public", "among the best public universities")
    assert not value_supported(F["institution_type"], "public", "open to public engagement")


def test_reclean_uses_stored_html_and_keeps_history(tmp_path, monkeypatch):
    from uniagent.fetch import clean

    store = SnapshotStore(tmp_path)
    html = "<html><body><p>Tuition is $60,000 per year.</p></body></html>"
    old = store.save("u", "tuition", FetchResult("x", "x", 200, html=html, markdown="old text"))
    new = store.reclean("u", "tuition")
    assert new.sha != old.sha and "$60,000" in new.markdown(tmp_path)
    assert (tmp_path / "u" / "history" / f"tuition-{old.sha}.md").read_text() == "old text"
    # a cleaner regression that loses most of the page is refused
    monkeypatch.setattr(clean, "html_to_markdown", lambda h, u=None: "x")
    assert store.reclean("u", "tuition").sha == new.sha
