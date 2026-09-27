"""Page -> typed, evidence-backed field values.

One LLM call per (university, page): every field whose home is that page type is requested
together, over the retrieved context. The model must return, per field, a value *and* the
verbatim sentence/table row it came from; values without verifiable evidence are dropped.

Confidence is computed from checks we run ourselves, not asked of the model:

    grounded quote + value literally in quote + primary page     0.90
    grounded quote + value literally in quote + fallback page    0.80
    grounded quote, value not literally in quote (derived)       0.50
    quote not found in page                                      rejected (or 0.20 without
                                                                 grounding, for the ablation)
The eval measures how well these tiers are calibrated.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from uniagent.extract.grounding import canon, verify
from uniagent.extract.llm import LLMClient
from uniagent.extract.normalize import normalize
from uniagent.extract.retrieve import select_context, truncate_context
from uniagent.schema import FieldSpec, Kind

SYSTEM = (
    "You extract facts from official university web pages for a database. Use ONLY the page "
    "text you are given — never background knowledge. Copy evidence verbatim. If the page does "
    "not state a fact, return null for it. Answer with a single JSON object."
)

_FORMATS = {
    Kind.YEAR: "integer year, e.g. 1861",
    Kind.INT: "integer, e.g. 11816",
    Kind.PERCENT: "number in percent, e.g. 4.6",
    Kind.MONEY: '{"amount": 59750, "currency": "USD", "period": "year|month|week|term"}',
    Kind.DATE: 'the date as written, e.g. "January 15" or "2027-01-15"',
    Kind.TEXT: "short string",
    Kind.CHOICE: "one of: {choices}",
    Kind.LIST: 'list of names, e.g. ["Name A", "Name B"]',
}


# Only numeric fields can legitimately differ from the literal quote (sums, computed rates).
_DERIVABLE = {Kind.INT, Kind.PERCENT, Kind.MONEY}

# Evidence that is real but points at the wrong thing for the field: the answer is kept with
# low confidence ("flagged") so a correct answer from another page can win the merge.
_SUSPICIOUS = {
    "living_cost": (re.compile(r"\btotal\b|tuition", re.I), None),
    "intl_undergrad_tuition": (
        re.compile(r"scholarship|bursary|award|\bhome\b|domestic|in-state|\bresident", re.I),
        re.compile(r"international|overseas|non-resident|out-of-state|all students", re.I),
    ),
    "application_deadline": (re.compile(r"\bearly\b|earlier|priority|scholarship", re.I), None),
}


def _suspicious(field: str, quote: str) -> bool:
    rule = _SUSPICIOUS.get(field)
    if rule is None:
        return False
    bad, unless = rule
    return bool(bad.search(quote)) and not (unless and unless.search(quote))


@dataclass
class Extraction:
    university: str
    field: str
    value: object
    raw_value: object
    quote: str | None
    note: str | None
    page_type: str
    url: str
    snapshot_sha: str
    status: str  # accepted | derived | rejected_ungrounded | not_found | llm_error
    grounded: bool
    value_in_quote: bool
    quote_score: float
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


def build_messages(
    uni_name: str,
    country: str,
    currency: str,
    page_type: str,
    url: str,
    fields: list[FieldSpec],
    context: str,
) -> list[dict]:
    spec_lines = []
    for f in fields:
        fmt = _FORMATS[f.kind].replace("{choices}", ", ".join(f.choices))
        spec_lines.append(f'- "{f.name}": {f.instruction} Value format: {fmt}.')
    example = {
        f.name: {"value": "...", "quote": "exact text copied from the page", "note": None}
        for f in fields[:1]
    }
    user = (
        f"University: {uni_name} ({country}). Local currency: {currency}.\n"
        f"Page type: {page_type}. URL: {url}\n\n"
        f"Fields to extract:\n" + "\n".join(spec_lines) + "\n\n"
        "Page text:\n<<<\n" + context + "\n>>>\n\n"
        "Return JSON with one key per field, each an object with:\n"
        '  "value": the value in the format above, or null if the page does not state it,\n'
        '  "quote": the shortest exact span (sentence or table row, max 300 chars) copied from '
        "the page text that contains the value, or null,\n"
        '  "note": optional caveat (e.g. which programme a fee is for), or null.\n'
        f"Example shape: {json.dumps(example)}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _parse_json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if isinstance(data, dict) and "fields" in data and isinstance(data["fields"], dict):
        data = data["fields"]
    return data if isinstance(data, dict) else None


@dataclass
class ExtractorOptions:
    retrieval: bool = True  # False = v1-style first-4000-chars truncation
    grounding: bool = True  # False = keep values even if the quote isn't on the page
    context_budget: int = 7000
    max_tokens: int = 1500


class Extractor:
    def __init__(self, llm: LLMClient, options: ExtractorOptions | None = None):
        self.llm = llm
        self.opt = options or ExtractorOptions()

    async def extract_page(
        self,
        *,
        university: str,
        uni_name: str,
        country: str,
        currency: str,
        page_type: str,
        url: str,
        snapshot_sha: str,
        markdown: str,
        fields: list[FieldSpec],
    ) -> list[Extraction]:
        if self.opt.retrieval:
            context = select_context(markdown, fields, budget=self.opt.context_budget)
        else:
            context = truncate_context(markdown, 4000)
        messages = build_messages(uni_name, country, currency, page_type, url, fields, context)

        data = None
        for attempt in range(2):
            try:
                reply = await self.llm.chat(
                    messages,
                    temperature=0.0,
                    seed=attempt,
                    max_tokens=self.opt.max_tokens,
                    json_mode=True,
                )
            except Exception as exc:  # LLMError / transport: record and move on
                return [
                    self._empty(university, f, page_type, url, snapshot_sha, "llm_error", str(exc))
                    for f in fields
                ]
            data = _parse_json(reply.text)
            if data is not None:
                break
        if data is None:
            return [
                self._empty(
                    university, f, page_type, url, snapshot_sha, "llm_error", "unparseable JSON"
                )
                for f in fields
            ]

        # Grounding is checked against the full page, not just the retrieved context.
        source = canon(markdown)
        out = []
        for f in fields:
            item = data.get(f.name)
            if not isinstance(item, dict):
                item = {"value": item, "quote": None}
            raw, quote, note = item.get("value"), item.get("quote"), item.get("note")
            value = normalize(f, raw, currency)
            if f.kind == Kind.LIST and value:
                value = [v for v in value if verify(f, [v], None, source).grounded] or None
            if value is None:
                out.append(
                    self._empty(
                        university, f, page_type, url, snapshot_sha, "not_found", note, raw, quote
                    )
                )
                continue
            g = verify(f, value, quote if isinstance(quote, str) else None, source)
            q = quote if isinstance(quote, str) else ""
            if g.grounded and g.value_in_quote:
                status = "accepted"
                conf = 0.9 if f.pages[0].value == page_type else 0.8
                if _suspicious(f.name, q):
                    status, conf = "flagged", 0.4
            elif g.grounded and f.kind in _DERIVABLE:
                status, conf = "derived", 0.5  # e.g. a rate computed from counts
            elif g.grounded:
                status, conf = "rejected_unsupported", 0.2  # quote real, but doesn't say it
            else:
                status, conf = "rejected_ungrounded", 0.2
            keep = not status.startswith("rejected") or not self.opt.grounding
            out.append(
                Extraction(
                    university,
                    f.name,
                    value if keep else None,
                    raw,
                    quote if isinstance(quote, str) else None,
                    note if isinstance(note, str) else None,
                    page_type,
                    url,
                    snapshot_sha,
                    status,
                    g.grounded,
                    g.value_in_quote,
                    round(g.quote_score, 1),
                    conf if keep else 0.0,
                )
            )
        return out

    @staticmethod
    def _empty(university, f, page_type, url, sha, status, note=None, raw=None, quote=None):
        return Extraction(
            university,
            f.name,
            None,
            raw,
            quote if isinstance(quote, str) else None,
            note if isinstance(note, str) else None,
            page_type,
            url,
            sha,
            status,
            False,
            False,
            0.0,
            0.0,
        )


def merge_field(candidates: list[Extraction]) -> Extraction | None:
    """Pick the final answer for one field from its per-page extractions.

    Highest confidence wins; when two pages agree on the value the answer gets a small bonus
    (independent sources agreeing), and a disagreement is recorded in the note.
    """
    found = [c for c in candidates if c.value is not None]
    if not found:
        return candidates[0] if candidates else None
    found.sort(key=lambda c: -c.confidence)
    best = found[0]
    others = [c for c in found[1:] if c.confidence >= 0.5]
    if others:
        agree = [c for c in others if _same(best.value, c.value)]
        if agree:
            best.confidence = min(0.97, best.confidence + 0.05)
            best.note = ((best.note or "") + f" [agrees with {agree[0].page_type} page]").strip()
        else:
            best.note = (
                (best.note or "") + f" [conflict: {others[0].page_type} page says "
                f"{others[0].value!r}]"
            ).strip()
    return best


def _same(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict) and "amount" in a and "amount" in b:
        return a.get("currency") == b.get("currency") and abs(a["amount"] - b["amount"]) <= (
            0.02 * max(a["amount"], b["amount"])
        )
    return a == b
