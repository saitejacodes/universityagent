"""Evidence verification: is the model's answer actually supported by the page?

For every non-null answer the model must return a verbatim quote. We then check, in code:

1. quote_found    the quote occurs in the source text (whitespace/markup-insensitive, with a
                  small fuzzy tolerance for PDF-ish hyphenation and smart quotes), and
2. value_in_quote the normalised value is literally present in that quote (the number, year,
                  date, or the key words of a text answer).

An answer that fails (1) is treated as a hallucination and dropped. One that passes (1) but not
(2) is kept with low confidence (the model may have done arithmetic, e.g. summing UG + PG).
These two booleans replace v1's "confidence" field, which was the model grading itself and came
out at 1.00 for every field.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz

from uniagent.extract.normalize import parse_number
from uniagent.schema import FieldSpec, Kind

_MONTH_NAMES = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]


def canon(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )
    text = re.sub(r"[*_#|>`]+", " ", text)  # markdown decoration
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [label](url) -> label
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass
class Grounding:
    quote_found: bool
    quote_score: float
    value_in_quote: bool

    @property
    def grounded(self) -> bool:
        return self.quote_found


def quote_score(quote: str, source_canon: str) -> float:
    q = canon(quote)
    if len(q) < 4:
        return 0.0
    if q in source_canon:
        return 100.0
    # Models often trim or re-join a sentence; score the best-matching window.
    return float(fuzz.partial_ratio(q, source_canon, score_cutoff=80) or 0.0)


def value_supported(spec: FieldSpec, value, quote: str) -> bool:
    q = canon(quote)
    tokens = re.findall(r"\d[\d,.´’'  ]*\d|\d", q)
    readings: set[float] = set()
    for tok in tokens:
        tok = tok.rstrip(".,")
        for dot_thousands in (False, True):  # "19.906" may be 19.906 or 19906
            n = parse_number(tok, thousands_dot=dot_thousands)
            if n is not None:
                readings.add(round(n, 4))

    def has_number(x: float) -> bool:
        return round(float(x), 4) in readings

    k = spec.kind
    if k in (Kind.YEAR, Kind.INT, Kind.PERCENT):
        return has_number(float(value))
    if k == Kind.MONEY:
        return has_number(float(value["amount"]))
    if k == Kind.DATE:
        month = _MONTH_NAMES[value["month"] - 1]
        if re.search(rf"(?<!\d)0?{value['day']}\.0?{value['month']}\.", q):  # 15.07.
            return True
        month_ok = month in q or month[:3] in q or f"-{value['month']:02d}-" in q
        return month_ok and bool(re.search(rf"(?<!\d){value['day']}(?!\d)", q))
    if k == Kind.CHOICE:
        # "public" must describe the institution, not e.g. "public lectures"
        return bool(
            re.search(
                rf"\b{re.escape(str(value))}(ly)?\b[\w\s,()-]{{0,40}}"
                r"(universit(?:y|ies)|institutions?|colleges?|research|funded|schools?)",
                q,
            )
        )
    if k == Kind.LIST:
        return all(fuzz.partial_ratio(canon(v), q) >= 85 for v in value)
    words = [w for w in re.findall(r"[a-z0-9-]+", canon(str(value))) if len(w) > 1]
    return bool(words) and sum(w in q for w in words) / len(words) >= 0.6


def verify(spec: FieldSpec, value, quote: str | None, source_canon: str) -> Grounding:
    if value is None:
        return Grounding(False, 0.0, False)
    if spec.kind == Kind.LIST:
        # each scholarship name must appear somewhere on the page; the quote is optional
        ok = all(fuzz.partial_ratio(canon(v), source_canon, score_cutoff=85) for v in value)
        return Grounding(ok, 100.0 if ok else 0.0, ok)
    if not quote:
        return Grounding(False, 0.0, False)
    score = quote_score(quote, source_canon)
    return Grounding(score >= 90.0, score, value_supported(spec, value, quote))
