"""Turn whatever the model wrote into canonical typed values.

Canonical forms (what gets stored and compared):
    YEAR / INT   int
    PERCENT      float in [0, 100]
    MONEY        {"amount": float, "currency": "USD", "period": "year|month|week|term|total"}
    DATE         {"month": 1-12, "day": 1-31, "year": int | None}
    TEXT         str
    CHOICE       one of the field's choices
    LIST         list[str]
Anything that can't be parsed becomes None (and is counted as an abstention, not a guess).
"""

from __future__ import annotations

import re
from datetime import date

from uniagent.schema import FieldSpec, Kind

_NUM = re.compile(r"-?\d[\d,  ]*(?:\.\d+)?")
_MONTHS = {
    m: i
    for i, m in enumerate(
        [
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
        ],
        start=1,
    )
}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})
_MONTHS["sept"] = 9

# Longest symbols first so "A$" wins over "$".
_CURRENCY_SYMBOLS = [
    ("NZ$", "NZD"),
    ("AU$", "AUD"),
    ("A$", "AUD"),
    ("CA$", "CAD"),
    ("C$", "CAD"),
    ("S$", "SGD"),
    ("HK$", "HKD"),
    ("US$", "USD"),
    ("£", "GBP"),
    ("€", "EUR"),
    ("₹", "INR"),
    ("¥", "JPY"),
    ("Rs.", "INR"),
    ("Rs", "INR"),
]
_CURRENCY_CODES = {
    "USD",
    "GBP",
    "EUR",
    "AUD",
    "CAD",
    "SGD",
    "INR",
    "CHF",
    "NZD",
    "HKD",
    "JPY",
    "CNY",
    "SEK",
    "DKK",
    "NOK",
}
_PERIODS = [
    ("month", ("per month", "/month", "a month", "monthly", "/mo", "per mo")),
    ("week", ("per week", "/week", "a week", "weekly", "/wk")),
    ("term", ("per semester", "per term", "semester", "per trimester")),
    (
        "year",
        (
            "per year",
            "/year",
            "a year",
            "annual",
            "annually",
            "per annum",
            "p.a.",
            "academic year",
            "yearly",
            "/yr",
        ),
    ),
]


def parse_number(text: str | float | int | None, thousands_dot: bool = False) -> float | None:
    """Parse the first number in `text`. Handles "11,816", "45k", Swiss "20´000" and, when
    `thousands_dot` is set (money / counts), European "19.906" and "1.234.567,50"."""
    if text is None or isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip().lower().replace("\u2009", "").replace("\xa0", " ").replace("\u202f", "")
    s = re.sub(r"(?<=\d)[´’'](?=\d{3})", "", s)  # Swiss/other apostrophe thousands separators
    euro = re.search(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?(?![\d.])", s)
    if euro and (thousands_dot or euro.group(0).count(".") > 1):
        raw = euro.group(0).replace(".", "").replace(",", ".")
        m_end = euro.end()
    else:
        m = _NUM.search(s)
        if not m:
            return None
        raw = m.group(0).replace(",", "").replace(" ", "")
        m_end = m.end()
    try:
        val = float(raw)
    except ValueError:
        return None
    tail = s[m_end : m_end + 10].strip()
    if tail.startswith(("k", "thousand")):
        val *= 1_000
    elif tail.startswith(("m ", "million")) or tail == "m":
        val *= 1_000_000
    return val


def detect_currency(text: str, default: str | None = None) -> str | None:
    up = text.upper()
    for code in _CURRENCY_CODES:
        if re.search(rf"\b{code}\b", up):
            return code
    for sym, code in _CURRENCY_SYMBOLS:
        if sym.upper() in up:
            return code
    if "$" in text:
        return default if default and default.endswith("D") else "USD"
    return default


def detect_period(text: str, default: str = "year") -> str:
    low = text.lower()
    for period, cues in _PERIODS:
        if any(c in low for c in cues):
            return period
    return default


def parse_money(value, default_currency: str | None) -> dict | None:
    if isinstance(value, dict):
        amount = parse_number(value.get("amount"), thousands_dot=True)
        text = " ".join(str(v) for v in value.values() if v is not None)
        currency = str(value.get("currency") or "").upper() or None
        if currency not in _CURRENCY_CODES:
            currency = detect_currency(text, default_currency)
        period = str(value.get("period") or "").lower() or detect_period(text)
        if period not in {"year", "month", "week", "term", "total"}:
            period = detect_period(period)
    else:
        text = str(value)
        amount = parse_number(text, thousands_dot=True)
        currency = detect_currency(text, default_currency)
        period = detect_period(text)
    if amount is None or amount <= 0:
        return None
    return {"amount": round(amount, 2), "currency": currency, "period": period}


def annualize(money: dict) -> float:
    factor = {"year": 1, "month": 12, "week": 52, "term": 2, "total": 1}.get(money["period"], 1)
    return money["amount"] * factor


def parse_date(value) -> dict | None:
    if isinstance(value, dict):
        try:
            return {
                "month": int(value["month"]),
                "day": int(value["day"]),
                "year": int(value["year"]) if value.get("year") else None,
            }
        except (KeyError, TypeError, ValueError):
            value = " ".join(str(v) for v in value.values())
    s = str(value).strip().lower()
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        return _valid_date(y, mo, d)
    m = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})?", s)  # European 15.07. / 15.07.2026
    if m:
        d, mo = int(m.group(1)), int(m.group(2))
        return _valid_date(int(m.group(3)) if m.group(3) else None, mo, d)
    year_m = re.search(r"\b(20\d{2})\b", s)
    year = int(year_m.group(1)) if year_m else None
    s_wo_year = re.sub(r"\b20\d{2}\b", " ", s)
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]{3,9})\.?", s_wo_year)
    if m and m.group(2) in _MONTHS:
        return _valid_date(year, _MONTHS[m.group(2)], int(m.group(1)))
    m = re.search(r"\b([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", s_wo_year)
    if m and m.group(1) in _MONTHS:
        return _valid_date(year, _MONTHS[m.group(1)], int(m.group(2)))
    return None


def _valid_date(year: int | None, month: int, day: int) -> dict | None:
    try:
        date(year or 2000, month, day)
    except ValueError:
        return None
    return {"month": month, "day": day, "year": year}


def normalize(spec: FieldSpec, value, default_currency: str | None = None):
    if value is None or (
        isinstance(value, str)
        and value.strip().lower()
        in {"", "null", "none", "n/a", "not found", "unknown", "not available"}
    ):
        return None
    k = spec.kind
    if k == Kind.YEAR:
        n = parse_number(value)
        return int(n) if n and 1000 <= n <= date.today().year else None
    if k == Kind.INT:
        n = parse_number(value, thousands_dot=True)
        return int(round(n)) if n and n > 0 else None
    if k == Kind.PERCENT:
        n = parse_number(value)
        if n is None:
            return None
        if 0 < n < 1 and "%" not in str(value):
            n *= 100  # model returned a fraction
        return round(n, 2) if 0 <= n <= 100 else None
    if k == Kind.MONEY:
        return parse_money(value, default_currency)
    if k == Kind.DATE:
        return parse_date(value)
    if k == Kind.CHOICE:
        low = str(value).strip().lower()
        return next((c for c in spec.choices if c in low), None)
    if k == Kind.LIST:
        items = value if isinstance(value, list) else re.split(r"[;\n]|,\s(?=[A-Z])", str(value))
        out = []
        for it in items:
            name = (it.get("name") if isinstance(it, dict) else str(it)).strip(" -•*\t")
            if name and name.lower() not in {x.lower() for x in out}:
                out.append(name)
        return out[:5] or None
    text = str(value).strip()
    return text or None
