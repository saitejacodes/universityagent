"""Field-aware comparison of a prediction against a gold label.

Tolerances follow the research brief (docs/EVALUATION.md): published figures legitimately vary
by reporting year and definition, so exact string equality would punish correct answers.
"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from uniagent.extract.normalize import annualize
from uniagent.schema import FIELD_BY_NAME, Kind

# canonical visa labels: any alias on the right maps to the key
_VISA_ALIASES = {
    "f-1": ("f-1", "f1", "f 1", "f-1 student"),
    "j-1": ("j-1", "j1"),
    "study permit": ("study permit",),
    "uk student visa": ("student visa", "student route", "tier 4", "uk student"),
    "subclass 500": ("subclass 500", "500", "student visa (subclass 500)"),
    "student's pass": ("student's pass", "student pass", "students pass"),
    "national visa (d)": ("national visa", "type d", "d visa", "residence permit"),
    "student visa (nz)": ("fee paying student visa",),
    "stamp 2": ("stamp 2",),
    "mvv": ("mvv", "residence permit for study"),
}

_TOLERANCE = {
    "total_enrollment": 0.05,  # relative
    "intl_undergrad_tuition": 0.02,  # relative
    "living_cost": 0.10,  # relative
    "median_salary": 0.05,  # relative
    "acceptance_rate": 1.0,  # absolute percentage points
    "employment_rate": 2.0,  # absolute percentage points
}


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().casefold()


# Specific identifiers (F-1, subclass 500, ...) win over generic phrases like "student visa".
_GENERIC_ALIASES = {"student visa", "student route", "residence permit", "uk student"}
_VISA_PATTERNS = sorted(
    ((alias, label) for label, aliases in _VISA_ALIASES.items() for alias in aliases),
    key=lambda x: (x[0] in _GENERIC_ALIASES, -len(x[0])),
)


def _visa_label(s: str) -> str:
    f = _fold(s)
    for alias, label in _VISA_PATTERNS:
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", f):
            return label
    return f


def _rel_ok(pred: float, gold: float, tol: float) -> bool:
    return gold != 0 and abs(pred - gold) / abs(gold) <= tol


def values_match(field: str, pred, gold, acceptable: list | None = None) -> bool:
    """True if `pred` matches `gold` (or any of `acceptable`) under the field's rule."""
    for g in [gold, *(acceptable or [])]:
        if g is not None and pred is not None and _match_one(field, pred, g):
            return True
    return False


def _match_one(field: str, pred, gold) -> bool:
    kind = FIELD_BY_NAME[field].kind
    if kind == Kind.YEAR:
        return int(pred) == int(gold)
    if kind == Kind.INT:
        return _rel_ok(float(pred), float(gold), _TOLERANCE.get(field, 0.0))
    if kind == Kind.PERCENT:
        return abs(float(pred) - float(gold)) <= _TOLERANCE.get(field, 1.0)
    if kind == Kind.MONEY:
        if not isinstance(pred, dict) or not isinstance(gold, dict):
            return False
        if (pred.get("currency") or "").upper() != (gold.get("currency") or "").upper():
            return False
        if "min" in gold and "max" in gold:  # gold given as a range
            lo, hi = sorted((gold["min"], gold["max"]))
            return lo * 0.98 <= annualize(pred) <= hi * 1.02
        return _rel_ok(annualize(pred), annualize(gold), _TOLERANCE.get(field, 0.02))
    if kind == Kind.DATE:
        return int(pred["month"]) == int(gold["month"]) and int(pred["day"]) == int(gold["day"])
    if kind == Kind.CHOICE:
        return _fold(pred) == _fold(gold)
    if kind == Kind.LIST:
        return set_f1(pred, gold)[2] >= 0.5
    if field == "student_visa":
        return _visa_label(pred) == _visa_label(gold)
    if field == "city":
        return _fold(pred).split(",")[0] == _fold(gold).split(",")[0]
    return fuzz.token_set_ratio(_fold(pred), _fold(gold)) >= 90


def set_f1(pred: list[str] | None, gold: list[str] | None) -> tuple[float, float, float]:
    """Greedy fuzzy matching of names (token_set_ratio >= 90). Returns (P, R, F1)."""
    pred, gold = list(pred or []), list(gold or [])
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    unmatched = list(gold)
    hits = 0
    for p in pred:
        best = max(unmatched, key=lambda g: fuzz.token_set_ratio(_fold(p), _fold(g)), default=None)
        if best is not None and fuzz.token_set_ratio(_fold(p), _fold(best)) >= 90:
            hits += 1
            unmatched.remove(best)
    precision, recall = hits / len(pred), hits / len(gold)
    f1 = 0.0 if hits == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1
