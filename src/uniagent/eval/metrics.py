"""Null-aware extraction metrics, calibration, and university-level bootstrap intervals.

One instance = (university, field). Outcomes:
    TP   gold present, prediction matches          CA   gold null, prediction null
    WV   gold present, prediction wrong value      HAL  gold null, prediction non-null
    MISS gold present, prediction null
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from uniagent.eval.match import values_match


@dataclass
class Instance:
    university: str
    field: str
    gold_present: bool
    pred_present: bool
    correct: bool  # TP or CA
    outcome: str  # TP | WV | MISS | HAL | CA
    confidence: float
    status: str  # extractor status (accepted/derived/...)


def score_instances(gold: list[dict], preds: dict[tuple[str, str], dict]) -> list[Instance]:
    out = []
    for g in gold:
        key = (g["university"], g["field"])
        p = preds.get(key) or {}
        pv = p.get("value")
        gp = g.get("status") == "PRESENT" and g.get("value") is not None
        pp = pv is not None
        if gp and pp:
            ok = values_match(g["field"], pv, g["value"], g.get("acceptable_values"))
            outcome = "TP" if ok else "WV"
        elif gp:
            outcome, ok = "MISS", False
        elif pp:
            outcome, ok = "HAL", False
        else:
            outcome, ok = "CA", True
        out.append(
            Instance(
                g["university"],
                g["field"],
                gp,
                pp,
                ok,
                outcome,
                float(p.get("confidence") or 0.0),
                p.get("status", "missing"),
            )
        )
    return out


def summary(instances: list[Instance]) -> dict:
    c = Counter(i.outcome for i in instances)
    tp, wv, miss, hal, ca = (c[k] for k in ("TP", "WV", "MISS", "HAL", "CA"))
    n_pred = tp + wv + hal
    n_gold = tp + wv + miss
    n_null = hal + ca
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(instances),
        "TP": tp,
        "WV": wv,
        "MISS": miss,
        "HAL": hal,
        "CA": ca,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + ca) / len(instances) if instances else 0.0,
        "hallucination_rate": hal / n_null if n_null else 0.0,
        "correct_abstention_rate": ca / n_null if n_null else 0.0,
        "wrong_value_rate": wv / n_gold if n_gold else 0.0,
        "miss_rate": miss / n_gold if n_gold else 0.0,
        # share of answered instances that are wrong in any way: what a user would experience
        "error_rate_when_answering": (wv + hal) / n_pred if n_pred else 0.0,
    }


def per_field(instances: list[Instance]) -> dict[str, dict]:
    by: dict[str, list[Instance]] = defaultdict(list)
    for i in instances:
        by[i.field].append(i)
    return {f: summary(v) for f, v in by.items()}


def calibration(instances: list[Instance], bins: int = 10) -> dict:
    """ECE (equal-mass bins), Brier score and AUROC over answered instances."""
    answered = sorted((i for i in instances if i.pred_present), key=lambda i: i.confidence)
    if not answered:
        return {"ece": None, "brier": None, "auroc": None, "n": 0}
    n = len(answered)
    ece = 0.0
    size = max(1, n // bins)
    for start in range(0, n, size):
        chunk = answered[start : start + size]
        acc = sum(i.correct for i in chunk) / len(chunk)
        conf = sum(i.confidence for i in chunk) / len(chunk)
        ece += len(chunk) / n * abs(acc - conf)
    brier = sum((i.confidence - i.correct) ** 2 for i in answered) / n
    pos = [i.confidence for i in answered if i.correct]
    neg = [i.confidence for i in answered if not i.correct]
    auroc = None
    if pos and neg:
        wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
        auroc = wins / (len(pos) * len(neg))
    return {"ece": ece, "brier": brier, "auroc": auroc, "n": n}


def reliability_by_status(instances: list[Instance]) -> dict[str, tuple[int, float]]:
    """Accuracy of answered instances per extractor status (accepted/derived/...)."""
    by: dict[str, list[bool]] = defaultdict(list)
    for i in instances:
        if i.pred_present:
            by[i.status].append(i.correct)
    return {s: (len(v), sum(v) / len(v)) for s, v in by.items()}


def bootstrap_ci(
    instances: list[Instance], metric: str, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% CI resampling *universities* (fields of one university are not independent)."""
    by_uni: dict[str, list[Instance]] = defaultdict(list)
    for i in instances:
        by_uni[i.university].append(i)
    unis = list(by_uni)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [inst for u in (rng.choice(unis) for _ in unis) for inst in by_uni[u]]
        vals.append(summary(sample)[metric])
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]


def paired_bootstrap(
    a: list[Instance], b: list[Instance], n_boot: int = 2000, seed: int = 0
) -> float:
    """P(system b is not better than a on null-aware accuracy), resampling universities."""
    key = lambda i: (i.university, i.field)  # noqa: E731
    bmap = {key(i): i for i in b}
    pairs = [(i, bmap[key(i)]) for i in a if key(i) in bmap]
    by_uni: dict[str, list[tuple[Instance, Instance]]] = defaultdict(list)
    for x, y in pairs:
        by_uni[x.university].append((x, y))
    unis = list(by_uni)
    rng = random.Random(seed)
    not_better = 0
    for _ in range(n_boot):
        diff = 0
        for u in (rng.choice(unis) for _ in unis):
            diff += sum(y.correct - x.correct for x, y in by_uni[u])
        not_better += diff <= 0
    return not_better / n_boot
