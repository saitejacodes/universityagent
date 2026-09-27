"""Build the evaluation report (Markdown) from extraction runs + gold labels."""

from __future__ import annotations

import json
from pathlib import Path

from uniagent.eval.metrics import (
    Instance,
    bootstrap_ci,
    calibration,
    paired_bootstrap,
    per_field,
    reliability_by_status,
    score_instances,
    summary,
)
from uniagent.extract.normalize import normalize
from uniagent.schema import FIELD_BY_NAME

DEV_UNIVERSITIES = {"mit", "utoronto", "imperial"}  # used for prompt development only


def load_gold(path: Path, split: str) -> list[dict]:
    """Load gold labels for a split. Instances with no snapshot for the field's pages are dropped:
    the system cannot see them, so counting a null there as a "correct abstention" would inflate
    the score for free."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if not (r.get("status") != "PRESENT" and not r.get("snapshot_sha"))]
    for r in rows:  # same canonical form as predictions
        spec = FIELD_BY_NAME[r["field"]]
        if r.get("value") is not None:
            r["value"] = normalize(spec, r["value"])
        r["acceptable_values"] = [
            v
            for v in (normalize(spec, a) for a in r.get("acceptable_values") or [])
            if v is not None
        ]
    if split == "dev":
        rows = [r for r in rows if r["university"] in DEV_UNIVERSITIES]
    elif split == "test":
        rows = [r for r in rows if r["university"] not in DEV_UNIVERSITIES]
    return rows


def load_preds(run_dir: Path) -> dict[tuple[str, str], dict]:
    rows = json.loads((run_dir / "predictions.json").read_text())
    return {(r["university"], r["field"]): r for r in rows}


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def build_report(runs: list[Path], gold_path: Path, split: str = "test") -> str:
    gold = load_gold(gold_path, split)
    all_rows = [json.loads(x) for x in gold_path.read_text().splitlines() if x.strip()]
    n_dropped = sum(
        1
        for r in all_rows
        if r.get("status") != "PRESENT"
        and not r.get("snapshot_sha")
        and (split == "all" or (r["university"] in DEV_UNIVERSITIES) == (split == "dev"))
    )
    unis = sorted({g["university"] for g in gold})
    lines = [
        "# Extraction evaluation",
        "",
        f"- Split: **{split}** — {len(unis)} universities, {len(gold)} (university, field) "
        f"instances; {sum(g['status'] == 'PRESENT' for g in gold)} with a value on the "
        f"official pages, {sum(g['status'] != 'PRESENT' for g in gold)} where the pages do not "
        "state it (the right answer is null).",
        f"- {n_dropped} instances excluded because no page of the required type could be "
        "fetched (nothing to extract from).",
        "- 95% CIs: bootstrap over universities (2,000 resamples).",
        "",
        "## Systems",
        "",
        "| Run | Model | Precision | Recall | F1 [95% CI] | Hallucination rate [95% CI] | "
        "Wrong when answering | p(not better than prev) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    scored: list[tuple[str, list[Instance], dict]] = []
    prev: list[Instance] | None = None
    for run in runs:
        cfg = json.loads((run / "config.json").read_text())
        inst = score_instances(gold, load_preds(run))
        s = summary(inst)
        lo, hi = bootstrap_ci(inst, "f1")
        hlo, hhi = bootstrap_ci(inst, "hallucination_rate")
        p = "" if prev is None else f"{paired_bootstrap(prev, inst):.3f}"
        lines.append(
            f"| `{run.name}` | `{cfg['model']}` | {_pct(s['precision'])} | {_pct(s['recall'])} "
            f"| **{_pct(s['f1'])}** [{_pct(lo)}–{_pct(hi)}] "
            f"| {_pct(s['hallucination_rate'])} [{_pct(hlo)}–{_pct(hhi)}] "
            f"| {_pct(s['error_rate_when_answering'])} | {p} |"
        )
        scored.append((run.name, inst, cfg))
        prev = inst

    # Headline comparisons: the full system (last row) against every earlier row, paired by
    # university. Consecutive steps can be individually small; this is the end-to-end claim.
    if len(scored) > 1:
        last_name, last_inst, _ = scored[-1]
        lines += [
            "",
            f"**`{last_name}` vs earlier systems** (paired bootstrap over universities, "
            "p = probability it is *not* better on null-aware accuracy):",
            "",
        ]
        for nm, inst, _ in scored[:-1]:
            lines.append(f"- vs `{nm}`: p = {paired_bootstrap(inst, last_inst):.3f}")

    name, best, cfg = scored[-1]
    lines += [
        "",
        f"## Per field (`{name}`)",
        "",
        "| Field | TP | Wrong value | Missed | Hallucinated | Correct null | F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for field, s in sorted(per_field(best).items()):
        lines.append(
            f"| {field} | {s['TP']} | {s['WV']} | {s['MISS']} | {s['HAL']} | {s['CA']} "
            f"| {_pct(s['f1'])} |"
        )

    # What a consumer sees with GET /universities/{id}?min_confidence=0.5: low-confidence
    # answers are withheld (treated as abstentions).
    thr = 0.5
    gated = score_instances(
        gold,
        {
            k: (v if (v.get("confidence") or 0) >= thr else {**v, "value": None})
            for k, v in load_preds(runs[-1]).items()
        },
    )
    sg = summary(gated)
    lines += [
        "",
        f"## Serving with a confidence threshold of {thr} (`{name}`)",
        "",
        "| Precision | Recall | F1 | Hallucination rate | Wrong when answering |",
        "|---|---|---|---|---|",
        f"| {_pct(sg['precision'])} | {_pct(sg['recall'])} | {_pct(sg['f1'])} | "
        f"{_pct(sg['hallucination_rate'])} | {_pct(sg['error_rate_when_answering'])} |",
    ]

    cal = calibration(best)
    if cal["n"]:
        auroc = "–" if cal["auroc"] is None else f"{cal['auroc']:.3f}"
        cal_line = (
            f"- ECE {cal['ece']:.3f} · Brier {cal['brier']:.3f} · AUROC {auroc} "
            f"(over {cal['n']} answered instances)"
        )
    else:
        cal_line = "- no answers"
    lines += [
        "",
        f"## Is the confidence score honest? (`{name}`)",
        "",
        cal_line,
        "",
        "| Evidence check result | Answers | Accuracy |",
        "|---|---|---|",
    ]
    for status, (n, acc) in sorted(reliability_by_status(best).items(), key=lambda x: -x[1][1]):
        lines.append(f"| {status} | {n} | {_pct(acc)} |")

    errors = [i for i in best if i.outcome in ("WV", "HAL")]
    if errors:
        preds = load_preds(runs[-1])
        gmap = {(g["university"], g["field"]): g for g in gold}
        lines += [
            "",
            "## Every wrong answer (for error analysis)",
            "",
            "| University | Field | Predicted | Gold | Outcome |",
            "|---|---|---|---|---|",
        ]
        for i in sorted(errors, key=lambda i: (i.field, i.university)):
            pv = preds.get((i.university, i.field), {}).get("value")
            gv = gmap[(i.university, i.field)].get("value")
            lines.append(
                f"| {i.university} | {i.field} | `{json.dumps(pv)[:60]}` | "
                f"`{json.dumps(gv)[:60]}` | {i.outcome} |"
            )
    return "\n".join(lines) + "\n"
