"""Helpers for building and auditing the gold label set.

    python scripts/gold_tool.py fields                 # print the field definitions labellers use
    python scripts/gold_tool.py pages <university>     # list snapshot files for a university
    python scripts/gold_tool.py check <labels.jsonl>   # audit labels against the snapshots
    python scripts/gold_tool.py merge <out> <in...>    # merge label files (later files win)
    python scripts/gold_tool.py sheet <labels.jsonl>   # write a human spot-check sheet (Markdown)
    python scripts/gold_tool.py agreement <a> <b>      # inter-annotator agreement on overlap
    python scripts/gold_tool.py refresh <labels.jsonl>  # re-point labels after snapshots change

Label record (one JSON object per line, one per university x field):
    {"university": "mit", "field": "founding_year",
     "status": "PRESENT" | "NOT_ON_PAGES",
     "value": 1861,                      # canonical form, see uniagent.extract.normalize
     "acceptable_values": [],            # other defensible answers (e.g. charter vs opening year)
     "evidence_quote": "1861: Founded in Boston",   # verbatim from the snapshot (PRESENT only)
     "page_type": "facts", "snapshot_sha": "...", "annotator": "...", "notes": ""}

`check` enforces what makes the labels trustworthy: every PRESENT label's quote must occur in
the snapshot it cites, and the value must be literally supported by that quote (or be marked in
notes as derived). It is run in CI-style before any evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from uniagent.config import get_settings
from uniagent.extract.grounding import canon, quote_score, value_supported
from uniagent.extract.normalize import normalize
from uniagent.fetch.snapshots import SnapshotStore
from uniagent.schema import FIELD_BY_NAME, FIELDS


def cmd_fields() -> None:
    for f in FIELDS:
        pages = ", ".join(p.value for p in f.pages)
        print(f"- {f.name} [{f.kind.value}; pages: {pages}]\n    {f.instruction}")


def cmd_pages(uni: str) -> None:
    store = SnapshotStore(get_settings().snapshots_dir)
    for page, snap in store.all_for(uni).items():
        md = get_settings().snapshots_dir / uni / f"{page}.md"
        print(f"{page:18} sha={snap.sha} chars={md.stat().st_size:>7} {md}  <- {snap.url}")


def _load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def cmd_check(path: str) -> int:
    settings = get_settings()
    store = SnapshotStore(settings.snapshots_dir)
    problems = 0
    seen = set()
    for i, lab in enumerate(_load(path), 1):
        key = (lab.get("university"), lab.get("field"))
        where = f"line {i} {key}"
        if key in seen:
            print(f"DUPLICATE {where}")
            problems += 1
        seen.add(key)
        spec = FIELD_BY_NAME.get(lab.get("field", ""))
        if spec is None:
            print(f"UNKNOWN FIELD {where}")
            problems += 1
            continue
        if lab.get("status") not in ("PRESENT", "NOT_ON_PAGES"):
            print(f"BAD STATUS {where}: {lab.get('status')}")
            problems += 1
            continue
        if lab["status"] == "NOT_ON_PAGES":
            if lab.get("value") is not None:
                print(f"NULL-STATUS WITH VALUE {where}")
                problems += 1
            continue
        value = normalize(spec, lab.get("value"))
        if value is None:
            print(f"UNPARSEABLE VALUE {where}: {lab.get('value')!r}")
            problems += 1
            continue
        snap = store.get(lab["university"], lab.get("page_type", ""))
        if snap is None:
            print(f"NO SNAPSHOT {where}: page_type={lab.get('page_type')}")
            problems += 1
            continue
        if lab.get("snapshot_sha") and lab["snapshot_sha"] != snap.sha:
            print(f"STALE SNAPSHOT {where}: label {lab['snapshot_sha']} vs current {snap.sha}")
            problems += 1
        source = canon(snap.markdown(settings.snapshots_dir))
        quote = lab.get("evidence_quote") or ""
        if spec.kind.value == "list":
            continue
        if quote_score(quote, source) < 90:
            print(f"QUOTE NOT IN SNAPSHOT {where}: {quote[:80]!r}")
            problems += 1
        elif not value_supported(spec, value, quote) and "derived" not in (lab.get("notes") or ""):
            print(f"VALUE NOT IN QUOTE {where}: {value!r} / {quote[:80]!r}")
            problems += 1
    n = len(seen)
    print(f"\n{n} labels checked, {problems} problem(s)")
    return 1 if problems else 0


def cmd_refresh(path: str) -> None:
    """After snapshots change (re-clean / re-crawl): re-point labels to the current snapshot when
    their evidence is still there, and list the labels a human/annotator must re-check:
    PRESENT labels whose quote vanished, and NOT_ON_PAGES labels on pages whose text changed
    (the fact may now be visible)."""
    settings = get_settings()
    store = SnapshotStore(settings.snapshots_dir)
    rows = _load(path)
    updated, recheck = 0, []
    for lab in rows:
        page = lab.get("page_type")
        if not page or not lab.get("snapshot_sha"):
            if lab["status"] != "PRESENT":
                # "no snapshot" labels: a page may exist now
                spec = FIELD_BY_NAME[lab["field"]]
                for pt in spec.pages:
                    if store.get(lab["university"], pt.value):
                        recheck.append((lab, f"snapshot now exists for {pt.value}"))
                        break
            continue
        snap = store.get(lab["university"], page)
        if snap is None or snap.sha == lab["snapshot_sha"]:
            continue
        source = canon(snap.markdown(settings.snapshots_dir))
        if lab["status"] == "PRESENT":
            quote = lab.get("evidence_quote") or ""
            if spec_is_list(lab) or quote_score(quote, source) >= 90:
                lab["snapshot_sha"] = snap.sha
                updated += 1
            else:
                recheck.append((lab, "quote no longer found"))
        else:
            recheck.append((lab, "page text changed"))
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{updated} label(s) re-pointed to current snapshots; {len(recheck)} need re-check:")
    for lab, why in recheck:
        print(f"  RECHECK {lab['university']}/{lab['field']} ({lab['status']}): {why}")


def spec_is_list(lab: dict) -> bool:
    return FIELD_BY_NAME[lab["field"]].kind.value == "list"


def cmd_merge(out: str, *inputs: str) -> None:
    merged: dict[tuple[str, str], dict] = {}
    for path in inputs:
        for lab in _load(path):
            merged[(lab["university"], lab["field"])] = lab
    rows = sorted(merged.values(), key=lambda r: (r["university"], r["field"]))
    Path(out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} labels to {out}")


def cmd_sheet(path: str) -> None:
    labels = _load(path)
    lines = [
        "# Gold label spot-check sheet",
        "",
        "Tick a row after checking the value against the live page (or the snapshot).",
        "",
        "| ✓ | University | Field | Status | Value | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    for lab in labels:
        v = json.dumps(lab.get("value"), ensure_ascii=False)
        q = (lab.get("evidence_quote") or lab.get("notes") or "").replace("|", "/")[:120]
        lines.append(
            f"| [ ] | {lab['university']} | {lab['field']} | {lab['status']} | `{v}` | {q} |"
        )
    out = Path(path).with_name("SPOT_CHECK.md")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def cmd_agreement(a_path: str, b_path: str) -> None:
    """Inter-annotator agreement on the overlapping (university, field) instances: raw agreement
    and Cohen's kappa on PRESENT vs NOT_ON_PAGES, and value agreement (same matching rules as the
    evaluation) where both annotators found a value."""
    from uniagent.eval.match import values_match

    a = {(r["university"], r["field"]): r for r in _load(a_path)}
    b = {(r["university"], r["field"]): r for r in _load(b_path)}
    keys = sorted(set(a) & set(b))
    if not keys:
        print("no overlap")
        return
    sa = [a[k]["status"] == "PRESENT" for k in keys]
    sb = [b[k]["status"] == "PRESENT" for k in keys]
    n = len(keys)
    po = sum(x == y for x, y in zip(sa, sb, strict=True)) / n
    pa, pb = sum(sa) / n, sum(sb) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    both = [k for k, x, y in zip(keys, sa, sb, strict=True) if x and y]
    agree_vals = []
    for k in both:
        spec = FIELD_BY_NAME[k[1]]
        va, vb = normalize(spec, a[k]["value"]), normalize(spec, b[k]["value"])
        acc_a = [normalize(spec, v) for v in a[k].get("acceptable_values") or []]
        acc_b = [normalize(spec, v) for v in b[k].get("acceptable_values") or []]
        agree_vals.append(values_match(k[1], va, vb, acc_b) or values_match(k[1], vb, va, acc_a))
    print(f"instances: {n}")
    print(f"status agreement: {po:.1%}  Cohen's kappa: {kappa:.2f}")
    if both:
        print(
            f"value agreement where both PRESENT: {sum(agree_vals)}/{len(both)} "
            f"({sum(agree_vals) / len(both):.1%})"
        )
    for k, x, y in zip(keys, sa, sb, strict=True):
        if x != y:
            print(f"  status differs {k}: {a[k]['status']} vs {b[k]['status']}")
    for k, ok in zip(both, agree_vals, strict=True):
        if not ok:
            print(f"  value differs {k}: {a[k]['value']!r} vs {b[k]['value']!r}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["help"]
    if cmd == "fields":
        cmd_fields()
    elif cmd == "pages":
        cmd_pages(args[0])
    elif cmd == "check":
        sys.exit(cmd_check(args[0]))
    elif cmd == "merge":
        cmd_merge(args[0], *args[1:])
    elif cmd == "refresh":
        cmd_refresh(args[0])
    elif cmd == "agreement":
        cmd_agreement(args[0], args[1])
    elif cmd == "sheet":
        cmd_sheet(args[0])
    else:
        sys.exit(__doc__)
