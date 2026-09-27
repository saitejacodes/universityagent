# Evaluation methodology

## What v1 claimed, and why it doesn't hold

v1's `eval_report.md` showed **confidence 1.00 on all 30 fields** and a "ground-truth" table in
which every extracted value matched. Three problems:

1. The confidence was the model grading itself; LLM self-reported confidence is known to be
   uninformative, and a constant 1.00 carries no signal at all.
2. The "ground truth" was not independent of the system, so it could not detect errors.
3. Fields were free-text blobs ("tuition_fees_all_levels"), which can't be scored automatically.

v2 replaces all three: typed fields, independent evidence-backed gold labels, and confidence
derived from checks the code performs.

## Gold labels

- **Universities.** Official pages for 32 universities were searched and every URL was verified
  with the pipeline's own fetcher (`scripts/probe_url.py`). **26 universities in 13 countries/
  regions** are usable (USA, Canada, UK, Ireland, Germany, Switzerland, Netherlands, Belgium,
  Australia, New Zealand, Singapore, Hong Kong, India). Six serve bot challenges on their main
  sites (Michigan, Oxford, UCL, Melbourne, NUS, Monash); they are listed under `excluded` in
  `config/universities.yaml` — we do not evade bot protection.
- **Size.** 312 labels (26 × 12 fields): 198 `PRESENT`, 114 `NOT_ON_PAGES`, of which 59 are
  "no page of that type could be fetched" and are excluded from scoring.
- **Frozen snapshots.** Labels are written against the stored snapshot (hash recorded in each
  label), not the live site, so the benchmark doesn't drift when a page changes.
- **Label schema.** One label per (university, field): `status` ∈ {`PRESENT`, `NOT_ON_PAGES`},
  canonical `value`, `acceptable_values` for legitimately ambiguous facts (e.g. charter vs. opening
  year), a **verbatim evidence quote**, page type and snapshot hash.
- **Independence.** Labels are produced by a separate annotation pass (a different, larger model
  working only from the snapshot text and the written field definitions), never from system
  output. Every label is machine-audited (`scripts/gold_tool.py check`): the quote must occur in
  the cited snapshot and the value must appear in the quote. A random sample of 20 labels
  (15 present, 5 absent) was re-checked against the **live** websites on 2026-09-26:
  **20/20 consistent** (`data/gold/SPOT_CHECK_20.md`).
- **Adjudication.** Three annotation passes (`labels_part{1,2,3}.jsonl`) are kept as written;
  disagreements in labelling *policy* found on review are resolved by explicit rules recorded in
  `data/gold/adjudications.jsonl` (e.g. "a city named in the university's official name counts as
  stated"), which override the raw labels when `labels.jsonl` is built
  (`scripts/gold_tool.py merge`). A blind second annotation of three universities
  (`labels_double.jsonl`) measures inter-annotator agreement: **97.2% agreement on
  PRESENT vs NOT_ON_PAGES (Cohen's κ = 0.93, 36 instances) and 100% value agreement (26/26)
  where both annotators found a value.** The one disagreement (Stanford scholarships) was resolved
  with the rule already applied elsewhere: a named scholarship on the scholarships page with no
  residency restriction counts as open to international students.
- **Snapshot changes.** Annotators found two cleaner bugs (a `<main>` nested inside `<nav>` by
  broken markup; collapsed accordion content marked `aria-hidden`). After each fix every
  snapshot was re-cleaned from its stored HTML (`uniagent reclean`), labels whose quote was still
  present were re-pointed automatically (`gold_tool.py refresh`), and every `NOT_ON_PAGES` label
  on a changed page was re-checked by an annotator (`rechecks.jsonl`; 1 of 9 changed: Toronto's
  deadline table became visible).
- **Split.** MIT, Toronto and Imperial are the **dev** set used while writing prompts and field
  definitions (v1's third university, Melbourne, now serves a Cloudflare challenge on every host,
  so it could not be used). Every other university is **test** and was not looked at during
  development. Reported numbers are test-only.

`NOT_ON_PAGES` labels matter: they measure whether the system abstains instead of filling a gap
from the model's memory (MIT's facts page never says "private"; the right output is null).

## Metrics (per university × field instance)

| Outcome | Gold | Prediction |
|---|---|---|
| TP | value | matching value |
| WV (wrong value) | value | different value |
| MISS | value | null |
| HAL (hallucination) | null | any value |
| CA (correct abstention) | null | null |

Precision = TP / (TP+WV+HAL) · Recall = TP / (TP+WV+MISS) · Hallucination rate = HAL / #gold-null ·
"Wrong when answering" = (WV+HAL) / answers.

Matching rules (`eval/match.py`): years exact (or any acceptable value); enrollment ±5%; tuition
same currency and ±2% after annualising; living cost ±10%; salary ±5%; acceptance rate ±1 pt;
employment ±2 pts; deadlines on month+day; visa types via a canonical label map; scholarships by
fuzzy set-F1 ≥ 0.5.

**Uncertainty:** 95% CIs from a bootstrap that resamples *universities* (fields of one university
are correlated). Systems are compared with a paired bootstrap on the same instances.

**Calibration:** ECE (equal-mass bins), Brier score, AUROC of confidence for separating right
from wrong answers, and accuracy per evidence-check outcome (accepted / derived).

## Ablation

| Run | What changes |
|---|---|
| `scrapegraph-qwen3_8b` | Off-the-shelf baseline: ScrapeGraphAI 2.2.4 `SmartScraperGraph` on the same stored HTML, same model (`qwen3:8b`, thinking off via `/no_think`), same field definitions in its prompt, no evidence check (`scripts/baseline_scrapegraph.py`). It uses its own HTML parsing and chunking, as the tool is meant to be used; 9 of its 163 page calls failed (invalid JSON or its 480 s chunk timeout) and count as empty answers |
| `v1-llama3.1` | v1 behaviour (first 4,000 chars of each page, no evidence check) with v1's model family |
| `v1-qwen3` | same, stronger local model — separates "better model" from "better system" |
| `retrieval-qwen3` | + per-field BM25 chunk retrieval over the whole page |
| `full-qwen3` | + evidence verification: answers whose quote isn't on the page are dropped |
