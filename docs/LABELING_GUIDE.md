# Gold labelling guide

These are the exact instructions every annotator followed. Labels grade an automated system, so
they must be exact, conservative and verifiable.

Work **only** from the stored snapshots. Do not browse the web or use background knowledge of
universities. Never open `runs/`, `.cache/` or any `predictions*.json` (that would contaminate
the benchmark). For double annotation, also never open other annotators' files in `data/gold/`.

1. **Field definitions** (these are the labelling rules):
   `.venv/bin/python scripts/gold_tool.py fields`
2. **Snapshot files for a university:**
   `.venv/bin/python scripts/gold_tool.py pages <university_id>`
   A field may only be labelled from the page types listed for it (e.g. `founding_year` only from
   `facts`; `living_cost` from `living_costs`, else `tuition`).
3. **One label per field (12 per university), JSON lines:**
   ```json
   {"university": "<id>", "field": "<field>", "status": "PRESENT" | "NOT_ON_PAGES",
    "value": <canonical value or null>, "acceptable_values": [],
    "evidence_quote": "<exact substring of the .md file that contains the value>" | null,
    "page_type": "<page type of the quote>", "snapshot_sha": "<sha from the pages listing>",
    "annotator": "<annotator id>", "notes": "<short reasoning, caveats>"}
   ```
   Canonical formats: year/int → integer; percent → number 0–100; money →
   `{"amount": 66720, "currency": "USD", "period": "year|month|week|term"}`; date →
   `{"month": 1, "day": 4, "year": null}`; text → short string; institution_type → `public` |
   `private`; scholarships → up to 5 exact names (quote optional).
4. **Rules**
   - `PRESENT` only if the page **explicitly** states the fact. If it would need world knowledge
     (a facts page that never says "private"), label `NOT_ON_PAGES`.
   - A figure for a different population (graduate tuition, domestic fees, one programme) needs
     judgement: pick the closest match to the definition, explain in notes, and put other
     defensible figures in `acceptable_values`.
   - `evidence_quote` is copied character-for-character from the `.md` file, as short as possible
     (< 300 chars) while containing the value.
   - Derived values only where the definition allows it (e.g. enrollment = UG + PG); start notes
     with `derived:` and quote the parts.
   - No snapshot for the field's page types → `NOT_ON_PAGES`, `page_type` and `snapshot_sha`
     null, notes `no snapshot` (these instances are excluded from scoring).
   - When a page gives a figure per month and per year, use the emphasised one and add the other
     to `acceptable_values`.
5. **Audit until clean:** `.venv/bin/python scripts/gold_tool.py check <file>` must print
   `0 problem(s)` (quote exists in the cited snapshot, value appears in the quote, formats valid).

## Adjudication rules (applied after review, recorded in `data/gold/adjudications.jsonl`)

- **City:** stated when the facts page names it, including via the university's official name
  when that name is a city ("UC Berkeley", "Imperial College London", "TU Delft").
- **Scholarships:** a named scholarship on the scholarships page with no residency restriction
  counts as open to international students.
