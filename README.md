# University Intelligence Agent

AI-powered scraping agent using **Groq LLM (llama-3.3-70b-versatile)** to build a structured,
validated database of university intelligence across 10 fields per university.

## Architecture

```
agent.py (orchestrator)
    ↓
scraper.py   → fetches pages (robots.txt compliance, retry + exponential backoff)
    ↓
extractor.py → Groq LLM extracts 10 fields; key fields cross-validated from 2 sources
    ↓
validator.py → checks confidence, plausibility, currency match, date range
    ↓
database.py  → stores raw fields + typed Pydantic model dump to SQLite
    ↓
evaluator.py → generates eval_report.md with per-field × per-university breakdown
```

Each module has exactly one responsibility. No module calls another's internal methods.
Adding a 4th university requires **zero Python code changes** — only a YAML block.

## Requirements

- **Python**: 3.11 or 3.12 recommended (3.13+ not yet supported by pydantic-core wheels)
- **GROQ API key**: free tier available at https://console.groq.com — create an account, go to API Keys, generate a key. Free tier gives 14,400 tokens/minute which is sufficient for one full run.

## Setup

```bash
git clone <repo>
cd university-agent

# Create a virtual environment (recommended)
python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Copy the example env file and add your key
cp .env.example .env
# Edit .env and replace 'your_groq_api_key_here' with your real key

python agent.py
```

## Output

After a successful run, you will find:

```
output/
  mit_output.json        ← all 10 fields extracted for MIT, with confidence + source URLs
  unimelb_output.json    ← same for University of Melbourne
  utoronto_output.json   ← same for University of Toronto
data/
  universities.db        ← SQLite database with raw_fields, universities, validation_reports tables
eval_report.md           ← per-field × per-university confidence table + ground truth verification section
```

### Sample output structure (truncated)

```json
{
  "university_id": "mit",
  "name": "Massachusetts Institute of Technology",
  "fields": {
    "tuition_raw": {
      "value": "$59,750 per year (graduate)",
      "confidence": 0.95,
      "source_url": "https://sfs.mit.edu/graduate-students/the-cost-of-attendance/annual-student-budget/",
      "raw_snippet": "The total cost of attendance for graduate students is $59,750...",
      "needs_review": false,
      "notes": "Cross-validated: secondary source agrees. (https://web.mit.edu/aboutmit/)"
    }
  }
}
```

## Add a University (Zero Code Changes)

Edit `config/universities.yaml` and add a new block:

```yaml
- id: oxford
  name: University of Oxford
  country: UK
  currency: GBP
  pages:
    about: https://www.ox.ac.uk/about
    tuition: https://www.ox.ac.uk/admissions/graduate/fees-and-funding
    scholarships: https://www.ox.ac.uk/admissions/graduate/fees-and-funding/scholarships
    courses: https://www.ox.ac.uk/admissions/graduate/courses
    deadlines: https://www.ox.ac.uk/admissions/graduate/applying-to-oxford/when-to-apply
    visa: https://www.ox.ac.uk/students/visa
    employment: https://www.careers.ox.ac.uk/career-outcomes
    living: https://www.ox.ac.uk/students/life/accommodation
```

Then re-run `python agent.py`. No Python changes required.

## Design Decisions

- **Groq `llama3-70b-8192`** at `temperature=0.0` — deterministic output ensures the same value is returned across re-runs, making the eval report reproducible. The 70B model was chosen over 8B for significantly better instruction-following (8B hallucinates values and ignores "return null if not found").
- **Cross-validation** — tuition, deadlines, employment, and salary are extracted from two independent source pages and merged. Agreement boosts confidence by +0.1; conflicts are flagged with both values preserved in `notes` for human review.
- **Confidence per field** (not per record) — granular confidence lets the eval report identify which specific fields need attention, not just which universities.
- **Null over hallucination** — the system prompt explicitly instructs the LLM to return `null` if a field is not present. Missing data is honest; fabricated data is disqualifying.
- **robots.txt respected** — checked per domain (not per URL) before any HTTP request. Domain-level caching avoids one redundant HTTP round-trip per page fetch.
- **Sequential scraping** — pages fetched sequentially with a 2-second gap, not concurrently. Concurrent fetches to the same domain trigger WAF rate-limiting and IP bans.
- **Shared `aiohttp.ClientSession`** — one session created and reused for all fetches, then closed in a `finally` block. Creating a new session per request wastes one TCP + SSL handshake per URL.
- **Partial saves** — the pipeline continues to the next university if one fails. Partial data is still written to the DB and output JSON.
- **SQLite** for zero-dependency portability — no Postgres/MySQL server required.

## Eval

See `eval_report.md` for:

- Per-field × per-university confidence scores
- Ground truth verification table (manually verified sample)
- Methodology: how confidence was computed, which fields were cross-validated, storage threshold rationale
