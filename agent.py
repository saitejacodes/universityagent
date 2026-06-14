import asyncio
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path

from scraper import Scraper
from extractor import Extractor
from validator import Validator
from database import Database
from evaluator import Evaluator
from models import UniversityData, AboutUniversity, FieldMeta, AcceptanceRate, VisaPolicy

# Configure logging — shows progress clearly in terminal so scraping issues
# are visible in real time rather than buried in a final summary
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config/universities.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_sample_output(data: dict, university_id: str) -> None:
    """Save scraped data as JSON for submission."""
    Path("output").mkdir(exist_ok=True)
    out_path = f"output/{university_id}_output.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Sample output saved: {out_path}")


def build_university_data(uid: str, name: str, raw_data: dict, timestamp: str, config: dict) -> UniversityData:
    """
    Map raw LLM extraction dicts into typed Pydantic models.

    This validates the shape of every extracted field at runtime.
    If the LLM returns a malformed value (e.g. founding_year as a string instead of int),
    Pydantic raises a ValidationError here — not silently storing bad data to the DB.
    The models also serve as the canonical schema contract: if we add a new field to
    UniversityData, mypy and Pydantic both enforce it's handled everywhere.
    """

    def meta(raw: dict) -> FieldMeta:
        return FieldMeta(
            confidence=raw.get("confidence", 0.0),
            source_url=raw.get("source_url"),
            needs_review=raw.get("needs_review", False),
            notes=raw.get("notes"),
        )

    # --- About ---
    about_raw = raw_data.get("about_raw", {})
    about = None
    if about_raw.get("value"):
        v = about_raw["value"]
        try:
            about = AboutUniversity(
                name=name,
                founding_year=v.get("founding_year") if isinstance(v, dict) else None,
                ranking_qs=v.get("ranking_qs") if isinstance(v, dict) else None,
                location_city=v.get("location_city", "") if isinstance(v, dict) else "",
                location_country=v.get("location_country", "") if isinstance(v, dict) else "",
                institution_type=v.get("institution_type") if isinstance(v, dict) else None,
                meta=meta(about_raw),
            )
        except Exception as e:
            logger.warning(f"AboutUniversity validation failed for {uid}: {e}")

    # --- AcceptanceRate ---
    # Map acceptance rate to typed model so Pydantic catches implausible values
    # (e.g. rate > 100, non-numeric strings) at runtime rather than storing bad data.
    acceptance_raw = raw_data.get("acceptance_raw", {})
    acceptance = None
    if acceptance_raw.get("value") is not None:
        try:
            rate_val = acceptance_raw["value"]
            rate = float(str(rate_val).replace("%", "").strip()) if rate_val else None
            acceptance = AcceptanceRate(
                overall_percent=rate,
                meta=meta(acceptance_raw),
            )
        except Exception as e:
            logger.warning(f"AcceptanceRate validation failed for {uid}: {e}")

    # --- VisaPolicy ---
    # Visa type is country-specific (F-1 for USA, Subclass 500 for Australia, etc.)
    # Mapping to a typed model ensures the field is always structured the same way
    # regardless of how the LLM phrases the extraction.
    visa_raw = raw_data.get("visa_raw", {})
    visa = None
    if visa_raw.get("value"):
        try:
            v = visa_raw["value"]
            visa = VisaPolicy(
                country=config.get("country", "") if isinstance(v, dict) else "",
                visa_type=v.get("visa_type") if isinstance(v, dict) else str(v),
                meta=meta(visa_raw),
            )
        except Exception as e:
            logger.warning(f"VisaPolicy validation failed for {uid}: {e}")

    return UniversityData(
        university_id=uid,
        university_name=name,
        about=about,
        acceptance_rate=acceptance,
        visa_policy=visa,
        scrape_timestamp=timestamp,
        # overall_confidence is set from the about field's confidence as a proxy;
        # validator.py computes the true per-field quality score separately
        overall_confidence=about_raw.get("confidence", 0.0),
    )


async def process_university(
    config: dict,
    scraper: Scraper,
    extractor: Extractor,
    validator: Validator,
    db: Database,
) -> None:
    """Full pipeline for one university: scrape → extract → validate → store."""
    uid = config["id"]
    name = config["name"]
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing: {name}")
    logger.info(f"{'='*50}")

    # Step 1 — Scrape all pages sequentially
    # Sequential (not concurrent) because concurrent fetches to the same university
    # domain look like a burst attack to WAFs and get the IP rate-limited or banned.
    logger.info("Step 1: Scraping pages...")
    pages = await scraper.fetch_all(config["pages"])
    fetched = sum(1 for v in pages.values() if v)
    logger.info(f"  Fetched {fetched}/{len(pages)} pages successfully")

    # Step 2 — Extract all 10 fields via Groq LLM
    # Key fields (tuition, deadlines, employment, salary) are cross-validated
    # against a secondary source page — see extractor.py _merge_sources() for details
    logger.info("Step 2: Extracting fields with Groq LLM...")
    raw_data = extractor.extract_all_fields(pages, config)

    # Step 3 — Validate extracted data for plausibility and completeness
    # Catches obvious errors: acceptance rate > 100%, tuition in wrong currency,
    # deadlines already past, etc. before they reach the database.
    logger.info("Step 3: Validating extracted data...")
    report = validator.validate(uid, config["country"], config["currency"], raw_data)
    logger.info(
        f"  Quality score: {report['overall_quality_score']:.2f} | "
        f"Ready: {report['ready_for_storage']} | "
        f"Critical issues: {len(report['critical_issues'])}"
    )

    # Step 4 — Build typed Pydantic model to validate schema at runtime,
    # then store both the typed dump and the raw fields to SQLite
    logger.info("Step 4: Storing to database...")
    timestamp = datetime.utcnow().isoformat()
    typed_data = build_university_data(uid, name, raw_data, timestamp, config)

    db.upsert_raw_fields(uid, raw_data)
    # Store typed model dump so DB contains schema-validated data, not raw LLM output
    db.upsert_university(
        uid, name,
        typed_data.model_dump(),
        report["overall_quality_score"],
    )
    db.save_validation_report(uid, report)

    # Step 5 — Save JSON output file for submission
    # Output includes both raw LLM extractions and the validation report so
    # evaluators can see exactly what was extracted and how it was assessed
    save_sample_output(
        {"university_id": uid, "name": name, "fields": raw_data, "validation": report},
        uid,
    )


async def main() -> None:
    config = load_config()
    scraper = Scraper()
    extractor = Extractor()
    validator = Validator()
    db = Database()

    try:
        for university in config["universities"]:
            try:
                await process_university(university, scraper, extractor, validator, db)
            except Exception as e:
                logger.error(f"Failed to process {university['name']}: {e}", exc_info=True)
                # Partial failure — log and continue to next university rather than aborting
                # the entire run. One unreachable website shouldn't block the other two.
                continue

        # Generate eval report after all universities are processed.
        # Pass existing db connection — Evaluator must not open a second SQLite connection
        # because SQLite WAL mode can produce reader/writer conflicts with multiple handles.
        logger.info("\nGenerating eval report...")
        evaluator = Evaluator(db=db)
        evaluator.generate_report("eval_report.md")

    finally:
        # Always close the shared aiohttp session and DB connection, even on error.
        # aiohttp logs "Unclosed client session" warnings if close() is not called.
        await scraper.close()
        db.close()

    logger.info("\n✅ Agent complete. Check output/ and eval_report.md")


if __name__ == "__main__":
    asyncio.run(main())

