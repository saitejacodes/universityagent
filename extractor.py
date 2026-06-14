import os
import json
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# llama-3.3-70b-versatile is the current production successor to llama3-70b-8192
# (which was decommissioned by Groq on 2025-06-14). The 70B class is chosen over
# 8B because it reliably returns valid JSON and correctly handles "return null if
# not found" — the 8B model frequently hallucinates values or ignores the null instruction.
# 128k context window gives us room to pass full page text without truncation concerns.
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an expert university data extraction agent.

RULES:
- Extract ONLY what is explicitly written in the provided text
- Never guess, hallucinate, or infer values not present in the text
- If a field cannot be found, return null for value and 0.0 for confidence
- Assign confidence: 1.0=explicit match, 0.8=strongly implied, 0.6=indirect, 0.0=not found
- All monetary values must include currency code
- All dates must be YYYY-MM-DD format
- Return ONLY valid JSON — no explanation, no markdown, no backticks"""

EXTRACTION_PROMPT = """University: {university_name}
Country: {country}
Currency: {currency}
Source URL: {source_url}
Field to extract: {field_name}

Page text:
{page_text}

Return ONLY this JSON (no extra text):
{{
  "field": "{field_name}",
  "value": <extracted value or null>,
  "confidence": <0.0 to 1.0>,
  "source_url": "{source_url}",
  "raw_snippet": "<exact text found, max 150 chars, or null>",
  "needs_review": <true if confidence < 0.7>,
  "notes": "<any flags or caveats, or null>"
}}"""


class Extractor:
    def __init__(self) -> None:
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def _call_groq(self, user_prompt: str) -> dict:
        """
        Call Groq API and parse the JSON response.

        temperature=0.0: extraction is a lookup task, not a generation task.
        Any non-zero temperature introduces randomness into field values — unacceptable
        when the same tuition fee must be consistent across re-runs for eval purposes.
        A hallucinated value at temperature=0.3 could score 0.95 confidence and still
        be wrong, and there's no way to detect that without cross-validation.
        """
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content.strip()
            # Some models prepend ```json fences despite explicit instructions not to.
            # Stripping them here is a defensive measure — we shouldn't need it, but
            # it prevents a hard crash when the model ignores formatting instructions.
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return {"value": None, "confidence": 0.0, "needs_review": True, "notes": "parse_error"}
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return {"value": None, "confidence": 0.0, "needs_review": True, "notes": str(e)}

    def extract_field(
        self,
        field_name: str,
        page_text: str,
        source_url: str,
        university_name: str,
        country: str,
        currency: str,
    ) -> dict:
        """
        Extract a single field from page text using Groq.

        Page text is capped at 6,000 chars before being sent.
        llama3-70b-8192 has an 8,192-token context window.
        The SYSTEM_PROMPT + EXTRACTION_PROMPT template consumes ~400 tokens,
        and we reserve ~1,400 tokens for the JSON response. That leaves ~6,400
        tokens for page text. 6,000 chars ≈ 1,500 tokens — comfortable headroom
        that avoids silent truncation mid-sentence which corrupts extraction.
        """
        prompt = EXTRACTION_PROMPT.format(
            field_name=field_name,
            page_text=page_text[:6000],
            source_url=source_url,
            university_name=university_name,
            country=country,
            currency=currency,
        )
        result = self._call_groq(prompt)
        logger.info(
            f"  [{field_name}] confidence={result.get('confidence', 0):.2f} "
            f"needs_review={result.get('needs_review', False)}"
        )
        return result

    def _merge_sources(self, primary: dict, secondary: dict, field: str) -> dict:
        """
        Merge two extractions of the same field from different source pages.

        Cross-validation strategy:
        - If both sources return a value and they AGREE → boost confidence by 0.1
          (two independent pages agreeing is strong evidence the value is correct)
        - If both return a value but they CONFLICT → flag needs_review=True and
          preserve both values in notes for human review
        - If only one source found a value → use that source's result as-is
        - If neither found anything → return primary (both null)

        This catches cases where a tuition page says "$59,750" but the scholarships
        page says "$61,000" — a discrepancy that would be invisible with single-source
        extraction. Conflicts are expected for international vs domestic fees being
        confused, or stale data on one page.
        """
        p_val = primary.get("value")
        s_val = secondary.get("value")

        if p_val is None and s_val is None:
            return primary  # both failed — return primary with its null

        if p_val is None:
            return secondary  # only secondary found something — use it

        if s_val is None:
            return primary  # only primary found something — use it

        # Both found values — compare as strings for simplicity
        if str(p_val).strip().lower() == str(s_val).strip().lower():
            # Agreement: two independent pages confirm the same value.
            # Boost confidence by 0.1, capped at 1.0.
            primary["confidence"] = min(1.0, primary.get("confidence", 0.0) + 0.1)
            primary["notes"] = (
                f"Cross-validated: secondary source agrees. "
                f"({secondary.get('source_url')})"
            )
            primary["needs_review"] = False
            return primary
        else:
            # Conflict: flag for human review with both values visible
            primary["needs_review"] = True
            primary["notes"] = (
                f"CONFLICT: primary={p_val} (from {primary.get('source_url')}) "
                f"vs secondary={s_val} (from {secondary.get('source_url')})"
            )
            return primary

    def extract_all_fields(
        self,
        pages: dict[str, str | None],
        config: dict,
    ) -> dict[str, list | dict | None]:
        """
        Extract all 10 intelligence fields from scraped pages.
        Fields mapped to the most data-rich page for that field.
        Tuition and deadlines are cross-validated against a second source.
        Employment and salary use a dedicated career outcomes page (not the about page)
        because about pages contain mission statements, not employment statistics.
        """
        name = config["name"]
        country = config["country"]
        currency = config["currency"]
        urls = config["pages"]

        def get_text(label: str) -> str:
            return pages.get(label) or ""

        def get_url(label: str) -> str:
            return urls.get(label, "")

        results = {}

        # Field 1 — About University
        about_raw = self.extract_field(
            "about_university",
            get_text("about"),
            get_url("about"),
            name, country, currency,
        )
        results["about_raw"] = about_raw

        # Field 2 — Tuition Fees: extract from dedicated tuition page, then
        # cross-validate against the about page. Universities sometimes list
        # headline fee figures in their about/overview copy that differ from
        # the detailed fee schedule — flagging this conflict is valuable signal.
        tuition_primary = self.extract_field(
            "tuition_fees_all_levels",
            get_text("tuition"),
            get_url("tuition"),
            name, country, currency,
        )
        tuition_secondary = self.extract_field(
            "tuition_fees_all_levels",
            get_text("about"),
            get_url("about"),
            name, country, currency,
        )
        results["tuition_raw"] = self._merge_sources(tuition_primary, tuition_secondary, field="tuition")

        # Field 3 — Living Costs: dedicated living page where available, fall back to tuition page
        # (many universities publish living cost estimates alongside tuition in one budget table)
        living_raw = self.extract_field(
            "living_costs_monthly_breakdown",
            get_text("living") or get_text("tuition"),
            get_url("living") or get_url("tuition"),
            name, country, currency,
        )
        results["living_raw"] = living_raw

        # Field 4 — Scholarships
        scholarships_raw = self.extract_field(
            "scholarships_list_with_value_and_eligibility",
            get_text("scholarships"),
            get_url("scholarships"),
            name, country, currency,
        )
        results["scholarships_raw"] = scholarships_raw

        # Field 5 — Acceptance Rate: about page is the correct source (admissions stats
        # are in institutional overview pages, not fee or deadline pages)
        acceptance_raw = self.extract_field(
            "acceptance_rate_percentage",
            get_text("about") or get_text("admissions"),
            get_url("about"),
            name, country, currency,
        )
        results["acceptance_raw"] = acceptance_raw

        # Field 6 — Graduate Employment: dedicated career outcomes page gives real employment
        # stats. Previously fell back to about page which only has mission-statement prose.
        employment_primary = self.extract_field(
            "graduate_employment_rate_within_6_months",
            get_text("employment"),
            get_url("employment"),
            name, country, currency,
        )
        employment_secondary = self.extract_field(
            "graduate_employment_rate_within_6_months",
            get_text("about"),
            get_url("about"),
            name, country, currency,
        )
        results["employment_raw"] = self._merge_sources(employment_primary, employment_secondary, field="employment")

        # Field 7 — Average Salaries: same career outcomes page as employment
        salary_primary = self.extract_field(
            "average_graduate_salaries_by_field",
            get_text("employment"),
            get_url("employment"),
            name, country, currency,
        )
        salary_secondary = self.extract_field(
            "average_graduate_salaries_by_field",
            get_text("about"),
            get_url("about"),
            name, country, currency,
        )
        results["salary_raw"] = self._merge_sources(salary_primary, salary_secondary, field="salary")

        # Field 8 — Visa Policies: visa/immigration office page is the authoritative source
        visa_raw = self.extract_field(
            "student_visa_type_requirements_processing_time",
            get_text("visa"),
            get_url("visa"),
            name, country, currency,
        )
        results["visa_raw"] = visa_raw

        # Field 9 — Intake Deadlines: cross-validate deadlines page against scholarships page.
        # Scholarship deadlines are often earlier than general application deadlines —
        # a conflict here is not an error but an important distinction to flag for review.
        deadlines_primary = self.extract_field(
            "application_deadlines_per_intake",
            get_text("deadlines"),
            get_url("deadlines"),
            name, country, currency,
        )
        deadlines_secondary = self.extract_field(
            "application_deadlines_per_intake",
            get_text("scholarships"),
            get_url("scholarships"),
            name, country, currency,
        )
        results["deadlines_raw"] = self._merge_sources(deadlines_primary, deadlines_secondary, field="deadlines")

        # Field 10 — Course Listings
        courses_raw = self.extract_field(
            "course_listings_with_code_credits_description",
            get_text("courses"),
            get_url("courses"),
            name, country, currency,
        )
        results["courses_raw"] = courses_raw

        return results
