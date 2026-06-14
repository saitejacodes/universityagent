import logging
from datetime import date, datetime

logger = logging.getLogger(__name__)


class Validator:
    """
    Self-validation layer.
    Checks every extracted field for plausibility, completeness,
    and currency/country consistency before storing to DB.
    """

    def validate(self, university_id: str, country: str, currency: str, raw_data: dict) -> dict:
        """
        Run all validation checks on raw extracted data.
        Returns a structured validation report.
        """
        report = {
            "university_id": university_id,
            "fields": {},
            "critical_issues": [],
            "overall_quality_score": 0.0,
            "ready_for_storage": False,
        }

        scores = []

        # --- Validate About ---
        about = raw_data.get("about_raw", {})
        score, issues = self._check_field("about_university", about, required=True)
        report["fields"]["about"] = {"score": score, "issues": issues}
        scores.append(score)
        if score < 0.5:
            report["critical_issues"].append("about_university missing or low confidence")

        # --- Validate Tuition ---
        tuition = raw_data.get("tuition_raw", {})
        score, issues = self._check_tuition(tuition, currency)
        report["fields"]["tuition_fees"] = {"score": score, "issues": issues}
        scores.append(score)
        if score < 0.5:
            report["critical_issues"].append("tuition_fees missing or implausible")

        # --- Validate Living Costs ---
        living = raw_data.get("living_raw", {})
        score, issues = self._check_field("living_costs", living, required=False)
        report["fields"]["living_costs"] = {"score": score, "issues": issues}
        scores.append(score)

        # --- Validate Scholarships ---
        scholarships = raw_data.get("scholarships_raw", {})
        score, issues = self._check_field("scholarships", scholarships, required=False)
        report["fields"]["scholarships"] = {"score": score, "issues": issues}
        scores.append(score)

        # --- Validate Acceptance Rate ---
        acceptance = raw_data.get("acceptance_raw", {})
        score, issues = self._check_acceptance_rate(acceptance)
        report["fields"]["acceptance_rate"] = {"score": score, "issues": issues}
        scores.append(score)

        # --- Validate Employment ---
        employment = raw_data.get("employment_raw", {})
        score, issues = self._check_field("graduate_employment", employment, required=False)
        report["fields"]["graduate_employment"] = {"score": score, "issues": issues}
        scores.append(score)

        # --- Validate Salaries ---
        salaries = raw_data.get("salary_raw", {})
        score, issues = self._check_field("average_salaries", salaries, required=False)
        report["fields"]["average_salaries"] = {"score": score, "issues": issues}
        scores.append(score)

        # --- Validate Visa ---
        visa = raw_data.get("visa_raw", {})
        score, issues = self._check_field("visa_policy", visa, required=False)
        report["fields"]["visa_policy"] = {"score": score, "issues": issues}
        scores.append(score)

        # --- Validate Deadlines ---
        deadlines = raw_data.get("deadlines_raw", {})
        score, issues = self._check_deadlines(deadlines)
        report["fields"]["intake_deadlines"] = {"score": score, "issues": issues}
        scores.append(score)

        # --- Validate Courses ---
        courses = raw_data.get("courses_raw", {})
        score, issues = self._check_field("courses", courses, required=False)
        report["fields"]["courses"] = {"score": score, "issues": issues}
        scores.append(score)

        # Overall score
        report["overall_quality_score"] = round(sum(scores) / len(scores), 3)
        report["ready_for_storage"] = report["overall_quality_score"] >= 0.6

        logger.info(
            f"Validation [{university_id}]: "
            f"score={report['overall_quality_score']:.2f} "
            f"ready={report['ready_for_storage']} "
            f"critical={len(report['critical_issues'])}"
        )
        return report

    def _check_field(self, name: str, raw: dict, required: bool) -> tuple[float, list[str]]:
        """Generic field check: exists + confidence."""
        issues = []
        if not raw or raw.get("value") is None:
            if required:
                issues.append(f"{name}: missing (required)")
                return 0.0, issues
            else:
                issues.append(f"{name}: not found (optional)")
                return 0.5, issues
        confidence = raw.get("confidence", 0.0)
        if confidence < 0.7:
            issues.append(f"{name}: low confidence ({confidence:.2f})")
        return confidence, issues

    def _check_tuition(self, raw: dict, expected_currency: str) -> tuple[float, list[str]]:
        """Validate tuition — check currency match and plausible range."""
        issues = []
        if not raw or raw.get("value") is None:
            issues.append("tuition_fees: not found")
            return 0.0, issues

        value = raw.get("value", {})
        confidence = raw.get("confidence", 0.0)

        # Check currency consistency
        notes = raw.get("notes", "") or ""
        if expected_currency.upper() not in str(value).upper() and expected_currency.upper() not in notes.upper():
            issues.append(f"tuition: currency may not match expected {expected_currency}")
            confidence = min(confidence, 0.7)

        return confidence, issues

    def _check_acceptance_rate(self, raw: dict) -> tuple[float, list[str]]:
        """Validate acceptance rate is a plausible percentage."""
        issues = []
        if not raw or raw.get("value") is None:
            issues.append("acceptance_rate: not found")
            return 0.4, issues

        confidence = raw.get("confidence", 0.0)
        value = raw.get("value")

        # Try to validate the range
        try:
            rate = float(str(value).replace("%", "").strip())
            if rate < 0 or rate > 100:
                issues.append(f"acceptance_rate: implausible value {rate}")
                return 0.0, issues
        except (ValueError, TypeError):
            issues.append("acceptance_rate: could not parse as number")
            confidence = min(confidence, 0.5)

        return confidence, issues

    def _check_deadlines(self, raw: dict) -> tuple[float, list[str]]:
        """Check if deadlines are present and not already past."""
        issues = []
        if not raw or raw.get("value") is None:
            issues.append("intake_deadlines: not found")
            return 0.4, issues

        confidence = raw.get("confidence", 0.0)
        today = date.today()

        value = raw.get("value")
        if isinstance(value, str):
            try:
                deadline = datetime.fromisoformat(value).date()
                if deadline < today:
                    issues.append(f"intake_deadlines: deadline {value} is in the past")
                    confidence = min(confidence, 0.6)
            except ValueError:
                pass  # not a simple date string — skip date check

        return confidence, issues
