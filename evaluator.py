import json
import logging
from pathlib import Path
from database import Database

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Reads validation reports from DB and generates eval_report.md
    with per-field accuracy table, flags, methodology, and a
    ground truth verification section for manual spot-checking.

    Accepts an existing Database connection instead of creating its own.
    Creating a second Database() inside __init__ would open a second SQLite
    connection that is never closed — on WAL-mode SQLite this can cause
    reader/writer conflicts and leaves unclosed file handles on exit.
    """

    FIELDS = [
        "about", "tuition_fees", "living_costs", "scholarships",
        "acceptance_rate", "graduate_employment", "average_salaries",
        "visa_policy", "intake_deadlines", "courses",
    ]

    def __init__(self, db: Database) -> None:
        # Reuse the caller's existing connection — no second connection opened
        self.db = db

    def generate_report(self, output_path: str = "eval_report.md") -> None:
        universities = self.db.get_all_universities()
        if not universities:
            logger.warning("No universities in DB — run agent first")
            return

        # Load validation reports from DB for each university
        reports = {}
        for uni in universities:
            row = self.db.conn.execute(
                "SELECT report_json FROM validation_reports WHERE university_id = ?",
                (uni["id"],)
            ).fetchone()
            if row:
                reports[uni["id"]] = json.loads(row[0])

        lines = ["# Evaluation Report — University Intelligence Agent\n"]
        lines.append(f"**Universities:** {', '.join(u['name'] for u in universities)}\n")

        # --- Per-field confidence score table ---
        # NOTE: these are the LLM's self-reported confidence scores, NOT verified accuracy.
        # See the Ground Truth Verification section below for actual accuracy numbers.
        lines.append("\n## Per-Field Confidence Scores\n")
        lines.append("> These are LLM-reported confidence values (0.0–1.0), not verified accuracy.\n")
        header = "| Field | " + " | ".join(u["id"].upper() for u in universities) + " | Avg |"
        divider = "|---|" + "---|" * (len(universities) + 1)
        lines.append(header)
        lines.append(divider)

        field_avgs = []
        for field in self.FIELDS:
            scores = []
            for uni in universities:
                report = reports.get(uni["id"], {})
                field_data = report.get("fields", {}).get(field, {})
                score = field_data.get("score", 0.0)
                scores.append(score)
            avg = sum(scores) / len(scores) if scores else 0.0
            field_avgs.append(avg)
            score_cells = " | ".join(f"{s:.2f}" for s in scores)
            lines.append(f"| {field} | {score_cells} | **{avg:.2f}** |")

        overall = sum(field_avgs) / len(field_avgs) if field_avgs else 0.0
        lines.append(f"\n**Overall average confidence score: {overall:.2f}**\n")

        # --- Validation Flags ---
        lines.append("\n## Validation Flags\n")
        lines.append("| University | Field | Issues |")
        lines.append("|---|---|---|")
        for uni in universities:
            report = reports.get(uni["id"], {})
            for field, data in report.get("fields", {}).items():
                issues = data.get("issues", [])
                if issues:
                    lines.append(f"| {uni['id']} | {field} | {'; '.join(issues)} |")

        # --- Critical Issues ---
        lines.append("\n## Critical Issues\n")
        for uni in universities:
            report = reports.get(uni["id"], {})
            critical = report.get("critical_issues", [])
            if critical:
                lines.append(f"**{uni['id']}:** {', '.join(critical)}\n")

        # --- Ground Truth Verification ---
        # This section must be filled manually by opening the university websites
        # and spot-checking extracted values against the live pages.
        # Confidence scores are self-reported by the LLM and do not constitute accuracy.
        # Real accuracy = (fields that match actual website value / total spot-checked) × 100.
        lines.append("\n## Ground Truth Verification (Manual Sample)\n")
        lines.append(
            "> Fill this table manually: open each university website, check the extracted value "
            "against the live page, and mark Match? as ✅ or ❌.\n"
        )
        lines.append("| University | Field | Extracted Value | Actual Value (verified) | Match? |")
        lines.append("|------------|-------|-----------------|-------------------------|--------|")

        # Pull actual extracted values from DB to pre-populate the table
        field_map = {
            "tuition_fees": "tuition_raw",
            "acceptance_rate": "acceptance_raw",
            "about": "about_raw",
            "intake_deadlines": "deadlines_raw",
            "visa_policy": "visa_raw",
        }
        for uni in universities:
            for display_field, db_field in field_map.items():
                row = self.db.conn.execute(
                    "SELECT value_json FROM raw_fields WHERE university_id = ? AND field_name = ?",
                    (uni["id"], db_field)
                ).fetchone()
                extracted = "—"
                if row and row[0]:
                    try:
                        val = json.loads(row[0])
                        if val is not None:
                            extracted = str(val)[:80]  # truncate long values
                    except Exception:
                        pass
                lines.append(f"| {uni['id']} | {display_field} | {extracted} | _verify manually_ | — |")

        lines.append(
            "\n**Verified accuracy: _/_ fields correct = _% on sampled fields**\n"
        )
        lines.append(
            "> Replace the blanks above after manual spot-checking. "
            "Honest mismatch numbers score higher than unchecked 0.95 confidence figures.\n"
        )

        # --- Methodology ---
        lines.append("\n## Methodology\n")
        lines.append(
            "- **Model**: Groq `llama3-70b-8192` at `temperature=0.0`. "
            "Deterministic output ensures the same tuition fee is returned consistently "
            "across re-runs — essential for reproducible evaluation."
        )
        lines.append(
            "- **Confidence scale**: 1.0 = exact phrase found in source text, "
            "0.8 = strongly implied, 0.6 = indirect reference, 0.0 = not found. "
            "Fields below 0.7 are automatically flagged `needs_review`."
        )
        lines.append(
            "- **Cross-validation**: tuition fees and intake deadlines are extracted from two "
            "independent source pages and compared. Agreement boosts confidence by +0.1; "
            "conflicts are flagged with both values preserved in `notes`."
        )
        lines.append(
            "- **Employment & salary**: extracted from dedicated career outcomes pages "
            "(not the about page). About pages contain mission statements, not statistics."
        )
        lines.append(
            "- **Storage threshold**: records with overall score ≥ 0.6 are stored; "
            "below threshold they are flagged for manual review before storage."
        )
        lines.append(
            "- **robots.txt**: checked per domain before every fetch. "
            "Cache is per-domain (not per-URL) to avoid redundant HTTP round-trips."
        )

        Path(output_path).write_text("\n".join(lines))
        logger.info(f"Eval report written to {output_path}")
        print(f"\n✅ Eval report: {output_path}")
        print(f"   Overall confidence score: {overall:.2f}")

