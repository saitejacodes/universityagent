"""The fields we extract, defined precisely enough to be scored.

v1 asked the model for free-text blobs ("tuition_fees_all_levels") that could never be checked
automatically. Every field here has a type, a unit convention and a home page type, so a gold
label and a prediction can be compared by code: years exactly, money by currency + amount
within tolerance, percentages within one point, and so on (see eval/match.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Kind(StrEnum):
    YEAR = "year"
    INT = "int"
    PERCENT = "percent"
    MONEY = "money"
    DATE = "date"  # month + day of a deadline; year optional
    TEXT = "text"
    CHOICE = "choice"
    LIST = "list"


class PageType(StrEnum):
    FACTS = "facts"
    ADMISSIONS_STATS = "admissions_stats"
    TUITION = "tuition"
    LIVING_COSTS = "living_costs"
    SCHOLARSHIPS = "scholarships"
    DEADLINES = "deadlines"
    VISA = "visa"
    OUTCOMES = "outcomes"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: Kind
    pages: tuple[PageType, ...]  # where to look, most specific first
    instruction: str  # what exactly counts as the answer
    keywords: tuple[str, ...]  # retrieval query terms for chunk selection
    choices: tuple[str, ...] = ()


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        "founding_year",
        Kind.YEAR,
        (PageType.FACTS,),
        "Year the university was founded/established/chartered (4-digit year).",
        ("founded", "established", "chartered", "history", "since"),
    ),
    FieldSpec(
        "city",
        Kind.TEXT,
        (PageType.FACTS,),
        "City where the main campus is located (city name only).",
        ("located", "campus", "city", "based in"),
    ),
    FieldSpec(
        "institution_type",
        Kind.CHOICE,
        (PageType.FACTS,),
        "Whether the university is public or private.",
        ("public", "private", "state", "research university", "land-grant"),
        choices=("public", "private"),
    ),
    FieldSpec(
        "total_enrollment",
        Kind.INT,
        (PageType.FACTS,),
        "Total number of enrolled students (all levels combined). If only undergraduate and "
        "graduate counts are given, their sum. Headcount, most recent year.",
        ("students", "enrollment", "enrolment", "total", "headcount", "undergraduate", "graduate"),
    ),
    FieldSpec(
        "acceptance_rate",
        Kind.PERCENT,
        (PageType.ADMISSIONS_STATS,),
        "Overall undergraduate (first-year) admission/acceptance rate as a percentage.",
        ("admit rate", "acceptance rate", "admitted", "applicants", "applied", "offers"),
    ),
    FieldSpec(
        "intl_undergrad_tuition",
        Kind.MONEY,
        (PageType.TUITION,),
        "Annual undergraduate tuition fee for INTERNATIONAL students (tuition only, not total "
        "cost of attendance). If it varies by programme, use a general/typical programme and "
        "say which in the note.",
        ("tuition", "fee", "international", "overseas", "undergraduate", "per year", "annual"),
    ),
    FieldSpec(
        "living_cost",
        Kind.MONEY,
        (PageType.LIVING_COSTS, PageType.TUITION),
        "The university's official estimate of LIVING expenses for one student (housing/"
        "accommodation, food, personal) — NEVER including tuition or fees and never a 'total cost"
        " of attendance'. Use a figure the page states as a single number (e.g. 'living costs "
        "£1,400 per month', 'housing and food $22,944'); keep the page's period. If the page only "
        "itemises separate lines without such a figure, give the housing line.",
        (
            "living",
            "cost of living",
            "accommodation",
            "housing",
            "food",
            "budget",
            "expenses",
            "per month",
            "per year",
        ),
    ),
    FieldSpec(
        "application_deadline",
        Kind.DATE,
        (PageType.DEADLINES,),
        "Main application deadline for first-year undergraduate admission for the next intake "
        "(regular decision / main round / UCAS equal-consideration date).",
        (
            "deadline",
            "apply by",
            "application",
            "closing date",
            "regular decision",
            "january",
            "due",
        ),
    ),
    FieldSpec(
        "student_visa",
        Kind.TEXT,
        (PageType.VISA,),
        "Name of the student visa/permit an international degree student needs "
        "(e.g. 'F-1', 'Student visa (subclass 500)', 'Study permit').",
        ("visa", "permit", "immigration", "student route", "subclass", "F-1", "study permit"),
    ),
    FieldSpec(
        "employment_rate",
        Kind.PERCENT,
        (PageType.OUTCOMES,),
        "Share of recent graduates employed (or employed/in further study, if that's the only "
        "figure given) per the university's graduate outcomes survey.",
        ("employed", "employment", "outcomes", "graduates", "further study", "within"),
    ),
    FieldSpec(
        "median_salary",
        Kind.MONEY,
        (PageType.OUTCOMES,),
        "Median (or, if only that is published, average) starting annual salary of recent "
        "graduates, overall rather than for one programme where possible.",
        ("salary", "median", "average", "starting", "earnings", "income"),
    ),
    FieldSpec(
        "scholarships",
        Kind.LIST,
        (PageType.SCHOLARSHIPS,),
        "Names of specific scholarships/awards that the page says are open to international "
        "(overseas) undergraduate students — exclude bursaries or awards only for home/domestic "
        "students (up to 5, exact names as written).",
        ("scholarship", "award", "bursary", "grant", "international", "merit"),
    ),
)

FIELD_BY_NAME = {f.name: f for f in FIELDS}


def fields_for_page(page: PageType) -> list[FieldSpec]:
    return [f for f in FIELDS if page in f.pages]
