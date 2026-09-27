import pytest

from uniagent.extract.grounding import canon, verify
from uniagent.extract.normalize import normalize, parse_date, parse_money
from uniagent.schema import FIELD_BY_NAME as F


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("$62,396 per year", {"amount": 62396.0, "currency": "USD", "period": "year"}),
        ("A$49,000", {"amount": 49000.0, "currency": "AUD", "period": "year"}),
        ("£1,250 per month", {"amount": 1250.0, "currency": "GBP", "period": "month"}),
        (
            {"amount": "38,702", "currency": "gbp", "period": "annual"},
            {"amount": 38702.0, "currency": "GBP", "period": "year"},
        ),
        ("CAD 60,510", {"amount": 60510.0, "currency": "CAD", "period": "year"}),
    ],
)
def test_parse_money(raw, expected):
    assert parse_money(raw, default_currency="USD") == expected


def test_dollar_sign_uses_local_currency():
    assert parse_money("$60,000", default_currency="CAD")["currency"] == "CAD"


@pytest.mark.parametrize(
    "raw,md",
    [
        ("January 1", (1, 1)),
        ("15 January 2027", (1, 15)),
        ("2027-01-15", (1, 15)),
        ("Jan. 5th", (1, 5)),
        ("1st of February", (2, 1)),
    ],
)
def test_parse_date(raw, md):
    d = parse_date(raw)
    assert (d["month"], d["day"]) == md


def test_normalize_by_kind():
    assert normalize(F["founding_year"], "Founded in 1861") == 1861
    assert normalize(F["founding_year"], "3024") is None  # impossible year
    assert normalize(F["acceptance_rate"], "4.6%") == 4.6
    assert normalize(F["acceptance_rate"], 0.046) == 4.6  # fraction -> percent
    assert normalize(F["total_enrollment"], "11,816 students") == 11816
    assert normalize(F["institution_type"], "A private research university") == "private"
    assert normalize(F["city"], "not found") is None
    assert normalize(F["scholarships"], ["A Award", "a award", "B Grant"]) == ["A Award", "B Grant"]


PAGE = canon("""
## Admissions statistics for the Class of 2029
| First-year applications | 29,281 |
| Percentage admitted | 4.6% |
MIT was founded in 1861 in Boston and moved to Cambridge in 1916.
""")


def test_grounded_quote_with_value_is_accepted():
    g = verify(F["acceptance_rate"], 4.6, "| Percentage admitted | 4.6% |", PAGE)
    assert g.grounded and g.value_in_quote


def test_fabricated_quote_is_rejected():
    g = verify(F["acceptance_rate"], 3.9, "The acceptance rate was 3.9% last year.", PAGE)
    assert not g.grounded


def test_real_quote_but_value_not_in_it_is_flagged():
    g = verify(F["founding_year"], 1916, "MIT was founded in 1861 in Boston", PAGE)
    assert g.grounded and not g.value_in_quote


def test_quote_matching_tolerates_markup_and_whitespace():
    g = verify(F["founding_year"], 1861, "MIT was founded in   1861 in Boston and moved", PAGE)
    assert g.grounded and g.value_in_quote


@pytest.mark.parametrize(
    "raw,amount",
    [
        ("€ 19.906 per year", 19906.0),
        ("€14.200", 14200.0),
        ("CHF 20´000", 20000.0),
        ("1.234.567,50 EUR", 1234567.5),
        ("$59,750", 59750.0),
    ],
)
def test_european_and_swiss_number_formats(raw, amount):
    assert parse_money(raw, default_currency="EUR")["amount"] == amount


def test_decimal_points_still_decimal_for_percentages():
    assert normalize(F["acceptance_rate"], "4.6%") == 4.6
    assert normalize(F["total_enrollment"], "26.000+ students") == 26000


def test_european_day_month_dates():
    d = parse_date("Application period 15.05. – 15.07.")
    assert (d["month"], d["day"]) == (5, 15)
    assert parse_date("deadline 15.07.2026") == {"month": 7, "day": 15, "year": 2026}


def test_grounding_accepts_european_formats():
    page = canon("Tuition: € 19.906 per year. Students: 26.000+. Apply 15.05. – 15.07.")
    assert verify(
        F["intl_undergrad_tuition"],
        {"amount": 19906.0, "currency": "EUR", "period": "year"},
        "Tuition: € 19.906 per year",
        page,
    ).value_in_quote
    assert verify(F["total_enrollment"], 26000, "Students: 26.000+", page).value_in_quote
    assert verify(
        F["application_deadline"],
        {"month": 7, "day": 15, "year": None},
        "Apply 15.05. – 15.07.",
        page,
    ).value_in_quote
    assert not verify(F["total_enrollment"], 27000, "Students: 26.000+", page).value_in_quote
