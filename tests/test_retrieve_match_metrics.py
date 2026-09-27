from uniagent.eval.match import set_f1, values_match
from uniagent.eval.metrics import calibration, score_instances, summary
from uniagent.extract.retrieve import chunk_markdown, select_context
from uniagent.schema import FIELD_BY_NAME as F

LONG_PAGE = "\n\n".join(
    [f"## Section {i}\n" + ("Campus life news and events. " * 30) for i in range(20)]
    + [
        "## Tuition and fees\n| Programme | International fee |\n| --- | --- |\n"
        "| Undergraduate | £38,702 per year |"
    ]
    + [f"## More {i}\n" + ("Research highlights and stories. " * 30) for i in range(20)]
)


def test_retrieval_finds_facts_beyond_the_first_4000_chars():
    assert LONG_PAGE.find("£38,702") > 4000
    ctx = select_context(LONG_PAGE, [F["intl_undergrad_tuition"]], budget=3000)
    assert "£38,702" in ctx and len(ctx) <= 3000


def test_tables_are_not_split_from_their_header():
    chunks = chunk_markdown(LONG_PAGE)
    fee = next(c for c in chunks if "£38,702" in c.text)
    assert "International fee" in fee.text


def test_match_rules():
    assert values_match("founding_year", 1861, 1861)
    assert values_match("founding_year", 1865, 1861, acceptable=[1865])
    assert values_match("acceptance_rate", 4.5, 4.6)  # within 1 point
    assert not values_match("acceptance_rate", 6.0, 4.6)
    usd = {"amount": 62396, "currency": "USD", "period": "year"}
    assert values_match("intl_undergrad_tuition", usd, {**usd, "amount": 62000})
    assert not values_match("intl_undergrad_tuition", {**usd, "currency": "CAD"}, usd)
    monthly = {"amount": 1500, "currency": "GBP", "period": "month"}
    assert values_match(
        "living_cost", monthly, {"amount": 18000, "currency": "GBP", "period": "year"}
    )
    assert values_match("student_visa", "F-1 Student Visa", "F-1")
    assert values_match("student_visa", "Student visa (subclass 500)", "subclass 500")
    assert values_match("city", "Cambridge, Massachusetts", "Cambridge")
    assert values_match(
        "application_deadline",
        {"month": 1, "day": 15, "year": 2027},
        {"month": 1, "day": 15, "year": None},
    )


def test_set_f1():
    p, r, f = set_f1(
        ["Lester B. Pearson Scholarship", "Made Up Award"],
        ["Lester B. Pearson International Scholarship", "Karen McKellin Award"],
    )
    assert p == 0.5 and r == 0.5


def test_null_aware_outcomes_and_calibration():
    gold = [
        {"university": "u", "field": "founding_year", "status": "PRESENT", "value": 1861},
        {"university": "u", "field": "acceptance_rate", "status": "NOT_ON_PAGES", "value": None},
        {"university": "u", "field": "city", "status": "PRESENT", "value": "Cambridge"},
        {"university": "u", "field": "median_salary", "status": "NOT_ON_PAGES", "value": None},
    ]
    preds = {
        ("u", "founding_year"): {"value": 1861, "confidence": 0.9, "status": "accepted"},
        ("u", "acceptance_rate"): {"value": 5.0, "confidence": 0.2, "status": "x"},  # HAL
        ("u", "city"): {"value": None},  # MISS
    }  # CA
    inst = score_instances(gold, preds)
    s = summary(inst)
    assert (s["TP"], s["HAL"], s["MISS"], s["CA"]) == (1, 1, 1, 1)
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["hallucination_rate"] == 0.5
    cal = calibration(inst)
    assert cal["n"] == 2 and cal["auroc"] == 1.0
