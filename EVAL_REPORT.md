# Extraction evaluation

- Split: **test** — 23 universities, 223 (university, field) instances; 175 with a value on the official pages, 48 where the pages do not state it (the right answer is null).
- 53 instances excluded because no page of the required type could be fetched (nothing to extract from).
- 95% CIs: bootstrap over universities (2,000 resamples).

## Systems

| Run | Model | Precision | Recall | F1 [95% CI] | Hallucination rate [95% CI] | Wrong when answering | p(not better than prev) |
|---|---|---|---|---|---|---|---|
| `scrapegraph-qwen3_8b` | `scrapegraphai/qwen3:8b` | 74.7 | 32.0 | **44.8** [34.2–54.0] | 25.0 [13.6–36.7] | 25.3 |  |
| `v1-llama3.1` | `llama3.1:8b` | 67.8 | 70.9 | **69.3** [60.0–77.7] | 60.4 [47.9–72.2] | 32.2 | 0.000 |
| `v1-qwen3` | `qwen3:8b` | 73.2 | 74.9 | **74.0** [65.3–82.1] | 50.0 [37.0–62.2] | 26.8 | 0.048 |
| `retrieval-qwen3` | `qwen3:8b` | 74.4 | 82.9 | **78.4** [71.4–84.5] | 54.2 [45.0–63.5] | 25.6 | 0.113 |
| `full-qwen3` | `qwen3:8b` | 83.4 | 74.9 | **78.9** [72.3–84.8] | 12.5 [5.1–19.0] | 16.6 | 0.116 |

**`full-qwen3` vs earlier systems** (paired bootstrap over universities, p = probability it is *not* better on null-aware accuracy):

- vs `scrapegraph-qwen3_8b`: p = 0.000
- vs `v1-llama3.1`: p = 0.002
- vs `v1-qwen3`: p = 0.035
- vs `retrieval-qwen3`: p = 0.116

## Per field (`full-qwen3`)

| Field | TP | Wrong value | Missed | Hallucinated | Correct null | F1 |
|---|---|---|---|---|---|---|
| acceptance_rate | 1 | 2 | 0 | 2 | 0 | 25.0 |
| application_deadline | 13 | 5 | 4 | 0 | 1 | 65.0 |
| city | 19 | 0 | 1 | 2 | 1 | 92.7 |
| employment_rate | 8 | 2 | 0 | 0 | 1 | 80.0 |
| founding_year | 14 | 0 | 1 | 0 | 8 | 96.6 |
| institution_type | 5 | 0 | 0 | 0 | 18 | 100.0 |
| intl_undergrad_tuition | 10 | 3 | 5 | 1 | 0 | 62.5 |
| living_cost | 12 | 4 | 3 | 1 | 2 | 66.7 |
| median_salary | 4 | 0 | 1 | 0 | 6 | 88.9 |
| scholarships | 9 | 4 | 3 | 0 | 1 | 62.1 |
| student_visa | 19 | 0 | 4 | 0 | 0 | 90.5 |
| total_enrollment | 17 | 0 | 2 | 0 | 4 | 94.4 |

## Serving with a confidence threshold of 0.5 (`full-qwen3`)

| Precision | Recall | F1 | Hallucination rate | Wrong when answering |
|---|---|---|---|---|
| 83.9 | 74.3 | 78.8 | 12.5 | 16.1 |

## Is the confidence score honest? (`full-qwen3`)

- ECE 0.088 · Brier 0.117 · AUROC 0.658 (over 157 answered instances)

| Evidence check result | Answers | Accuracy |
|---|---|---|
| accepted | 141 | 88.7 |
| flagged | 2 | 50.0 |
| derived | 14 | 35.7 |

## Every wrong answer (for error analysis)

| University | Field | Predicted | Gold | Outcome |
|---|---|---|---|---|
| berkeley | acceptance_rate | `4.6` | `null` | HAL |
| edinburgh | acceptance_rate | `4.6` | `53.0` | WV |
| mcgill | acceptance_rate | `4.6` | `null` | HAL |
| uw | acceptance_rate | `4.6` | `39.0` | WV |
| anu | application_deadline | `{"month": 6, "day": 14, "year": 2026}` | `{"month": 12, "day": 15, "year": 2026}` | WV |
| auckland | application_deadline | `{"month": 7, "day": 1, "year": 2027}` | `{"month": 12, "day": 8, "year": 2026}` | WV |
| berkeley | application_deadline | `{"month": 10, "day": 1, "year": null}` | `{"month": 11, "day": 30, "year": null}` | WV |
| uq | application_deadline | `{"month": 11, "day": 30, "year": null}` | `{"month": 12, "day": 7, "year": 2026}` | WV |
| usyd | application_deadline | `{"month": 1, "day": 15, "year": null}` | `{"month": 12, "day": 1, "year": null}` | WV |
| stanford | city | `"Stanford"` | `null` | HAL |
| uq | city | `"Brisbane"` | `null` | HAL |
| anu | employment_rate | `4.6` | `90.0` | WV |
| edinburgh | employment_rate | `85.0` | `70.0` | WV |
| auckland | intl_undergrad_tuition | `{"amount": 59750.0, "currency": "NZD", "period": "year"}` | `{"amount": 40225.0, "currency": "NZD", "period": "year"}` | WV |
| berkeley | intl_undergrad_tuition | `{"amount": 37602.0, "currency": "USD", "period": "year"}` | `{"amount": 60140.0, "currency": "USD", "period": "year"}` | WV |
| ubc | intl_undergrad_tuition | `{"amount": 1769.4, "currency": "CAD", "period": "term"}` | `null` | HAL |
| uq | intl_undergrad_tuition | `{"amount": 8355.0, "currency": "AUD", "period": "year"}` | `{"amount": 60952.0, "currency": "AUD", "period": "year"}` | WV |
| cmu | living_cost | `{"amount": 91214.0, "currency": "USD", "period": "year"}` | `{"amount": 11700.0, "currency": "USD", "period": "year"}` | WV |
| edinburgh | living_cost | `{"amount": 248.0, "currency": "GBP", "period": "month"}` | `{"amount": 1546.0, "currency": "GBP", "period": "month"}` | WV |
| ethz | living_cost | `{"amount": 26000.0, "currency": "CHF", "period": "year"}` | `null` | HAL |
| iitb | living_cost | `{"amount": 22500.0, "currency": "INR", "period": "term"}` | `{"amount": 19950.0, "currency": "INR", "period": "term"}` | WV |
| kuleuven | living_cost | `{"amount": 1200.0, "currency": "EUR", "period": "month"}` | `{"amount": 1050.0, "currency": "EUR", "period": "month"}` | WV |
| auckland | scholarships | `["Michael Joseph Savage Memorial Award in Music", "Veza Fami` | `["University of Auckland International School Leaver Scholar` | WV |
| edinburgh | scholarships | `["The British Standards Institution Scholarship for Artifici` | `["Edinburgh Global Undergraduate Mathematics Scholarships", ` | WV |
| uiuc | scholarships | `["Native American Scholarship \u2013 Peoria Tribe Scholarshi` | `["Illinois Achievement Scholarship", "Stamps Scholarship", "` | WV |
| uq | scholarships | `["UQ Faculty International Scholarship"]` | `["UQ International Excellence Scholarship", "UQ Internationa` | WV |
