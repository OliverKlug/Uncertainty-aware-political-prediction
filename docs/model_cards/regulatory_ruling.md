# Model Card – Regulatory Rulings

**Category ID:** `regulatory_ruling`
**Base rate (historical):** ~61% enforcement action rate across NLRB, FTC, SEC (complaint-stage)

---

## Feature Set

| Feature | Type | Description |
|---------|------|-------------|
| `agency_prior_enforcement_rate` | float [0, 1] | Rolling 3-year enforcement rate for the relevant agency |
| `complaint_severity_score` | float [0, 1] | Composite score derived from complaint text and alleged harm magnitude |
| `political_appointee_alignment` | float [−1, 1] | Ideological alignment of current agency leadership with the administration |

## Feature Engineering Notes

- `agency_prior_enforcement_rate` is agency-specific and updated quarterly.
- `complaint_severity_score` is produced by a fine-tuned legal NLP model trained on resolved cases.
- `political_appointee_alignment` uses DW-NOMINATE-style scores estimated from agency rulemaking and enforcement patterns.

## Training Data

- Source: Agency administrative records (NLRB, FTC, SEC, EPA, CFPB), 1993–present
- Binary outcome: enforcement action proceeds / dismissed or settled pre-enforcement

## Known Limitations

- Agencies differ substantially in transparency; NLRB records are more complete than, e.g., NRC.
- Settlement outcomes (where enforcement technically proceeds but with minimal consequence) are coded as enforcement successes, which may overstate true enforcement rates.
- Political appointee alignment scores lag appointment dates by ~6 months as behavioural patterns must accumulate.
