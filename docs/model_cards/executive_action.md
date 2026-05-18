# Model Card – Executive Action

**Category ID:** `executive_action`
**Base rate (historical):** ~78% of signed EOs survive first-year legal challenge intact

---

## Feature Set

| Feature | Type | Description |
|---------|------|-------------|
| `prior_eo_count_term` | int | Number of EOs signed so far in the current presidential term |
| `congress_opposition_index` | float [0, 1] | Share of Congress members publicly opposing the action |
| `legal_challenge_likelihood` | float [0, 1] | Estimated probability of immediate legal challenge (derived from prior EO patterns) |

## Feature Engineering Notes

- `legal_challenge_likelihood` is derived from a separate logistic model trained on historical EO challenge data (DOJ Office of Legal Counsel records).
- `congress_opposition_index` is constructed from roll-call votes and floor statements within 30 days of the EO's signing date.

## Training Data

- Source: Federal Register (EO text + dates), Federal court dockets (challenge outcomes), 1993–present
- Binary outcome: EO remains in effect / enjoined or rescinded within 12 months

## Known Limitations

- Low N for injunction outcomes; bootstrap CI intervals are wide for this category.
- Does not distinguish between full rescission, partial injunction, and stay pending appeal.
- Presidential term effects are partially captured but not explicitly modelled.
