# Model Card – Elections

**Category ID:** `election`
**Base rate (historical):** ~75% incumbent re-election rate (House); varies by office

---

## Feature Set

| Feature | Type | Description |
|---------|------|-------------|
| `incumbent_approval_rating` | float [0, 1] | Most recent presidential approval rating (538 aggregate) |
| `generic_ballot_spread` | float | D − R generic ballot spread in percentage points |
| `fundraising_ratio` | float | Candidate fundraising ratio (candidate / opponent) |
| `days_to_election` | int | Days until election day at prediction time |
| `polling_average` | float [0, 1] | FiveThirtyEight polling average for the candidate |

## Feature Engineering Notes

- `days_to_election` is log-transformed; polls converge toward outcomes as this value decreases.
- `fundraising_ratio` is winsorised at the 1st and 99th percentiles.
- Polling average is the most recent published 7-day average; if unavailable (e.g., unpolled special elections), the national generic ballot is substituted.

## Training Data

- Source: MIT Election Lab, FiveThirtyEight historical polls, FEC filings, 1992–present
- Separate models for Presidential, Senate, House, and Gubernatorial races
- Binary outcome: candidate wins / loses

## Known Limitations

- Special elections are underrepresented; model may underweight candidate-quality effects in low-information environments.
- Redistricting cycles can cause structural breaks in district-level base rates.
- Polling sparse for primaries in non-presidential years.
