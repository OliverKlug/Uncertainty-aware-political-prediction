# Model Card – Central Bank Decisions

**Category ID:** `central_bank_decision`
**Base rate (historical):** ~38% probability of a rate change at any given FOMC meeting

---

## Feature Set

| Feature | Type | Description |
|---------|------|-------------|
| `current_rate` | float | Current federal funds target rate (%) |
| `inflation_delta_vs_target` | float | CPI YoY minus 2% target (percentage points) |
| `unemployment_delta` | float | Unemployment rate minus NAIRU estimate (percentage points) |
| `market_implied_change` | float | Fed funds futures implied rate change (%) |
| `forward_guidance_score` | float [−1, 1] | NLP score of most recent FOMC statement (hawkish/dovish) |

## Feature Engineering Notes

- `market_implied_change` is computed from the nearest-maturity Fed funds futures contract, adjusted for calendar-day effects.
- `forward_guidance_score` uses a dictionary-based approach with financial domain sentiment lexicons (Loughran-McDonald).

## Training Data

- Source: FRED (macro data), CME FedWatch (market prices), FOMC statements and minutes, 1994–present
- Binary outcome: rate change (either direction) / hold

## Known Limitations

- The ZLB (zero lower bound) period 2009–2015 is structurally different; model performance degrades for rates near zero.
- Emergency inter-meeting cuts (2020, 2008) are excluded from training as non-representative.
- Forward guidance text has changed in style substantially across Fed chairs; NLP scores may have systematic drift.
