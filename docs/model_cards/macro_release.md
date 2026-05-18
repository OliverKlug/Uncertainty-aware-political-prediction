# Model Card – Macroeconomic Releases

**Category ID:** `macro_release`
**Base rate (historical):** ~48% of CPI prints beat consensus; ~50% of NFP prints beat consensus

---

## Feature Set

| Feature | Type | Description |
|---------|------|-------------|
| `consensus_estimate` | float | Bloomberg consensus estimate at prediction time |
| `prior_reading` | float | Previous release actual value |
| `surprise_index_3m` | float | Citi Economic Surprise Index (3-month trailing average) |
| `market_positioning_z` | float | Z-score of net speculative positioning in related futures |

## Feature Engineering Notes

- `surprise_index_3m` captures the regime of whether economic data has been systematically surprising to the upside or downside.
- `market_positioning_z` is computed from CFTC Commitments of Traders reports; may have 3-day publication lag.
- Consensus estimate is the median of surveyed economists at T−1 (day before release).

## Training Data

- Source: Bloomberg (consensus), FRED (actuals), CFTC, 2000–present
- G7 macro releases: CPI, PPI, NFP, GDP flash, retail sales, PMI
- Binary outcome: actual ≥ consensus (beat) / actual < consensus (miss)

## Known Limitations

- Revisions are not modelled; the first-release figure is used as the outcome.
- Beat/miss framing collapses ordinal magnitude into binary; large beats and small beats are treated equivalently.
- COVID-era releases (2020Q1–2021Q2) are flagged as outliers and down-weighted in training.
