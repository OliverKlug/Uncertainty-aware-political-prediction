# Model Card – Geopolitical Shifts

**Category ID:** `geopolitical_shift`
**Base rate (historical):** varies widely by sub-type (treaty ratification ~55%; sanctions passage ~40%)

---

## Feature Set

| Feature | Type | Description |
|---------|------|-------------|
| `diplomatic_tension_index` | float [0, 1] | Composite index of diplomatic communications and media tone |
| `trade_volume_change` | float | YoY change in bilateral trade volume (%) |
| `ally_support_count` | int | Number of G7/NATO allies publicly supporting the action |

## Feature Engineering Notes

- `diplomatic_tension_index` aggregates NLP scores from State Department press briefings and UN General Assembly statements.
- `trade_volume_change` uses WTO bilateral trade statistics (6-month lag).

## Training Data

- Source: GDELT, ICEWS, State Department archives, 1995–present
- Restricted to G7-involving events with binary outcome structure

## Known Limitations

- Thinnest category in the base rate library; many event sub-types are near or below the 30-event threshold.
- Geopolitical events are highly non-stationary; historical base rates from the 1990s may be poor priors for post-2022 dynamics.
- `ally_support_count` is zero for unilateral US actions, which are the majority of the dataset.
