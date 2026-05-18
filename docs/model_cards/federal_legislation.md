# Model Card – Federal Legislation

**Category ID:** `federal_legislation`
**Base rate (historical):** ~12% of introduced bills pass (floor vote subset: ~68%)

---

## Feature Set

| Feature | Type | Description |
|---------|------|-------------|
| `sponsor_count` | int | Number of co-sponsors at prediction time |
| `committee_markup` | bool | Whether the bill has passed committee markup |
| `bipartisan` | bool | Whether at least one co-sponsor is from the opposing party |
| `days_since_introduction` | int | Calendar days since the bill was introduced |
| `chamber_majority_margin` | int | Seat margin of the majority party in the relevant chamber |

## Feature Engineering Notes

- `sponsor_count` is capped at 50 to reduce right-skew (bills with >50 co-sponsors are functionally equivalent for prediction purposes).
- `days_since_introduction` is log-transformed before model input.

## Training Data

- Source: Congress.gov API, GovTrack, 103rd–118th Congress
- Restricted to bills that reached a floor vote (subset model); separate model for floor-vote likelihood pending.
- Senate and House are treated as separate observations.

## Known Limitations

- Procedural votes (cloture, rule adoption) have different dynamics from substantive legislation but share the same feature schema pending category refinement.
- Majority margin is a lagged feature (updated after each special election); may be stale mid-cycle.
