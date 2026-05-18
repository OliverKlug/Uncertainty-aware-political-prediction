# Model Card – Supreme Court Rulings

**Category ID:** `supreme_court_ruling`
**Base rate (historical):** ~61% affirm / reverse split (varies by term; see base rate library)

---

## Feature Set

| Feature | Type | Description |
|---------|------|-------------|
| `oral_argument_sentiment` | float [−1, 1] | Aggregate sentiment score from oral argument transcripts (NLP-derived) |
| `precedent_alignment` | float [0, 1] | Fraction of cited precedents favouring petitioner's position |
| `cert_granted_unanimous` | bool | Whether certiorari was granted unanimously |
| `lower_court_direction` | int {−1, 0, 1} | Direction of lower court ruling relative to petitioner |
| `ideological_alignment_score` | float [−1, 1] | Mean ideological score of writing justices (negative = conservative) |

## Feature Engineering Notes

- `oral_argument_sentiment` is computed from the most recent oral argument transcript using a fine-tuned legal-domain BERT model. Only statements by justices are scored (counsel statements excluded).
- `ideological_alignment_score` uses Martin-Quinn scores from the most recent published term.

## Training Data

- Source: Supreme Court Database (Spaeth), 1990–present
- Cases with DIG (dismissed as improvidently granted) are excluded
- Per-term stratification used to avoid term-clustering bias

## Known Limitations

- Sentiment model was trained on a general legal corpus; domain shift on novel constitutional questions is possible.
- Low event count in some sub-categories (e.g., original jurisdiction, per curiam reversals without argument).
- Unanimous cert grants are rare; this feature has high missing-value rate pre-1995.
