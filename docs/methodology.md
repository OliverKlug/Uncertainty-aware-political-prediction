# Extended Methodology Notes

This document supplements the README with deeper technical rationale and implementation details for each stage of the forecasting pipeline.

---

## 1. Base Rate Construction

Base rates are computed from the curated reference library spanning 1990–present. Each event category is treated independently: pooling categories (e.g., averaging SCOTUS ruling rates with CPI-miss rates) would destroy the structural differences in outcome distributions.

For a given category **C** with **N** historical events and **k** positive outcomes:

```
base_rate(C) = k / N
```

Categories with N < 30 are excluded from signal generation and flagged in the output. The threshold of 30 was chosen as a conservative minimum for isotonic regression to produce stable estimates.

---

## 2. Temporal Splitting Protocol

All data splits respect chronological order to prevent look-ahead leakage:

- **Training split**: events up to a cutoff date, used to fit classifier weights.
- **Calibration split**: the next chronological block, used exclusively to fit the isotonic calibrator.
- **Test split**: the final OOS block, used only for evaluation and never touched during model fitting.

No random shuffling is applied at any stage.

---

## 3. Classifier Architecture

Each of the 8 event categories has a dedicated logistic regression classifier (L2 regularisation, C=1.0) with a category-specific feature set. Logistic regression was chosen over more complex models (gradient boosting, neural networks) because:

1. **Interpretability**: coefficients are directly readable as log-odds.
2. **Sample efficiency**: political event datasets are small; high-variance models overfit.
3. **Probabilistic output**: native probability calibration is better-behaved than post-hoc temperature scaling on complex models.

Feature engineering per category is documented in `/docs/model_cards/`.

---

## 4. Calibration Procedure

Raw classifier outputs are systematically overconfident (a well-documented property of logistic regression on imbalanced or sparse political event data). Calibration proceeds as follows:

1. **Isotonic regression** is fitted on the calibration split. Isotonic regression is a non-parametric monotone transform; it is preferred over Platt scaling (logistic) because political event probability distributions are not constrained to sigmoid shape.
2. **Bootstrap CI** (1,000 resamples, seed=42) produces 95% confidence intervals around each calibrated estimate. Resampling is done with replacement from the calibration set; the fitted isotonic curve is re-estimated on each resample.
3. **Benjamini-Hochberg correction** is applied when evaluating miscalibration across multiple categories simultaneously, controlling FDR at 5%.

---

## 5. Evaluation Protocol

All reported metrics are computed on **out-of-sample** events only. No in-sample performance is reported anywhere in `/results/`.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Brier Score | mean((p − o)²) | Overall probabilistic accuracy; lower = better |
| ECE | Σ_b (n_b/N) \|acc_b − conf_b\| | Miscalibration magnitude; lower = better |
| ROC-AUC | Area under ROC curve | Discrimination; 0.5 = random, 1.0 = perfect |
| Log-loss | −mean(o·log p + (1−o)·log(1−p)) | Penalises confident errors; lower = better |

Backtesting uses an **expanding window**: the training set grows by one event per fold, with no overlap between calibration and test periods.

---

## 6. Audit Trail

Every prediction record is written to an append-only JSONL file. Each entry is SHA-256 signed over its content (excluding the signature field itself) to detect tampering. The audit trail is read-only after write; no update or delete operations are implemented.

Fields logged per prediction:

- `event_id`, `category`, `scheduled_date`
- `calibrated_prob`, `ci_lower`, `ci_upper`
- `prior` (base rate anchor)
- `model_version` (hash of model artefact)
- `calibration_timestamp`
- `_sha256` (entry signature)

---

## 7. Shadow Portfolio

Candidate signals are tracked as hypothetical binary positions. No real capital is deployed. Expected-value calculations use market-implied probabilities as the passive benchmark:

```
EV = p_model × payout_if_correct − (1 − p_model) × stake
```

where `payout_if_correct` and `stake` are derived from the market price at prediction time. The shadow portfolio serves purely as a research instrument to assess whether model probabilities diverge systematically from market-implied prices.
