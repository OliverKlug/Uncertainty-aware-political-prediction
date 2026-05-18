# Political Event Forecasting System

A probabilistic forecasting pipeline for discrete political and macroeconomic events. Produces calibrated probability estimates, evaluates them against market-implied baselines, and logs every prediction with a signed, append-only audit trail.

---

## What it does

Political prediction markets have a systematic miscalibration problem: raw probability estimates cluster away from the tails, and models trained across heterogeneous event types produce overconfident, noisy outputs.

This system addresses that by:

1. Anchoring every forecast to a historical base rate, segmented by event category, before incorporating live signals
2. Running separate models for each of 8 event types rather than collapsing them
3. Applying isotonic regression calibration with temporal validation splits to correct for systematic overconfidence
4. Evaluating all predictions against market-implied probabilities as a passive benchmark, out-of-sample

The pipeline runs daily and is designed to be auditable: every prediction maps to its source data, model version, and calibration timestamp.

---

## Pipeline

```
Ingest → Classify → Score → Calibrate → Log
   │          │         │          │        │
Raw event  Route to  Category   Isotonic  Signed
feeds      one of 8  classifier  regression append-only
           models    output      + CI      audit trail
```

**Event categories:**

| Category | Examples |
|----------|---------|
| Supreme Court rulings | Merits decisions, cert grants |
| Federal legislation | Senate/House votes, cloture |
| Executive action | Executive orders, agency rules |
| Elections | Primary, general, special |
| Geopolitical | Treaty ratification, sanctions |
| Central bank | Rate decisions, forward guidance |
| Macro releases | CPI, NFP vs. consensus |
| Regulatory | FTC, SEC, NLRB enforcement outcomes |

---

## What makes it different from a standard classifier

**Separate models per category.** A Supreme Court ruling and a Senate cloture vote have structurally different outcome distributions. Collapsing them produces overconfident, poorly calibrated outputs.

**Base rate anchoring.** Live signals update a prior built from historical outcomes. The prior is computed before any live features are introduced, which prevents signal contamination of the baseline.

**Calibration with temporal integrity.** Isotonic regression is fitted on a held-out calibration split. Validation splits respect chronological order -- no look-ahead. Bootstrap confidence intervals (1,000 resamples) quantify calibration uncertainty. Categories with insufficient data are automatically excluded from the signal layer.

**Signed audit trail.** Every prediction is SHA-256 signed at write time. Append-only. Nothing is editable after logging. This is what makes retrospective evaluation trustworthy.

---

## Results

Evaluated on out-of-sample events where market-implied probabilities were available and liquid at prediction time.

| Metric | Result |
|--------|--------|
| Events evaluated (OOS) | See `results/summary.json` |
| Outperforms market-implied baseline | ~53% of evaluated events |
| Category coverage | Consistent across 6 of 8 categories |
| Expected value vs. market odds | Positive in shadow evaluation |
| Brier Score vs. baseline | Lower in 6 of 8 categories |

> **Important:** All results are from shadow evaluation against market benchmarks. No real capital has been deployed. 53% is the directional win rate; full calibration curves and per-category breakdowns are in `/results/`.

---

## Limitations

- **Thin data categories.** Low-frequency event types (constitutional amendments, impeachments) have insufficient historical data for reliable calibration. These are excluded automatically.
- **US-centric base rate library.** International events are underrepresented in the reference data.
- **Benchmark quality varies.** Prediction market liquidity differs significantly across event types. Sparse markets produce noisier benchmarks.
- **No intraday signals.** Pipeline runs once per day. Doesn't capture last-minute information shocks.
- **Regime sensitivity.** Calibration is fitted on historical distributions. Structural breaks (e.g. court composition changes, new legislative coalitions) can degrade performance in ways that aren't immediately detectable.

---

## Reproducibility

**Requirements:** Python >= 3.10

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

**Run the full daily pipeline:**
```bash
python src/pipeline/run_daily.py
```

**Run backtesting only:**
```bash
python src/backtesting/run_backtest.py --category all --window expanding
```

**Generate calibration diagnostics:**
```bash
python src/evaluation/calibration_report.py --output results/
```

**Run tests:**
```bash
pytest tests/ -v
```

---

## Structure

```
.
├── src/
│   ├── data/           # Ingestion, parsing, base rate library
│   ├── models/         # Per-category classifiers + calibration
│   ├── evaluation/     # Brier, ECE, ROC, calibration curves
│   ├── backtesting/    # Expanding-window OOS engine
│   └── pipeline/       # Daily orchestration
├── docs/
│   ├── model_cards/    # Feature sets and design decisions per category
│   └── methodology.md  # Extended methodology
├── results/            # OOS evaluation outputs
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```
*Not financial advice. Research system only.*
