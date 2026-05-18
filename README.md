# Political & Macroeconomic Event Forecasting Pipeline

A research system for evaluating predictive signals in political and macroeconomic event-based data.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Limitations & Disclaimers](#limitations--disclaimers)

---

## Overview

This pipeline generates calibrated probabilistic forecasts for discrete political and macroeconomic events — judicial rulings, legislative outcomes, elections, and geopolitical shifts. Predictions are anchored to historical base rates before incorporating live signals, and every output is logged with a cryptographically signed, append-only audit trail that maps each forecast to its source data, model version, and calibration timestamp.

The system runs on a daily schedule: ingest → classify → score → generate candidate signals → log.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Daily Pipeline                        │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────┐ │
│  │  Ingest  │───▶│ Classify │───▶│  Score   │───▶│  Log  │ │
│  └──────────┘    └──────────┘    └──────────┘    └───────┘ │
│        │               │               │                    │
│        ▼               ▼               ▼                    │
│   Raw event       Event-type      Calibrated             Signed
│   data feeds      routing         probabilities          audit trail
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Model Layer (per category)               │
│                                                             │
│   Base Rate Library ──▶ Category Classifier ──▶ Calibrator  │
│         │                      │                    │       │
│   Historical         8 separate models        Isotonic      │
│   reference          (one per event type)     regression +  │
│   library                                     bootstrap CI  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Evaluation Layer                         │
│                                                             │
│   Shadow Portfolio ──▶ Backtesting Engine ──▶ Reporting     │
│   (hypothetical         (temporal splits,      (Brier,      │
│    positions)            OOS validation)        ECE, ROC)   │
└─────────────────────────────────────────────────────────────┘
```

**Event categories (8 total):**

| # | Category | Examples |
|---|----------|---------|
| 1 | Supreme Court rulings | Certiorari grants, merits decisions |
| 2 | Federal legislation | Senate/House floor votes, cloture |
| 3 | Executive action | EOs, agency rule publication |
| 4 | Elections | Primary and general, special elections |
| 5 | Geopolitical shifts | Treaty ratification, sanctions |
| 6 | Central bank decisions | Rate decisions, forward guidance |
| 7 | Macroeconomic releases | CPI prints, NFP vs. consensus |
| 8 | Regulatory rulings | NLRB, FTC, SEC enforcement |

---

## Data

### Sources

| Layer | Type | Timeframe | Notes |
|-------|------|-----------|-------|
| Base rate library | Historical outcomes, manually curated | 1990–present | Segmented by event category; primary input for prior construction |
| Live signals | Public filings, legislative calendars, docket updates | Rolling 90-day window | Sourced from government portals and structured news feeds |
| Prediction markets | Market-implied probabilities | Rolling | Used as benchmark only; not used in model training |

### Universe

- US federal political events (primary scope)
- G7 macroeconomic releases
- Select geopolitical events with binary or ordinal outcome structure

### Limitations

- Base rate library is thinner for low-frequency event types (e.g., constitutional amendments, impeachments). Categories with insufficient calibration data are automatically excluded from the signal layer.
- Market benchmark comparisons are limited to events with liquid, verifiable market prices at prediction time.
- No intraday signals. Pipeline runs once per day.

---

## Methodology

### 1. Base Rate Anchoring

Every forecast starts with a historical base rate pulled from the reference library, segmented by event category. This prior is constructed before any live signals are introduced. Collapsing categories into a single model was rejected early — a Supreme Court ruling and a Senate cloture vote have structurally different outcome distributions.

### 2. Signal Generation

Live signals update the prior using a category-specific classifier. Features vary by category (e.g., sponsor count and committee markup status for legislation; oral argument sentiment and precedent alignment for SCOTUS). Feature sets are documented per model in `/docs/model_cards/`.

### 3. Calibration

Raw classifier outputs are systematically overconfident. Calibration uses:

- **Isotonic regression** fitted on a held-out calibration split
- **Temporal validation splits** (no look-ahead leakage; splits respect chronological order)
- **Bootstrap confidence intervals** (1,000 resamples) on calibration curve estimates
- **Multiple-comparison corrections** (Benjamini-Hochberg) when evaluating across categories simultaneously

Categories without sufficient calibration data are flagged and excluded from downstream signal generation automatically.

### 4. Evaluation

Primary metrics:

- **Brier Score** — overall probabilistic accuracy
- **Expected Calibration Error (ECE)** — miscalibration magnitude
- **ROC-AUC** — discrimination ability
- **Log-loss** — penalty for confident wrong predictions

Backtesting uses expanding-window out-of-sample evaluation. No in-sample performance is reported.

### 5. Shadow Portfolio & Audit Trail

Candidate signals are tracked in a shadow portfolio of hypothetical positions. The audit log is append-only and cryptographically signed (SHA-256 per entry). Every prediction record contains:

- Event identifier and category
- Forecast probability and CI bounds
- Model version hash
- Calibration timestamp
- Source data snapshot reference

Nothing is editable post-write.

---

## Results

Evaluated on out-of-sample events with verifiable market benchmarks available at prediction time.

| Metric | Value | Notes |
|--------|-------|-------|
| Events evaluated (OOS) | — | See `/results/summary.json` |
| Brier Score vs. market-implied baseline | Lower in 6 of 8 categories | Excluding two thin-data categories |
| Outperforms market-implied probability | ~53% of evaluated events | Consistent across categories, not concentrated in easy cases |
| Expected value vs. market odds | Positive | Shadow portfolio; no real capital deployed |

> **Note:** These results are from controlled backtesting with out-of-sample evaluation against market-implied probabilities as a passive benchmark. Past performance in a research setting does not imply future predictive accuracy. No real positions have been taken.

Full result tables, calibration curves, and per-category breakdowns are in `/results/`.

---

## Reproducibility

### Requirements

```
python >= 3.10
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Environment Setup

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # fill in any required API keys
```

### Run the Pipeline

```bash
# Full daily pipeline
python src/pipeline/run_daily.py

# Backtesting only
python src/backtesting/run_backtest.py --category all --window expanding

# Calibration diagnostics
python src/evaluation/calibration_report.py --output results/
```

### Project Structure

```
.
├── src/
│   ├── data/               # Ingestion, parsing, base rate library
│   ├── models/             # Category-specific classifiers + calibration
│   ├── evaluation/         # Brier, ECE, ROC, calibration curves
│   ├── backtesting/        # Expanding-window OOS backtesting engine
│   └── pipeline/           # Daily orchestration
├── docs/
│   ├── model_cards/        # Per-category feature sets and decisions
│   └── methodology.md      # Extended methodology notes
├── results/                # OOS evaluation outputs (read-only artifacts)
├── tests/                  # Unit and integration tests
├── .env.example
├── requirements.txt
└── README.md
```

### Running Tests

```bash
pytest tests/ -v
```

---

## Limitations & Disclaimers

- **Research only.** This system has not been used to trade real capital. The shadow portfolio is a research instrument.
- **No investment advice.** Nothing here constitutes financial, legal, or investment advice.
- **Base rate gaps.** Low-frequency event categories have thin historical data and are excluded from signal generation by design.
- **Benchmark quality.** Prediction market prices are used as a benchmark. Their quality varies by event and market liquidity.
- **US-centric.** The base rate library is primarily built on US federal events. International event categories are less mature.
