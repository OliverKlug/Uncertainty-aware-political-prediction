# Uncertainty-aware political prediction

Research scaffold for discrete political and macro event probabilities: one logistic model per category, isotonic calibration on a later time split, expanding-window backtest.

This is not a live forecasting service. Ingest does not call Congress.gov, CourtListener, FRED, or any news API. There is no event store, no fitted production models, and no market-implied evaluation. `results/summary.json` is empty on purpose.

## What actually runs

- **Parsers** for two toy JSON shapes: `congress_calendar`, `scotus_docket`
- **Base-rate lookup** from a JSON file if you provide one; missing or thin categories fall back to 0.5
- **CategoryClassifier**: L2 logistic regression + standardisation
- **IsotonicCalibrator** with bootstrap intervals on the calibration curve
- **Metrics**: Brier, ECE (last bin includes p = 1), ROC-AUC, log-loss
- **BacktestEngine**: expanding window, chronological fit / calibrate / test, no shuffle
- **Daily CLI**: writes empty staging JSON, scores the historical prior only, appends a SHA-256 *checksum* (not a cryptographic signature) to `results/audit_log.jsonl`

## Pipeline (implemented vs not)

```
Ingest          Classify              Score                 Log
empty stubs     2 parsers             prior only            JSONL + checksum
no HTTP         6 sources unimplemented   no fitted models
```

**Event category schemas** (feature lists only; no trained weights):

| Category | Examples |
|----------|----------|
| Supreme Court rulings | Merits decisions, cert grants |
| Federal legislation | Senate/House votes, cloture |
| Executive action | Executive orders, agency rules |
| Elections | Primary, general, special |
| Geopolitical | Treaty ratification, sanctions |
| Central bank | Rate decisions, forward guidance |
| Macro releases | CPI, NFP vs. consensus |
| Regulatory | FTC, SEC, NLRB enforcement outcomes |

## Setup

Python >= 3.10.

```bash
git clone https://github.com/OliverKlug/Uncertainty-aware-political-prediction.git
cd Uncertainty-aware-political-prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python src/pipeline/run_daily.py
python src/backtesting/run_backtest.py --category election --synthetic
python src/evaluation/calibration_report.py --output results/ --synthetic
pytest tests/ -v
```

`--synthetic` is required for backtest and calibration report. It uses an explicit RNG and labels the JSON `data: synthetic`. Do not treat those files as OOS market results.

## Layout

```
src/data/           ingest stubs, parsers, base-rate library
src/models/         per-category logistic + isotonic calibration
src/evaluation/     Brier, ECE, ROC, calibration report
src/backtesting/    expanding-window engine
src/pipeline/       daily orchestration (prior-only scores)
docs/               methodology notes + intended feature cards
tests/
```

Not financial advice. Research code only.
