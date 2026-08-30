# Methodology

What the code does today, not a prospectus.

## Implemented

**Base rates.** `BaseRateLibrary` reads a JSON map `{category: {positive_rate, n_events}}`. Categories with `n_events < 30` or a missing file use prior 0.5. No 1990–present reference library is in this repo.

**Temporal splits.** Expanding-window backtest: train indices are always strictly before test indices. The train block is split in half, earlier half fits the logistic, later half fits isotonic. No shuffle.

**Classifier.** One `sklearn` L2 logistic per category schema (`C=1.0`), with a standard scaler. Feature *names* live in `CATEGORY_FEATURES`. There are no fitted weights in the repo.

**Calibration.** `IsotonicRegression(out_of_bounds="clip")` on the later train half. Bootstrap intervals resample that calibration set. Default 1,000 draws, seed passed in (CLI synthetic path uses seed 0).

**Benjamini–Hochberg.** Step-up: largest `k` with `p_(k) <= (k/n) * FDR`, then reject `1..k`. Independent per-rank compares are wrong here; they can reject a larger p-value and keep a smaller one.

**Metrics.** Brier, ECE (last bin closed so `p=1` counts), ROC-AUC, log-loss. Empty or non-finite inputs raise.

**Audit log.** Append-only JSONL. Each row gets a SHA-256 checksum of the payload. That detects accidental edits. It is not a signature: there is no key, and anyone can rewrite the file and re-hash.

## Not implemented

Live ingest, six of eight parsers, an event store, market prices, a shadow book, or any OOS result vs a prediction market. `run_backtest.py` and `calibration_report.py` only emit numbers if you pass `--synthetic`.

## Model cards

`docs/model_cards/` are intended feature lists. Treat historical base-rate figures there as notes, not as numbers this code computed.
