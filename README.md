# algotrader-research

A research-grade, leakage-aware pipeline for testing ML trading signals on daily bars:
`yfinance -> TA-Lib features -> XGBoost (purged walk-forward) -> backtesting.py -> metrics vs buy-and-hold -> Streamlit`.

The goal of Phase 1 is an **honest, reproducible** answer to "does this signal survive costs out of sample?", not a strategy that beats the market.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                                        # 6 tests, ~2s, no network needed
python -m algotrader.cli --source synthetic --tickers SYN  # plumbing check on fake data
python -m algotrader.cli --config configs/baseline.yaml    # real run (needs Yahoo access)
streamlit run dashboard/app.py                             # view runs/
```

Docker: `docker build -t algotrader . && docker run --rm -v $PWD/runs:/app/runs algotrader`

## Layout

```
configs/baseline.yaml        tickers, dates, features, model, CV, costs, thresholds, seed
algotrader/
  config.py                  load/validate config, config hash, seeding
  data/ingest.py             yfinance fetch, sanitize, parquet cache, synthetic source
  features/technical.py      TA-Lib features (scale-free) + execution-aligned target
  models/xgb_model.py        model factory (any fit/predict_proba model plugs in; LSTM later)
  models/walkforward.py      purged expanding-window walk-forward, OOS predictions only
  backtest/strategy.py       backtesting.py strategy + runner (commission + slippage)
  evaluation/metrics.py      Sharpe, max drawdown, cumulative return, CAGR, win rate, buy-and-hold
  pipeline.py, cli.py        one ticker end to end; run folder with config, env.json, artifacts
dashboard/app.py             read-only Streamlit/Plotly view over runs/
tests/test_pipeline.py       causality, label timing, purge gap, sanitize, metrics, end-to-end
```

Each run writes `runs/<timestamp>_<confighash>/` containing `config.yaml`, `env.json` (seed, git commit, package versions), `summary.csv`, and per ticker: equity, trades, OOS predictions, fold diagnostics, feature importance, metrics.

## Design decisions that protect the results

| Risk | Control |
|---|---|
| Lookahead in features | All TA-Lib features are backward-looking; a test checks that truncating history leaves earlier rows unchanged |
| Label/execution mismatch | Target = sign of `open[t+1+h] / open[t+1] - 1`: signal at close `t`, fill at next open, matching backtesting.py's default fill timing |
| Overlapping labels across the train/test boundary | `TimeSeriesSplit(gap = horizon + 1)`, asserted in tests |
| In-sample backtest | The backtest runs only on stitched out-of-sample predictions (contiguous test blocks) |
| Non-stationary inputs | Features are ratios or bounded oscillators, no raw price levels |
| Flattering benchmark | Buy-and-hold pays one entry cost and is scored with the same metric code as the strategy |
| Split/dividend artifacts | `auto_adjust=True` adjusts O/H/L/C consistently; end date is pinned and part of the cache key |
| Unreproducible runs | Seed, config hash, versions and git commit are stored with every run |

Cost model: `commission_bps` is a relative commission on every order and `slippage_bps` is passed as backtesting.py's `spread`, applied to every fill. Total cost per side is their sum (10 bps by default, so a round trip is about 20 bps).

## Verification status

Tested here on **synthetic data only** (the build sandbox cannot reach Yahoo): the unit tests, the full CLI run and a headless Streamlit render all pass. The yfinance fetch path is written against its documented API but has **not** been run against live data, so do a first run with `--tickers SPY` and check `data_cache/` before trusting anything.

## Known limitations of the Phase 1 data and setup

- yfinance is an unofficial scraper: fine for research, not for production. Adjusted history is revised retroactively, which is why caching and a pinned end date matter.
- Single-ticker runs on survivors (e.g. SPY) carry survivorship bias for individual stocks.
- Entry/exit thresholds (`enter_prob`, `exit_prob`) are hyperparameters. Do not tune them on the same OOS window you report; see Phase 2 holdout.
- Daily direction on liquid ETFs is close to a coin flip. A realistic Phase 1 outcome is "AUC slightly above 0.5, Sharpe below buy-and-hold after costs". That is a valid result.

## Roadmap

**Phase 1 (2-3 weeks)**: scaffold above, then: (1) first live run on SPY/QQQ, (2) shuffled-label control (a model trained on permuted labels must show no edge; if it does, there is a leak), (3) logistic-regression and simple momentum baselines, (4) cost sensitivity sweep (0, 5, 10, 20 bps).

**Phase 2 (4-6 weeks)**: PyTorch LSTM behind the same `fit/predict_proba` interface; macro features (VIX, rates, via additional sources, lagged to avoid publication-time leakage); volatility regimes; risk layer (volatility-targeted sizing, stop-loss, drawdown kill-switch); rolling-window walk-forward; a locked final holdout touched once; parameter-sensitivity heatmaps in the dashboard.

**Phase 3 (ongoing)**: Optuna sweeps scored on inner-fold metrics only, SHAP, MLflow (or equivalent) for artifacts and model registry, containerized scheduled runs, deflated Sharpe / multiple-testing correction across all configs tried.

## Defaults chosen (change in `configs/baseline.yaml`)

Indicators: lagged returns (1/5/10/20), SMA 10/50, EMA 12/26, RSI 14, MACD 12/26/9, Bollinger 20/2, ATR 14, 20-day realized vol and volume ratio. Horizon: 1 day. Model: shallow, heavily regularized XGBoost. Splits: 5 expanding folds. Costs: 5 bps commission + 5 bps slippage per side.
