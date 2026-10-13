# Quantitative-Trading-System


**A leakage-aware machine-learning research framework for testing daily equity trading signals.**

This project combines **TA-Lib technical indicators**, **XGBoost classification**, **purged walk-forward validation**, and **cost-aware backtesting** to study whether a directional signal remains useful out of sample. An extended second phase benchmarks alternative models, tests market-regime and macroeconomic inputs, explores risk controls, and evaluates the selected approach on a locked holdout period.

> **Research project, not a live trading system.** The objective is rigorous experimentation—not a claim of profitable performance. Backtested results do not guarantee future returns.

## Highlights

- **Multiple data sources:** Tiingo (default), Yahoo Finance via `yfinance`, user-supplied OHLCV CSV, or synthetic data for offline smoke tests.
- **Technical features:** lagged returns, SMA/EMA ratios, RSI, MACD, Bollinger Bands, ATR, realized volatility, and relative volume.
- **Time-aware evaluation:** expanding-window `TimeSeriesSplit` with a purge gap between training and test observations.
- **Executable backtests:** next-bar execution logic, commission and modeled slippage, and comparison against buy-and-hold.
- **Broader model benchmarking:** baseline and alternative models, including an optional LSTM, evaluated on the development period.
- **Robustness experiments:** alternative prediction horizons, transaction-cost sweeps, rolling analysis, and risk-management variants.
- **Explainability and tuning:** optional SHAP feature attribution and Optuna hyperparameter search on inner folds.
- **Final holdout:** a roughly 12-month period reserved for one-shot evaluation after model and risk-rule selection.

## Research Workflow

```text
Daily OHLCV prices
       |
       v
Cleaning and local CSV caching
       |
       v
TA-Lib features + optional regime/macro features
       |
       v
Forward-return classification target
       |
       v
Purged walk-forward model evaluation
       |
       v
Predicted probabilities -> trading decisions
       |
       v
Cost-aware backtest + buy-and-hold comparison
       |
       v
Development-only comparisons, tuning, risk controls
       |
       v
Locked holdout evaluation
```

The signal is computed from information available through the close of day *t*. The notebook constructs labels from subsequent opening prices; the baseline horizon is **one trading day**. A purge gap of `horizon + 1` observations separates training and test periods to reduce label overlap.

## Repository Contents

```text
.
├── algo_trading_research (1).ipynb   # Complete research workflow (Phases 1 and 2)
├── README.md                         # Project documentation
└── data/                             # Created by the notebook for cached price CSVs
```

> The notebook is currently the source of truth. This repository does not include a production trading API, broker integration, standalone Python package, or precomputed performance report.

## Getting Started

### 1. Requirements

- Python 3 and a Jupyter-compatible environment (JupyterLab, Jupyter Notebook, or Google Colab).
- Internet access to download real market prices; macroeconomic features may also request FRED data.
- A **Tiingo API token** if using the default data source.

The notebook contains its own installation cells. For a local environment, install the equivalent dependencies:

```bash
python -m pip install numpy pandas requests scikit-learn xgboost TA-Lib backtesting yfinance plotly torch shap optuna jupyter
```

`torch`, `shap`, and `optuna` support the extended Phase 2 experiments. Installation of TA-Lib can depend on your operating system and Python version; consult the [TA-Lib Python wrapper documentation](https://github.com/TA-Lib/ta-lib-python) if installation fails.

### 2. Launch the notebook

```bash
jupyter notebook "algo_trading_research (1).ipynb"
```

Or upload the notebook to Google Colab and run its cells in order.

### 3. Configure market data

Edit the `CFG` dictionary in **Section 1: Configuration**. Its default dataset is **SPY daily prices from 2010-01-01 through 2024-12-31**, downloaded from Tiingo using adjusted OHLCV fields. Downloaded data is cached in `data/` for subsequent runs.

For Tiingo, set an environment variable before launching Jupyter:

```bash
export TIINGO_API_KEY="your_token_here"    # macOS / Linux
```

On Windows PowerShell:

```powershell
$env:TIINGO_API_KEY = "your_token_here"
```

Alternatively, provide the token through a Colab Secret named `TIINGO_API_KEY` or enter it at the secure prompt when the notebook requests it. **Do not commit API keys.**

Other supported modes can be configured in `CFG["data"]`:

| `source` | Data source | Notes |
|---|---|---|
| `"tiingo"` | Tiingo EOD API | Default; requires token |
| `"yfinance"` | Yahoo Finance | No token; unofficial data interface |
| `"csv"` | Local CSV | Must provide a valid `csv_path` |
| `"synthetic"` | Generated prices | Pipeline testing only; **not evidence of a trading edge** |

For custom CSV input, supply date-indexed data with `Date, Open, High, Low, Close, Volume`. An `Adj Close` column is supported by the notebook's CSV ingestion path.

### 4. Run the analysis

Execute notebook cells **top to bottom**. Phase 1 runs the baseline pipeline and diagnostics. Phase 2 adds further experiments and ends with its reserved holdout evaluation. **Do not repeatedly inspect or tune against the holdout**: doing so defeats its purpose. The notebook has a session-level guard against repeat holdout evaluation.

## Notebook Guide

| Sections | Purpose |
|---|---|
| **1–3** | Configuration, price ingestion, technical features, and forward targets |
| **4–7** | Purged XGBoost walk-forward evaluation, backtesting, performance metrics, and leakage checks |
| **8** | Interpretation of the baseline results |
| **9–10** | Phase 2 settings, market-regime features, FRED series, and development/holdout split |
| **11–12** | Alternative predictive models and development-set comparisons |
| **13–14** | Horizon comparison and sensitivity to trading costs |
| **15** | Volatility-based position sizing, stop-loss, and drawdown controls |
| **16–17** | Optuna tuning and SHAP feature analysis |
| **18** | Sanity checks and final one-shot holdout backtest |
| **19** | Interpretation, limitations, and possible future work |

### Key configuration values

| Setting | Notebook default | Meaning |
|---|---|---|
| `CFG["seed"]` | `42` | Random seed |
| `CFG["data"]["ticker"]` | `SPY` | Instrument |
| `CFG["target"]["horizon"]` | `1` | Forward holding horizon in trading days |
| `CFG["model"]["cv"]["n_splits"]` | `5` | Baseline walk-forward folds |
| `CFG["backtest"]["cash"]` | `100000` | Starting backtest capital |
| `CFG["backtest"]["commission_bps"]` | `5` | Commission per side, in basis points |
| `CFG["backtest"]["slippage_bps"]` | `5` | Modeled slippage per side, in basis points |
| `CFG["backtest"]["enter_prob"]` | `0.52` | Long-entry probability threshold |
| `CFG["backtest"]["exit_prob"]` | `0.50` | Exit threshold |
| `P2["holdout_months"]` | `12` | Approximate locked holdout length |
| `P2["macro"]["series"]` | `VIXCLS`, `DGS10` | Optional FRED indicators |
| `P2["macro"]["lag"]` | `2` | Lag applied to macro inputs, in trading days |
| `P2["optuna"]["n_trials"]` | `20` | Tuning budget |

For additional hyperparameters and risk limits, consult `CFG` and `P2` directly in the notebook.

## Modeling and Validation Details

**Target.** For day *t*, the binary target represents the sign of the return between the next session's opening price and an opening price `horizon` sessions later:

```text
forward_return[t] = Open[t + 1 + horizon] / Open[t + 1] - 1
target[t]         = 1 if forward_return[t] > 0 else 0
```

**Feature timing.** Technical features are built using historical observations through day *t*. The macroeconomic series are forward-filled to trading dates and then shifted by a configurable lag to reduce timing leakage. This is a practical approximation, **not** a point-in-time historical release database.

**Validation.** The baseline uses an expanding-window time-series split with a label-purging gap. Phase 2 separates development observations from a final holdout and conducts comparisons and tuning without using the holdout for parameter selection.

**Trading simulation.** The notebook tests long-side decision rules with explicit commission and modeled slippage assumptions. It compares the strategy's results with buy-and-hold and evaluates how sensitive conclusions are to increasing costs. Simulated execution is necessarily simpler than real-world fills, spreads, liquidity constraints, and market impact.

**Reproducibility.** A fixed seed, pinned historical end date, configuration blocks, and cached input prices support more repeatable experiments. Different library versions, providers, and available data can still produce differing results.

## How to Interpret Results

Look beyond classification accuracy. The notebook examines predictive discrimination and the economic effects of acting on predictions, including backtest metrics and buy-and-hold comparisons. In particular:

1. Compare each complex model against simpler alternatives, not just against a zero-skill assumption.
2. Check performance **across walk-forward folds**, rather than highlighting a best fold.
3. Evaluate whether the apparent edge survives realistic transaction costs.
4. Separate hyperparameter-selection scores from subsequent out-of-sample performance.
5. Treat a strong result on a roughly one-year holdout cautiously; it remains a small sample and cannot establish future profitability.

**No performance figures are asserted here** because outcomes depend on actually executing the notebook with the chosen data, environment, and configuration.

## Limitations

- Research focuses on a daily-price workflow, with **SPY as the default instrument**, rather than a diversified live portfolio.
- Historical backtests are subject to regime shifts, data-provider differences, and execution-model assumptions.
- The notebook's simulated slippage is not a full market-microstructure or fill-probability model.
- The FRED lag is not a guarantee of historically correct as-of availability or vintage data.
- Testing many alternatives on the same development set introduces selection bias; even the winning development model may be overfit.
- A single reserved holdout is valuable but insufficient for proving a persistent edge.
- The notebook does **not** place real trades or implement production monitoring, broker authentication, or live risk enforcement.

## Future Directions

- Multi-asset and pooled-model experiments.
- Point-in-time macro data and more realistic execution simulations.
- Multiple-testing corrections, such as deflated Sharpe ratio approaches.
- Experiment tracking with MLflow and environment/version locking.
- Scheduled research jobs and separate production-grade execution infrastructure, if appropriate.

## Data and Tools

Key libraries include [XGBoost](https://xgboost.readthedocs.io/), [TA-Lib](https://ta-lib.org/), [backtesting.py](https://kernc.github.io/backtesting.py/), [scikit-learn](https://scikit-learn.org/), [Plotly](https://plotly.com/python/), [PyTorch](https://pytorch.org/), [Optuna](https://optuna.org/), and [SHAP](https://shap.readthedocs.io/). Market data options include [Tiingo](https://www.tiingo.com/), [Yahoo Finance](https://finance.yahoo.com/), and [FRED](https://fred.stlouisfed.org/).

## Disclaimer

**For educational and research purposes only.** This repository does not provide financial advice, a recommendation to buy or sell securities, or assurance of trading profitability. Users are responsible for validating data, models, costs, and risks before making any financial decisions.

## License

No license is specified in the provided notebook. Add a `LICENSE` file if you intend to grant others permission to reuse or redistribute this project.
