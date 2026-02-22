# S&P 500 Next-Day Direction Predictor

Predicts whether the S&P 500 will close higher or lower tomorrow using a gradient-boosted classifier trained on cross-asset features derived entirely from public data. Originally inspired by an AmplifyME quantitative finance competition (ROC-AUC 0.61, top ranked), then rebuilt from scratch with free data sources, a fixed-window evaluation framework, and beam-search feature selection.


## Quick Start

```bash
pip install -r requirements.txt

python main.py                    # Full pipeline: beam search → evaluation → plots
python main.py --skip-selection   # Skip feature selection (faster, uses all 22 features)
python main.py --skip-eda         # Skip exploratory plots
python main.py --no-walkforward   # Skip secondary walk-forward check
python main.py --refresh-data     # Force re-download all market data
```

First run downloads ~10 years of daily data from yfinance and FRED (takes 1–2 minutes). Subsequent runs use cached CSVs in `data/`.


## Project Structure

```
sp500-predictor/
├── config.py          Dates, tickers, FRED series IDs, hyperparameters
├── data_loader.py     yfinance + FRED download, caching, alignment
├── features.py        Stationary feature engineering (22 active features)
├── model.py           HistGradientBoosting, fixed-window eval, beam search
├── visualize.py       10 diagnostic plots (EDA + model evaluation)
├── main.py            CLI entry point
├── requirements.txt   Dependencies (pure scikit-learn, no LightGBM)
├── data/              Cached CSVs (auto-created, safe to delete)
└── output/            Plots and CSV results (auto-created)
```


## Methodology

### Data

All data is freely available — no proprietary feeds or API keys required.

| Source | Series | Frequency |
|--------|--------|-----------|
| yfinance | S&P 500 OHLCV, VIX (OHLC), DXY, EUR/USD, USD/JPY, WTI crude, Brent crude, gold, copper, natural gas, SOX semiconductor index | Daily |
| FRED | Treasury yields (1Y, 5Y, 10Y, 20Y, 30Y), effective fed funds rate, Fed balance sheet, CPI, unemployment, consumer confidence, export/import prices, corporate profits | Daily to quarterly |

Data starts from 2015-01-01 (configured in `config.py`). All series are aligned to a common business-day index and forward-filled — macro data that publishes monthly stays constant at its last known value until the next release, which is financially correct.

The data loader splits everything into three DataFrames that mirror the original AmplifyME competition structure: `price_df` (S&P 500 OHLCV), `add_df` (VIX, DXY, commodities, currencies, yields, SOX), and `macro_df` (CPI, unemployment, confidence, Fed balance sheet, etc.).


### Feature Engineering

Every feature is a stationary transformation — returns, z-scores, ratios, or diffs. Raw price levels are never used as features because they trend upward over time and would make the model memorize price ranges rather than learn predictive patterns.

The 22 currently active features fall into eight categories:

**S&P 500 price action** — 1-day return, 21-day realized volatility (annualized), distance from 50-day SMA, SMA crossovers (5/21 and 50/200), intraday body (close vs open as a fraction of close), volume ratio (today vs 21-day average), overnight gap (open vs prior close), close location value within the day's range (CLV: +1 near high, −1 near low).

**VIX & volatility** — 1-day VIX change, VIX distance from 21-day SMA, VIX 63-day z-score, volatility risk premium in two forms (VIX minus annualized realized vol, and the same in percentage-point units), and a panic reversal score that multiplies the magnitude of a down day by the magnitude of a VIX spike on the same day — this captures capitulation events where mean-reversion the next day is historically likely.

**Yield curve** — 1Y–10Y spread (level and 21-day change) and 21-day change in the fed funds rate.

**Dollar strength** — 21-day DXY return and 63-day DXY z-score.

**Macro** — Fed balance sheet 63-day growth rate (liquidity proxy).

**Cross-asset regime** — 21-day rolling correlation between S&P 500 returns and VIX changes.



### Target

Binary classification: 1 if tomorrow's close is higher than today's close, 0 otherwise. The dataset has a slight positive skew (~52% up days).


### Evaluation: Fixed 8-Year / 2-Year Split

The primary evaluation uses a single fixed time-based split rather than rolling walk-forward cross-validation:

| Window | Dates | Purpose |
|--------|-------|---------|
| Train | 2015-01-01 → 2022-12-31 | ~2,000 trading days |
| Test | 2023-01-01 → 2024-12-31 | ~500 trading days |

This same fixed split is used both for final evaluation and as the scoring function inside beam-search feature selection. 

The `evaluate_fixed_year_windows` function enforces that `start_date + train_years + test_years == end_date`, so the split is always internally consistent.

A ratio-based holdout (`evaluate_holdout`, 80/20 split) and a rolling-window walk-forward (`evaluate_walkforward`) are still available in `model.py` as secondary tools, but neither drives the main pipeline.


### Feature Selection: Beam-Search Backward Elimination

Standard greedy backward elimination commits to one drop order — once a feature is removed in round 1, that decision is permanent even if it blocks a better subset later. With correlated features (e.g., VIX change and VIX z-score share information), the globally optimal subset often requires a different early drop that looks slightly worse in isolation.

Beam search solves this by maintaining multiple candidate paths:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beam_width` | 10 | Number of surviving feature subsets per round |
| `children_per_state` | 3 | Each survivor tries dropping its 3 worst features |
| `scorer` | `fixed_8y2y` | Each subset scored on the fixed 8Y/2Y AUC |
| `min_features` | 5 | Floor — never drops below 5 features |
| `patience_rounds` | 2 | Stops after 2 rounds without global AUC improvement |
| `threshold` | 0.001 | Minimum AUC gain to count as improvement |

Each round, every surviving subset generates children by trying single-feature drops. Children are ranked by AUC, and only the top `beam_width` subsets advance. A memoized cache ensures that identical feature subsets (reached via different drop orders) are only evaluated once. The search typically evaluates hundreds of unique subsets.

The greedy `backward_elimination` function is also still available for faster, lower-quality selection when iterating quickly.


### Model

**scikit-learn `HistGradientBoostingClassifier`** — histogram-based gradient boosting in the same algorithm family as LightGBM, but ships with scikit-learn (no separate install). Handles NaN natively via learned missing-value splits.

| Hyperparameter | Value | Why |
|----------------|-------|-----|
| `max_iter` | 600 | Upper bound; early stopping selects the actual count |
| `learning_rate` | 0.05 | Moderate: slow enough to generalize, fast enough to converge |
| `max_depth` | 6 | Allows two- and three-way feature interactions |
| `max_leaf_nodes` | 31 | Matches LightGBM's default; sufficient capacity for weak signal |
| `min_samples_leaf` | 20 | Fine enough partitions without memorizing daily noise |
| `l2_regularization` | 0.3 | Moderate weight shrinkage |
| `max_features` | 0.8 | Column subsampling per tree to reduce overfitting |
| `early_stopping` | True | 10% validation fraction; stops after 50 rounds with no improvement |

Data preparation before training: rows where > 50% of features are NaN (SMA-200 burn-in period) are dropped; remaining NaN are forward-filled then back-filled, which is financially appropriate since macro indicators hold their last-published value until the next release.


### Feature Importance

Permutation importance on the held-out test set: for each feature, randomly shuffle its column 15 times and measure the average drop in AUC. This approach is model-agnostic and avoids the known biases of tree-based split-count importance. Results are reported as raw importance and normalized gain percentage.



## Design Decisions

**Why a fixed split instead of rolling walk-forward?** Feature selection needs a stable scoring function. With a signal-to-noise ratio near zero, short walk-forward folds (6 months) are dominated by random variance — the same model can score 0.48 or 0.60 depending on which 6-month window it lands on. A single 2-year test window (~500 days) gives a statistically meaningful sample for deciding which features help.

**Why beam search instead of greedy elimination?** Features are correlated. Dropping VIX_Change_1d might look slightly bad in round 1, but if it lets the model rely more cleanly on VIX_ZScore_63, the resulting 2-feature-drop subset might outperform anything greedy elimination can reach. Beam search with width 10 explores these alternative paths.

**Why HistGradientBoostingClassifier instead of LightGBM?** Same histogram-based gradient boosting algorithm. Ships with scikit-learn — no extra dependency, no installation headaches. Hyperparameters map nearly 1:1 (`max_leaf_nodes` ↔ `num_leaves`, `l2_regularization` ↔ `reg_lambda`, etc.).

**Why not deep learning?** ~2,000 training observations. Neural networks need orders of magnitude more data to generalize at this noise level. Gradient-boosted trees are the right tool for tabular data with limited samples.

**Why the panic reversal score?** Markets tend to mean-revert after capitulation events — days where the S&P drops significantly and VIX spikes simultaneously. The feature is the product of the down-move magnitude and the VIX spike magnitude (both clipped at zero so it only fires on simultaneous down + VIX-up days). It gives the model a direct signal for these regime-specific rebound opportunities.


## Output Files

A full run produces the following in `output/`:

| File | Contents |
|------|----------|
| `feature_summary.csv` | Mean, std, min, max, NaN% for each feature |
| `feature_importance.csv` | Permutation importance ranking |
| `walkforward_folds.csv` | Per-fold metrics (if walk-forward was run) |
| `01_sp500_vix_regimes.png` | S&P 500 price with VIX high-vol shading |
| `02_yield_curve_spread.png` | 10Y−1Y spread with inversion highlighting |
| `03_vol_premium.png` | VIX minus realized vol over time |
| `04_gold_copper_ratio.png` | Risk-off/risk-on ratio |
| `05_feature_correlations.png` | Cross-correlation heatmap |
| `06_roc_curve.png` | ROC curve with AUC |
| `07_feature_importance.png` | Gain and split-count bar charts |
| `08_walkforward_folds.png` | Per-fold AUC (if walk-forward was run) |
| `09_cumulative_accuracy.png` | Running accuracy over the test period |
| `10_confusion_matrix.png` | Prediction counts |


## Dependencies

```
numpy        >= 1.24
pandas       >= 2.0
scikit-learn >= 1.3
yfinance     >= 0.2.31
pandas-datareader >= 0.10
matplotlib   >= 3.7
```
