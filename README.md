# S&P 500 Next-Day Direction Predictor

A feature-engineering-driven machine learning system that predicts whether the S&P 500 will close higher or lower tomorrow. Originally developed for the AmplifyME S&P 500 Price Prediction Challenge (achieving **AUC 0.61** vs. a 0.51 baseline), this is a generalized, reproducible version that sources live data from public APIs and can be re-run at any time.

---

## Table of Contents

1. [Why This Is Hard (and What Makes It Work)](#why-this-is-hard)
2. [Methodology](#methodology)
3. [Feature Design](#feature-design)
4. [Model & Validation](#model--validation)
5. [Project Structure](#project-structure)
6. [Quick Start](#quick-start)
7. [Results](#results)
8. [Design Decisions & Lessons Learned](#design-decisions)

---

<a id="why-this-is-hard"></a>
## 1. Why This Is Hard (and What Makes It Work)

Predicting daily stock direction is one of the hardest problems in quantitative finance. Markets are highly efficient at the daily frequency — most public information is already priced in by the time you observe it. A random coin flip achieves AUC 0.50, and most naive models hover around that baseline.

What makes a meaningful edge possible:

- **Feature engineering over model complexity.** LightGBM with 400 trees is plenty. The alpha comes from *which signals* you feed it, not from stacking more layers.
- **Stationarity.** Raw price levels are non-stationary and will fool any model. Every feature in this system is a return, ratio, z-score, or diff — transformations that are mean-reverting or bounded.
- **Time-frequency alignment.** A critical and often overlooked issue: CPI updates monthly, GDP quarterly, but you're predicting daily. Using `diff(1)` on CPI produces a feature that is zero 95% of the time and noise the other 5%. Using `diff(63)` (one quarter) captures the *actual* information content at the frequency the data is released.
- **Controlled dimensionality.** With ~2,500 training days and dozens of candidate features, overfitting is the primary enemy. Backward elimination with a strict improvement threshold keeps only features that genuinely help out-of-sample.

---

<a id="methodology"></a>
## 2. Methodology

### Pipeline Overview

```
  ┌──────────────┐     ┌──────────────────┐     ┌────────────────┐
  │  Data Loader  │────▶│  Feature Engine   │────▶│  Feature Select │
  │  (yfinance,   │     │  (8 categories,   │     │  (backward      │
  │   FRED)       │     │   ~28 features)   │     │   elimination)  │
  └──────────────┘     └──────────────────┘     └───────┬────────┘
                                                         │
                       ┌──────────────────┐     ┌───────▼────────┐
                       │  Walk-Forward CV  │◀────│   LightGBM     │
                       │  (6-month folds)  │     │   Classifier   │
                       └──────────────────┘     └────────────────┘
```

1. **Data acquisition**: Download S&P 500 OHLCV, VIX, DXY, commodities, and Treasury yields from yfinance. Fetch macroeconomic series (CPI, fed funds, balance sheet, consumer confidence) from FRED. Cache everything locally as CSV.

2. **Feature engineering**: Transform raw data into ~28 stationary features grouped by economic theme (see Section 3).

3. **Feature selection**: Backward elimination iteratively removes the feature whose absence improves hold-out AUC the most, stopping when no removal helps.

4. **Model training**: LightGBM binary classifier with conservative hyperparameters (max_depth=4, num_leaves=15, min_child_samples=50) to prevent overfitting on a relatively small dataset.

5. **Evaluation**: Both chronological hold-out (last 20%, matching the competition) and walk-forward cross-validation (expanding window, 6-month test folds).

---

<a id="feature-design"></a>
## 3. Feature Design

Features are grouped into 8 economically-motivated categories. Every feature is a stationary transformation — no raw levels.

### Category 1: S&P 500 Return & Momentum (core signals)

| Feature | Formula | Rationale |
|---|---|---|
| `SP500_Return_1d` | close pct_change(1) | Most direct signal; captures overnight + intraday return |
| `SP500_Return_5d` | close pct_change(5) | Weekly momentum |
| `SP500_Return_21d` | close pct_change(21) | Monthly momentum |
| `SP500_RealVol_21d` | std(1d returns) × √252 | Annualized realized vol — high-vol regimes behave differently |
| `SP500_Dist_SMA_{10,50,200}` | close / SMA − 1 | Distance from moving average = mean-reversion pressure |
| `SMA_5_21_Cross` | SMA(5) / SMA(21) − 1 | Short-term trend signal |
| `SMA_50_200_Cross` | SMA(50) / SMA(200) − 1 | Long-term trend (golden/death cross) |
| `SP500_Range` | (High − Low) / Close | Intraday range as proxy for intraday volatility |
| `SP500_Body` | (Close − Open) / Close | Candlestick body: bullish vs bearish day structure |
| `Volume_Ratio` | Volume / SMA(Volume, 21) | Unusual volume = institutional activity |

### Category 2: VIX & Volatility

| Feature | Rationale |
|---|---|
| `VIX_Change_1d`, `VIX_Change_5d` | Fear gauge direction — VIX spikes precede continued selling, VIX drops signal calm |
| `VIX_Dist_SMA_21` | Extreme VIX relative to recent norm |
| `Vol_Premium_21d` | VIX − realized vol: the *volatility risk premium*. Well-documented in literature as equity predictor. Persistently positive (investors overpay for protection), but its *level* signals regime |

### Category 3: Yield Curve

| Feature | Rationale |
|---|---|
| `Spread_1Y_10Y` | 10Y − 1Y: the classic recession indicator. Inverted curve → growth pessimism |
| `Spread_1Y_10Y_Chg_21d` | *Change* in spread: steepening = expectations improving |
| `FedFunds_Chg_21d` | Fed policy direction — rate hikes vs cuts over the past month |

### Category 4: Dollar Strength

| Feature | Rationale |
|---|---|
| `DXY_Return_21d` | Strong dollar = headwind for S&P (multinationals earn abroad) |
| `DXY_ZScore_63` | Extreme dollar levels tend to mean-revert; identifies turning points |

### Category 5: Commodities

| Feature | Rationale |
|---|---|
| `Gold_Copper_Ratio_Chg` | Gold/Copper ratio change: gold is defensive, copper is cyclical. A rising ratio = risk-off rotation |
| `WTI_Return_5d` | Oil is an input cost. Sharp oil moves affect corporate margins and consumer spending |

### Category 6: Macro

| Feature | Rationale |
|---|---|
| `CPI_Chg_63d` | Inflation acceleration (quarterly diff matches monthly release cadence) |
| `ConsConf_Chg_21d` | Consumer confidence momentum — leads spending |
| `Fed_BS_Chg_63d` | Fed balance sheet growth rate — liquidity injection/withdrawal |
| `Semi_Chg_63d` | Semiconductor index momentum — tech sector leading indicator (SOX as proxy) |

### Category 7: Cross-Asset Regime Detection

| Feature | Rationale |
|---|---|
| `SP_Bond_CoMove_21d` | Rolling correlation between stocks and bond yields. When it flips sign, it signals a regime change (growth scare vs inflation shock) |
| `SP_VIX_Corr_21d` | Rolling stock-VIX correlation. Normally strongly negative; decorrelation = unusual market structure |

---

<a id="model--validation"></a>
## 4. Model & Validation

### Why LightGBM?

Gradient boosted trees are the workhorse of tabular prediction for good reasons: they handle missing values natively (important here — macro features have NaN during burn-in), capture nonlinear interactions between features, require no feature scaling, and train fast enough to run backward elimination over dozens of iterations.

### Why Conservative Hyperparameters?

With ~2,000–4,000 training days and ~25 features, overfitting is the dominant risk. The model uses:

- `max_depth=4`, `num_leaves=15` — shallow trees that can't memorize
- `min_child_samples=50` — each leaf needs substantial evidence
- `subsample=0.8`, `colsample_bytree=0.8` — row and column subsampling for regularization
- `reg_alpha=0.1`, `reg_lambda=1.0` — L1/L2 penalties

### Walk-Forward Validation

A random train/test split is **invalid** for financial time series — it lets the model see 2023 data while being tested on 2020 data, which leaks future information. Walk-forward validation fixes this:

```
Fold 1: Train [2005 → 2012], Test [2012 → mid-2013]
Fold 2: Train [2005 → mid-2013], Test [mid-2013 → 2014]
Fold 3: Train [2005 → 2014], Test [2014 → mid-2015]
  ...
Fold N: Train [2005 → 2024], Test [2024 → 2025]
```

Each fold trains only on past data and tests on the immediate future. The expanding window means later folds benefit from more training data. The pooled AUC across all folds is the most realistic estimate of real-world performance.

The chronological hold-out (last 20%) is also reported for direct comparison with the AmplifyME competition scoring.

### Feature Selection

Backward elimination is run on the hold-out AUC. In each round, every remaining feature is tentatively removed, and the one whose removal *most improves* AUC is permanently dropped. This continues until no single removal improves AUC by more than 0.001. This is more principled than forward selection for this problem because it starts with the full interaction structure intact.

---

<a id="project-structure"></a>
## 5. Project Structure

```
sp500-predictor/
├── main.py              # Orchestrator — run this
├── config.py            # All tunable parameters, paths, tickers, FRED IDs
├── data_loader.py       # yfinance + FRED download, caching, alignment
├── features.py          # Feature engineering pipeline + target
├── model.py             # LightGBM, holdout eval, walk-forward CV, selection
├── visualize.py         # EDA + model evaluation plots
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── data/                # Cached CSVs (auto-created on first run)
│   ├── yfinance_raw.csv
│   └── fred_raw.csv
└── output/              # All outputs (auto-created)
    ├── feature_summary.csv
    ├── feature_importance.csv
    ├── walkforward_folds.csv
    └── *.png            # All generated plots
```

---

<a id="quick-start"></a>
## 6. Quick Start

```bash
# Clone or copy the project
cd sp500-predictor

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Options
python main.py --skip-eda            # Skip visualization (faster)
python main.py --skip-selection      # Skip feature selection (use all)
python main.py --no-walkforward      # Skip walk-forward CV (faster)
python main.py --refresh-data        # Force re-download data
```

**First run** downloads ~20 years of daily data from yfinance and FRED (takes 1–2 minutes depending on connection). Subsequent runs use cached CSVs.

**Full pipeline** (with feature selection + walk-forward) takes roughly 3–8 minutes depending on hardware. Use `--skip-selection --no-walkforward` for a quick ~30-second run.

### Data Requirements

- **yfinance**: No API key needed. Pulls S&P 500, VIX, DXY, commodities.
- **FRED** (via `pandas-datareader`): No API key needed for `pandas-datareader` access. Pulls Treasury yields, CPI, fed funds, balance sheet, consumer confidence.

If any individual series fails to download, the pipeline gracefully degrades — it builds only the features for which data is available.

---

<a id="results"></a>
## 7. Results

### Competition Performance (AmplifyME)

| Metric | Score |
|---|---|
| AUC ROC | 0.61 |
| Baseline (random) | 0.51 |
| Rank | 1st |

### What to Expect from This Generalized Version

Because this version uses publicly-sourced data (not the competition's proprietary Quantlify dataset), and because the date range and exact data alignment may differ, expect AUC in the **0.55–0.62** range on hold-out, and **0.52–0.58** on walk-forward (which is a harder, more realistic test). Any consistent AUC above 0.53 on walk-forward represents genuine predictive signal.

---

<a id="design-decisions"></a>
## 8. Design Decisions & Lessons Learned

**Why not use raw price levels or raw macro values?**  
Non-stationary features (trending upward forever) give tree models the illusion of a pattern: "if price > X, it goes up" works beautifully on training data but breaks when the price regime shifts. Every feature here is a return, diff, ratio, or z-score.

**Why so few features (~25) when there are hundreds of possible signals?**  
With ~2,500 training days, the curse of dimensionality is real. Each additional noisy feature gives the model more opportunities to overfit. The backward elimination step enforces parsimony. In the competition, cutting features from 31 to ~25-28 improved AUC.

**Why not deep learning?**  
LSTMs and transformers need orders of magnitude more data. With daily S&P data going back 20 years, we have ~5,000 observations. Gradient boosted trees are the right tool for this data scale.

**Why does the volatility risk premium work?**  
VIX systematically overestimates future volatility (the "variance risk premium"). The gap between implied and realized vol is informative: when it's unusually high, fear is elevated and the market tends to recover; when it's compressed, complacency prevails and corrections become more likely.

**Why gold/copper and not just gold?**  
Gold alone is a noisy signal. The *ratio* isolates the risk-appetite component: gold rises in fear (safe haven), copper rises in growth (industrial demand). Their ratio distills macro sentiment more cleanly than either alone.

**The time-frequency problem.**  
The most subtle trap in this kind of project. CPI is released monthly. If you compute `CPI.diff(1)` on a daily-forward-filled series, the feature is zero on 95% of days and a random jump on the other 5% — this is noise, not signal. Using `diff(63)` (roughly one quarter) captures the underlying economic trend at the frequency the data actually updates. This insight, flagged by a mentor during the competition, was one of the biggest AUC improvements.

---

## License

This project is for educational and research purposes. It is not financial advice.
