"""
config.py — Central configuration for the S&P 500 direction predictor.

All tuneable parameters live here so experiments are reproducible
and nothing is hard-coded deep inside the pipeline.
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════
# DATE RANGE
# ══════════════════════════════════════════════════════════════
# 2013-01-01 gives ~200 trading days of burn-in for SMA-200,
# so real features start ~Oct 2013.  Avoids GFC-era (2008-2012)
# market microstructure which poisons modern predictions.
# Walk-forward first test fold begins ~Oct 2016.
START_DATE = "2015-01-01"
END_DATE = None  # defaults to today

# ══════════════════════════════════════════════════════════════
# DATA SOURCES — yfinance
# ══════════════════════════════════════════════════════════════
YFINANCE_TICKERS = {
    "SP500":     "^GSPC",
    "VIX":       "^VIX",
    "DXY":       "DX-Y.NYB",
    "EURUSD":    "EURUSD=X",
    "USDJPY":    "JPY=X",
    "WTI":       "CL=F",
    "Brent":     "BZ=F",
    "Gold":      "GC=F",
    "Copper":    "HG=F",
    "NaturalGas":"NG=F",
}

# ══════════════════════════════════════════════════════════════
# DATA SOURCES — FRED
# ══════════════════════════════════════════════════════════════
FRED_SERIES = {
    # Treasury yields (daily)
    "US_1Y":   "DGS1",
    "US_5Y":   "DGS5",
    "US_10Y":  "DGS10",
    "US_20Y":  "DGS20",
    "US_30Y":  "DGS30",
    # Policy rate
    "US_Effective_Fed_Funds":                  "DFF",
    # Balance sheet (weekly)
    "US_Fed_Balance_Sheet_Total_Assets":       "WALCL",
    # Macro (monthly/quarterly)
    "US_CPI":                                  "CPIAUCSL",
    "US_Unemployment_Rate":                    "UNRATE",
    "US_Consumer_Confidence_Index":            "UMCSENT",
    # Trade prices
    "US_Export_Price_Index":                    "IQ",
    "US_Import_Price_Index":                   "IR",
    # Corporate profits (quarterly)
    "US_Corporate_Profits":                    "CP",
}

# SOX index from yfinance as semiconductor momentum proxy
SOX_TICKER = "^SOX"

# ══════════════════════════════════════════════════════════════
# MODEL PARAMETERS
# ══════════════════════════════════════════════════════════════
RANDOM_SEED = 42

# Walk-forward cross-validation
WALKFORWARD_MIN_TRAIN_DAYS = 756    # ~3 years minimum training window
WALKFORWARD_MAX_TRAIN_DAYS = 1512   # ~6 years max (rolling window — drops old data)
WALKFORWARD_TEST_DAYS = 126         # ~6 months per fold
WALKFORWARD_STEP_DAYS = 126         # slide forward 6 months each fold

# Simple hold-out split (matches original competition: 80/20)
HOLDOUT_TEST_RATIO = 0.20

# Backward feature selection
MAX_BACKWARD_ROUNDS = 15
ABLATION_IMPROVEMENT_THRESHOLD = 0.001
