"""
data_loader.py — Fetch, cache, and align all data needed for the predictor.

Data sources
────────────
  • yfinance   — S&P 500 OHLCV, VIX, DXY, commodities, currencies, SOX
  • FRED       — treasury yields, fed funds, CPI, unemployment, confidence,
                 fed balance sheet, corporate profits, trade price indices

Every series is forward-filled to a common business-day index so that
downstream feature engineering never sees misaligned dates.

Caching: raw downloads are saved to data/*.csv.  Delete the files (or the
whole data/ folder) to force a fresh download.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    DATA_DIR, START_DATE, END_DATE,
    YFINANCE_TICKERS, FRED_SERIES, SOX_TICKER,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _cache_path(name: str) -> Path:
    return DATA_DIR / f"{name}.csv"


def _save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(_cache_path(name))


def _load(name: str) -> pd.DataFrame | None:
    p = _cache_path(name)
    if p.exists():
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df
    return None


# ──────────────────────────────────────────────────────────────
# yfinance downloader
# ──────────────────────────────────────────────────────────────
def _download_yfinance() -> pd.DataFrame:
    """Download all yfinance tickers into a single wide DataFrame."""
    cached = _load("yfinance_raw")
    if cached is not None and len(cached) > 100:
        print(f"  [cache] yfinance data loaded  ({len(cached)} rows)")
        return cached

    print("  [download] Fetching yfinance tickers …")
    frames = {}

    for label, ticker in YFINANCE_TICKERS.items():
        try:
            raw = yf.download(
                ticker, start=START_DATE, end=END_DATE,
                auto_adjust=True, progress=False,
            )
            if raw.empty:
                print(f"    ⚠ {label} ({ticker}): empty result, skipping")
                continue

            # yfinance sometimes returns MultiIndex columns; flatten
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            # Rename columns with prefix
            if label == "SP500":
                rename = {
                    "Open": "SP500_Open", "High": "SP500_High",
                    "Low": "SP500_Low", "Close": "SP500_Close",
                    "Volume": "SP500_Volume",
                }
            elif label == "VIX":
                rename = {
                    "Open": "VIX_Open", "High": "VIX_High",
                    "Low": "VIX_Low", "Close": "VIX_Close",
                }
            else:
                rename = {"Close": f"{label}_Close"}

            for old, new in rename.items():
                if old in raw.columns:
                    frames[new] = raw[old]

            print(f"    ✓ {label:12s} ({ticker}): {len(raw)} rows")
        except Exception as e:
            print(f"    ✗ {label} ({ticker}): {e}")

    # SOX (semiconductor) index
    try:
        sox = yf.download(SOX_TICKER, start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)
        if isinstance(sox.columns, pd.MultiIndex):
            sox.columns = sox.columns.get_level_values(0)
        if not sox.empty:
            frames["SOX_Close"] = sox["Close"]
            print(f"    ✓ {'SOX':12s} ({SOX_TICKER}): {len(sox)} rows")
    except Exception as e:
        print(f"    ✗ SOX: {e}")

    df = pd.DataFrame(frames)
    df.index.name = "Date"
    _save(df, "yfinance_raw")
    print(f"  [saved] yfinance_raw.csv  ({len(df)} rows, {len(df.columns)} cols)")
    return df


# ──────────────────────────────────────────────────────────────
# FRED downloader
# ──────────────────────────────────────────────────────────────
def _download_fred() -> pd.DataFrame:
    """Download FRED series via pandas-datareader (no API key needed)."""
    cached = _load("fred_raw")
    if cached is not None and len(cached) > 100:
        print(f"  [cache] FRED data loaded  ({len(cached)} rows)")
        return cached

    print("  [download] Fetching FRED series via pandas-datareader …")

    try:
        import pandas_datareader.data as web
    except ImportError:
        print("    ⚠ pandas-datareader not installed. Run:")
        print("      pip install pandas-datareader")
        print("    Returning empty FRED DataFrame.")
        return pd.DataFrame()

    frames = {}
    for label, series_id in FRED_SERIES.items():
        try:
            s = web.DataReader(series_id, "fred", START_DATE, END_DATE)
            s = s.iloc[:, 0]        # Series
            s.name = label
            frames[label] = s
            print(f"    ✓ {label:45s} ({series_id}): {s.notna().sum()} values")
        except Exception as e:
            print(f"    ✗ {label:45s} ({series_id}): {e}")

    df = pd.DataFrame(frames)
    df.index.name = "Date"
    _save(df, "fred_raw")
    print(f"  [saved] fred_raw.csv  ({len(df)} rows, {len(df.columns)} cols)")
    return df


# ──────────────────────────────────────────────────────────────
# Merge & align
# ──────────────────────────────────────────────────────────────
def load_all_data(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """
    Returns a dict with three DataFrames aligned to the same business-day
    index, matching the original challenge structure:

        price_df : SP500 OHLCV
        add_df   : VIX, DXY, commodities, currencies, yields, SOX
        macro_df : CPI, unemployment, confidence, fed balance sheet, etc.

    All series are forward-filled so NaNs only appear at the very start
    (before the earliest observation for a given series).
    """
    if force_refresh:
        for p in DATA_DIR.glob("*.csv"):
            p.unlink()

    print("═" * 60)
    print("DATA ACQUISITION")
    print("═" * 60)

    yf_df = _download_yfinance()
    fred_df = _download_fred()

    # ── Build a common business-day index ────────────────────
    all_dates = yf_df.index.union(fred_df.index) if len(fred_df) else yf_df.index
    bdays = pd.bdate_range(all_dates.min(), all_dates.max())

    combined = pd.DataFrame(index=bdays)
    combined.index.name = "Date"

    # Merge yfinance
    combined = combined.join(yf_df, how="left")

    # Merge FRED
    if len(fred_df):
        combined = combined.join(fred_df, how="left")

    # Forward-fill (macro/weekly data published then constant until update)
    combined = combined.ffill()

    # ── Split into the three DataFrames ──────────────────────
    # Explicit column assignment to match the original challenge structure:
    #   price_df : S&P 500 OHLCV
    #   add_df   : VIX, DXY, commodities, currencies, yields, SOX
    #   macro_df : CPI, unemployment, confidence, fed balance sheet, etc.

    sp500_cols = [c for c in combined.columns if c.startswith("SP500_")]

    # Macro = slow-moving economic indicators (monthly/quarterly)
    macro_names = {
        "US_Effective_Fed_Funds",
        "US_Fed_Balance_Sheet_Total_Assets",
        "US_CPI",
        "US_Unemployment_Rate",
        "US_Consumer_Confidence_Index",
        "US_Export_Price_Index",
        "US_Import_Price_Index",
        "US_Corporate_Profits",
    }
    macro_cols = [c for c in combined.columns if c in macro_names]

    # Everything else (VIX, DXY, commodities, currencies, yields, SOX)
    add_cols = [c for c in combined.columns
                if c not in sp500_cols and c not in macro_cols]

    price_df = combined[sp500_cols].copy()
    macro_df = combined[macro_cols].copy()
    add_df = combined[add_cols].copy()

    # ── Drop rows where S&P 500 close is missing (pre-history) ──
    valid = price_df["SP500_Close"].notna()
    price_df = price_df.loc[valid]
    macro_df = macro_df.loc[valid]
    add_df = add_df.loc[valid]

    print()
    print(f"  price_df : {price_df.shape}  cols={list(price_df.columns)}")
    print(f"  macro_df : {macro_df.shape}  cols={list(macro_df.columns)}")
    print(f"  add_df   : {add_df.shape}  cols={list(add_df.columns)}")
    print("═" * 60)
    print()

    return {"price_df": price_df, "macro_df": macro_df, "add_df": add_df}


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_all_data()
    for name, df in data.items():
        print(f"\n{name}:")
        print(df.tail(3))
