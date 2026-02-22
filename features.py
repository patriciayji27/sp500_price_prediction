"""
features.py — Feature engineering pipeline for S&P 500 direction prediction.

Design principles
─────────────────
1. Every feature is a *stationary transformation* (returns, z-scores, ratios,
   diffs) — never raw price levels, which are non-stationary and leak trend.
2. Features are grouped into economically meaningful categories so the
   feature-selection stage can reason about *why* something works.
3. Time-frequency awareness: daily signals use short windows (1–21 days),
   while macro signals that update monthly/quarterly use 21–63 day diffs
   so the feature actually captures a genuine change.
4. The target is next-day direction: 1 if close[t+1] > close[t], else 0.

Dependency ordering
───────────────────
All features are computed from raw data (price_df, add_df, macro_df)
directly.  When a derived feature needs another feature already in X,
it is computed immediately after its dependency, never in a separate block.
This prevents the KeyError cascades that plagued earlier versions.
"""

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════
# TARGET VARIABLE
# ══════════════════════════════════════════════════════════════
def binary_target(close: pd.Series) -> pd.Series:
    """
    Binary target: 1 if next-day close is higher, 0 otherwise.
    The last row is NaN (no future available).
    """
    change = close.shift(-1) - close
    target = (change > 0).astype(float)
    target[change.isna()] = np.nan
    return target


# ══════════════════════════════════════════════════════════════
# HELPER: safe column check
# ══════════════════════════════════════════════════════════════
def _has(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().sum() > 50


# ══════════════════════════════════════════════════════════════
# FEATURE BUILDER
# ══════════════════════════════════════════════════════════════
def build_features(
    price_df: pd.DataFrame,
    add_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the full candidate feature matrix and target variable.

    Returns
    -------
    X : pd.DataFrame   — all candidate features, NaNs at burn-in only
    y : pd.Series       — binary target (next-day up = 1)
    """
    X = pd.DataFrame(index=price_df.index)
    close = price_df["SP500_Close"]

    # Short-term return (most directly relevant for 1-day prediction)
    X["SP500_Return_1d"] = close.pct_change(1)
    #X["SP500_Return_5d"] = close.pct_change(5)
    #X["SP500_Return_21d"] = close.pct_change(21)

    # Realized volatility (one window is enough)
    X["SP500_RealVol_21d"] = X["SP500_Return_1d"].rolling(21).std() * np.sqrt(252)

    # Distance from SMA (mean-reversion signals — no raw SMA levels!)
    for w in [ 50]:
        X[f"SP500_Dist_SMA_{w}"] = (close / close.rolling(w).mean()) - 1

    # SMA crossover (trend signal — use the spread, not the levels)
    X["SMA_5_21_Cross"] = close.rolling(5).mean() / close.rolling(21).mean() - 1
    X["SMA_50_200_Cross"] = close.rolling(50).mean() / close.rolling(200).mean() - 1

    # Intraday signals (already ratios = stationary)
    #X["SP500_Range"] = (price_df["SP500_High"] - price_df["SP500_Low"]) / close
    X["SP500_Body"] = (close - price_df["SP500_Open"]) / close

    # Volume ratio (stationary)
    X["Volume_Ratio"] = price_df["SP500_Volume"] / price_df["SP500_Volume"].rolling(21).mean()


    # ── 2. VIX (key risk signal) ───────────────────────────────
    vix = add_df["VIX_Close"]

    X["VIX_Change_1d"] = vix.pct_change(1)
    #X["VIX_Change_5d"] = vix.pct_change(5)
    X["VIX_Dist_SMA_21"] = (vix / vix.rolling(21).mean()) - 1

    # Vol risk premium (VIX vs realized — one of the best predictors in literature)
    X["Vol_Premium_21d"] = vix - (X["SP500_Return_1d"].rolling(21).std() * np.sqrt(252))


    # ── 3. YIELD CURVE (2-3 key signals, not 29) ──────────────
    # The 1Y-10Y spread is the single most important curve signal
    X["Spread_1Y_10Y"] = add_df["US_10Y"] - add_df["US_1Y"]
    X["Spread_1Y_10Y_Chg_21d"] = X["Spread_1Y_10Y"].diff(21)

    # Fed Funds rate change (policy direction)
    X["FedFunds_Chg_21d"] = macro_df["US_Effective_Fed_Funds"].diff(21)


    # ── 4. DOLLAR STRENGTH (equity headwind/tailwind) ─────────
    dxy = add_df["DXY_Close"]
    #X["DXY_Return_5d"] = dxy.pct_change(5)
    X["DXY_Return_21d"] = dxy.pct_change(21)

    # DXY z-score (extreme dollar = mean-reversion opportunity)
    dxy_mean = dxy.rolling(63).mean()
    dxy_std = dxy.rolling(63).std()
    X["DXY_ZScore_63"] = (dxy - dxy_mean) / dxy_std


    # ── 5. COMMODITIES (keep only the best signals) ───────────
    # Gold/Copper ratio change (risk-off vs risk-on — excellent macro signal)
    gold_copper = add_df["Gold_Close"] / add_df["Copper_Close"]
    #X["Gold_Copper_Ratio_Chg"] = gold_copper.pct_change(21)

    # Oil return (energy cost impact)
    #X["WTI_Return_5d"] = add_df["WTI_Close"].pct_change(5)


    # ── 6. POSITIONING (contrarian signal) ─────────────────────
    #spec = add_df["SP500_Net_Spec_Positions"]
    #spec_mean = spec.rolling(63).mean()
    #spec_std = spec.rolling(63).std()
    #X["Spec_Pos_ZScore"] = (spec - spec_mean) / spec_std


    # ── 7. MACRO (only quarterly-appropriate signals) ──────────
    # These change infrequently so only long-horizon changes make sense
    #X["CPI_Chg_63d"] = macro_df["US_CPI"].diff(63)
    #X["ConsConf_Chg_21d"] = macro_df["US_Consumer_Confidence_Index"].diff(21)

    # Fed balance sheet growth (liquidity)
    X["Fed_BS_Chg_63d"] = macro_df["US_Fed_Balance_Sheet_Total_Assets"].pct_change(63)

    # Semiconductor momentum (tech leading indicator)
    #X["Semi_Chg_63d"] = macro_df["US_Semiconductor_Sales_Index"].pct_change(63)


    # ── 8. CROSS-ASSET CORRELATION (regime detection) ─────────
    ret_1d = close.pct_change(1)
    #X["SP_Bond_CoMove_21d"] = ret_1d.rolling(21).corr(add_df["US_10Y"].diff(1))
    X["SP_VIX_Corr_21d"] = ret_1d.rolling(21).corr(vix.pct_change(1))



    # -- additoinal features
        # Short-horizon reversal / follow-through signals (useful for reducing FN on rebounds)
    prev_close = close.shift(1)
    true_range = (price_df["SP500_High"] - price_df["SP500_Low"]).replace(0, np.nan)

    # Overnight gap relative to prior close
    X["SP500_Gap_1d"] = (price_df["SP500_Open"] - prev_close) / prev_close

    # Close location value in daily range: +1 = close near high, -1 = close near low
    X["SP500_CloseLoc"] = (
        ((close - price_df["SP500_Low"]) - (price_df["SP500_High"] - close)) / true_range
    )

    #X["SP500_Drawdown_5d"] = close / close.rolling(5).max() - 1

        # VIX regime / shock features
    vix_mean_63 = vix.rolling(63).mean()
    vix_std_63 = vix.rolling(63).std()
    X["VIX_ZScore_63"] = (vix - vix_mean_63) / vix_std_63

    # Corrected vol risk premium in percentage-point units (VIX ~ 20 means 20%)
    rv_21 = X["SP500_Return_1d"].rolling(21).std() * np.sqrt(252)
    X["Vol_Premium_21d_pp"] = vix - (100.0 * rv_21)

    # "Panic" score: down day + VIX spike (can help catch next-day mean reversion up)
    X["Panic_Reversal_Score"] = (
        (-X["SP500_Return_1d"]).clip(lower=0) * X["VIX_Change_1d"].clip(lower=0)
    )

        # Explicit interactions can help a small-sample tree model pick up rebound regimes faster
    # X["Oversold_x_VIXStress"] = (
    #     (-X["SP500_Dist_SMA_50"]).clip(lower=0) * X["VIX_Dist_SMA_21"].clip(lower=0)
    # )

    # X["WeakBody_x_HighVol"] = (
    #     (-X["SP500_Body"]).clip(lower=0) * (X["Volume_Ratio"] - 1).clip(lower=0)
    # )





    # ── TARGET ──────────────────────────────────────────────
    y = binary_target(close)

    # ── Summary ─────────────────────────────────────────────
    print(f"  Built {X.shape[1]} features: {list(X.columns)}")

    return X, y


# ══════════════════════════════════════════════════════════════
# FEATURE SUMMARY
# ══════════════════════════════════════════════════════════════
def feature_summary(X: pd.DataFrame) -> pd.DataFrame:
    """Quick stats table for the feature matrix."""
    stats = pd.DataFrame({
        "mean":     X.mean(),
        "std":      X.std(),
        "min":      X.min(),
        "max":      X.max(),
        "nan_pct":  X.isna().mean() * 100,
    })
    return stats.round(4)


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_all_data

    data = load_all_data()
    X, y = build_features(**data)
    print(f"\nFeature matrix: {X.shape}")
    print(f"Target: {y.shape},  class balance: {y.mean():.3f}")
    print(f"\n{feature_summary(X)}")
