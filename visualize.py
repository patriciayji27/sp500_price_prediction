"""
visualize.py — All plotting functions for the S&P 500 predictor.

Saves every figure to output/ as a PNG for easy embedding in reports.
Also displays via plt.show() when run interactively.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from pathlib import Path

from config import OUTPUT_DIR

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass  # fall back to default
plt.rcParams.update({
    "figure.dpi": 140,
    "figure.figsize": (12, 5),
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.size": 10,
})


def _save(fig, name: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{name}.png", bbox_inches="tight", dpi=150)


# ══════════════════════════════════════════════════════════════
# 1. EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════

def plot_sp500_with_regimes(price_df: pd.DataFrame, add_df: pd.DataFrame) -> None:
    """S&P 500 close with VIX overlay to visualize risk regimes."""
    fig, ax1 = plt.subplots(figsize=(14, 5))
    close = price_df["SP500_Close"]
    ax1.plot(close.index, close, color="#1a1a2e", linewidth=0.9, label="S&P 500 Close")
    ax1.set_ylabel("S&P 500", color="#1a1a2e")
    ax1.set_xlabel("")

    if "VIX_Close" in add_df.columns:
        ax2 = ax1.twinx()
        vix = add_df["VIX_Close"]
        ax2.fill_between(vix.index, 0, vix, alpha=0.25, color="#e63946", label="VIX")
        ax2.set_ylabel("VIX", color="#e63946")
        ax2.set_ylim(0, vix.max() * 1.3)

    ax1.set_title("S&P 500 Close with VIX Risk Regimes")
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    _save(fig, "01_sp500_vix_regimes")
    plt.show()


def plot_yield_curve_spread(add_df: pd.DataFrame) -> None:
    """10Y-1Y spread over time — recession indicator."""
    if "US_10Y" not in add_df.columns or "US_1Y" not in add_df.columns:
        print("  [skip] Yield data not available for spread plot.")
        return
    spread = add_df["US_10Y"] - add_df["US_1Y"]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(spread.index, spread, color="#457b9d", linewidth=0.8)
    ax.axhline(0, color="#e63946", linestyle="--", linewidth=1, alpha=0.7)
    ax.fill_between(spread.index, spread, 0,
                    where=(spread < 0), color="#e63946", alpha=0.2, label="Inverted")
    ax.fill_between(spread.index, spread, 0,
                    where=(spread >= 0), color="#2a9d8f", alpha=0.15, label="Normal")
    ax.set_title("US 10Y − 1Y Treasury Spread (Yield Curve)")
    ax.set_ylabel("Spread (pp)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    _save(fig, "02_yield_curve_spread")
    plt.show()


def plot_vol_premium(X: pd.DataFrame) -> None:
    """Volatility risk premium: implied (VIX) minus realized."""
    if "Vol_Premium_21d" not in X.columns:
        return
    vp = X["Vol_Premium_21d"].dropna()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(vp.index, vp, color="#6a4c93", linewidth=0.7, alpha=0.8)
    ax.axhline(vp.mean(), color="#333", linestyle="--", linewidth=0.8, label=f"Mean = {vp.mean():.1f}")
    ax.set_title("Volatility Risk Premium (VIX − Realized Vol 21d)")
    ax.set_ylabel("Premium (vol points)")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    _save(fig, "03_vol_premium")
    plt.show()


def plot_gold_copper_ratio(add_df: pd.DataFrame) -> None:
    """Gold/Copper ratio: risk-off vs risk-on sentiment."""
    if "Gold_Close" not in add_df.columns or "Copper_Close" not in add_df.columns:
        return
    ratio = (add_df["Gold_Close"] / add_df["Copper_Close"]).dropna()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ratio.index, ratio, color="#d4a373", linewidth=0.8)
    ax.set_title("Gold / Copper Ratio (Risk-Off vs Risk-On)")
    ax.set_ylabel("Ratio")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    _save(fig, "04_gold_copper_ratio")
    plt.show()


def plot_feature_correlations(X: pd.DataFrame) -> None:
    """Heatmap of feature cross-correlations."""
    corr = X.dropna().corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    _save(fig, "05_feature_correlations")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 2. MODEL EVALUATION PLOTS
# ══════════════════════════════════════════════════════════════

def plot_roc_curve(y_true, y_prob, label: str = "Hold-out") -> None:
    """ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#1d3557", linewidth=2,
            label=f"{label}  AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random (0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, "06_roc_curve")
    plt.show()


def plot_feature_importance(imp_df: pd.DataFrame, top_n: int = 20) -> None:
    """Horizontal bar chart of feature importance (gain)."""
    df = imp_df.head(top_n).iloc[::-1]  # flip for horizontal bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(df) * 0.35)))

    # Gain
    ax1.barh(df["feature"], df["gain_pct"], color="#264653")
    ax1.set_xlabel("Gain (%)")
    ax1.set_title("Feature Importance — Gain")

    # Split
    ax2.barh(df["feature"], df["split"], color="#2a9d8f")
    ax2.set_xlabel("Split count")
    ax2.set_title("Feature Importance — Split")

    fig.tight_layout()
    _save(fig, "07_feature_importance")
    plt.show()


def plot_walkforward_folds(wf_result: dict) -> None:
    """Bar chart of AUC per walk-forward fold."""
    folds = wf_result["fold_results"]
    fig, ax = plt.subplots(figsize=(max(8, len(folds) * 0.8), 5))

    colors = ["#2a9d8f" if a >= 0.5 else "#e76f51" for a in folds["auc"]]
    bars = ax.bar(folds["fold"], folds["auc"], color=colors, edgecolor="white", width=0.7)

    ax.axhline(0.5, color="#333", linestyle="--", linewidth=0.8, label="Random baseline (0.50)")
    ax.axhline(wf_result["pooled_auc"], color="#e63946", linestyle="-",
               linewidth=1.2, label=f"Pooled AUC = {wf_result['pooled_auc']:.4f}")

    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC")
    ax.set_title("Walk-Forward Cross-Validation — AUC per Fold")
    ax.set_ylim(0.35, 0.75)
    ax.legend()

    # Add fold date labels
    for i, row in folds.iterrows():
        label = f"{row['test_start'].strftime('%y/%m')}–{row['test_end'].strftime('%y/%m')}"
        ax.text(row["fold"], row["auc"] + 0.008, label,
                ha="center", va="bottom", fontsize=6.5, rotation=45)

    fig.tight_layout()
    _save(fig, "08_walkforward_folds")
    plt.show()


def plot_prediction_timeline(result: dict) -> None:
    """Cumulative accuracy over the test period to show stability."""
    y_test = result["y_test"]
    y_pred = result["y_pred"]

    correct = (y_test.values == y_pred).astype(float)
    cum_acc = pd.Series(correct, index=y_test.index).expanding().mean()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(cum_acc.index, cum_acc, color="#264653", linewidth=1)
    ax.axhline(0.5, color="#e63946", linestyle="--", linewidth=0.8, label="Random (50%)")
    ax.set_title("Cumulative Accuracy over Test Period")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 0.65)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    _save(fig, "09_cumulative_accuracy")
    plt.show()


def plot_confusion_matrix(y_true, y_pred) -> None:
    """Simple confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Down (0)", "Up (1)"])
    ax.set_yticklabels(["Down (0)", "Up (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    _save(fig, "10_confusion_matrix")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 3. MASTER EDA
# ══════════════════════════════════════════════════════════════

def run_eda(price_df, add_df, macro_df, X) -> None:
    """Run all exploratory data analysis plots."""
    print("\n" + "═" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("═" * 60)

    plot_sp500_with_regimes(price_df, add_df)
    plot_yield_curve_spread(add_df)
    plot_vol_premium(X)
    plot_gold_copper_ratio(add_df)
    plot_feature_correlations(X)

    print(f"\n  All EDA plots saved to {OUTPUT_DIR}/")


def run_evaluation_plots(holdout_result: dict, wf_result: dict, imp_df) -> None:
    """Run all model evaluation plots."""
    print("\n" + "═" * 60)
    print("MODEL EVALUATION PLOTS")
    print("═" * 60)

    plot_roc_curve(holdout_result["y_test"], holdout_result["y_prob"])
    plot_feature_importance(imp_df)
    if wf_result.get("fold_results") is not None:
        plot_walkforward_folds(wf_result)
    plot_prediction_timeline(holdout_result)
    plot_confusion_matrix(holdout_result["y_test"], holdout_result["y_pred"])

    print(f"\n  All evaluation plots saved to {OUTPUT_DIR}/")
