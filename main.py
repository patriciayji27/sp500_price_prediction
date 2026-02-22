"""
main.py — S&P 500 Next-Day Direction Predictor

Run the full pipeline:
    python main.py

Flags:
    --skip-eda          Skip exploratory visualizations
    --skip-selection    Skip backward feature selection (use all features)
    --refresh-data      Force re-download of all data
    --no-walkforward    Skip walk-forward CV (faster, holdout only)
"""

import sys
import time
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from config import OUTPUT_DIR, HOLDOUT_TEST_RATIO
from data_loader import load_all_data
from features import build_features, feature_summary

from visualize import run_eda, run_evaluation_plots

from model import (
    evaluate_holdout,
    evaluate_fixed_year_windows,
    backward_elimination_beam,   # <-- add this
    feature_importance,
)

def main():
    parser = argparse.ArgumentParser(description="S&P 500 Direction Predictor")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip exploratory data analysis plots")
    parser.add_argument("--skip-selection", action="store_true",
                        help="Skip backward feature selection")
    parser.add_argument("--refresh-data", action="store_true",
                        help="Force re-download all data")
    parser.add_argument("--no-walkforward", action="store_true",
                        help="Skip walk-forward CV")
    args = parser.parse_args()

    t0 = time.time()

    # ══════════════════════════════════════════════════════════
    # STEP 1: DATA ACQUISITION
    # ══════════════════════════════════════════════════════════
    data = load_all_data(force_refresh=args.refresh_data)
    price_df = data["price_df"]
    add_df = data["add_df"]
    macro_df = data["macro_df"]

    # ══════════════════════════════════════════════════════════
    # STEP 2: FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════
    print("═" * 60)
    print("FEATURE ENGINEERING")
    print("═" * 60)

    X, y = build_features(price_df, add_df, macro_df)

    print(f"  Feature matrix : {X.shape[0]} rows × {X.shape[1]} features")
    print(f"  Target balance : {y.mean():.3f} (fraction of up days)")
    print(f"  Date range     : {X.index[0].date()} → {X.index[-1].date()}")
    print()

    summary = feature_summary(X)
    print(summary.to_string())
    summary.to_csv(OUTPUT_DIR / "feature_summary.csv")
    print(f"\n  Feature summary saved to {OUTPUT_DIR}/feature_summary.csv")

    # ══════════════════════════════════════════════════════════
    # STEP 3: EXPLORATORY DATA ANALYSIS
    # ══════════════════════════════════════════════════════════
    if not args.skip_eda:
        run_eda(price_df, add_df, macro_df, X)

    # ══════════════════════════════════════════════════════════
    # STEP 4: FEATURE SELECTION (backward elimination)
    # ══════════════════════════════════════════════════════════
    if not args.skip_selection:
        print("\n" + "═" * 60)
        print("BACKWARD FEATURE SELECTION")
        print("═" * 60)

        kept, removed, sel_auc = backward_elimination_beam(X, y,
                                                           scorer="fixed_8y2y",
                                                           beam_width=10,
                                                           children_per_state=3, min_features=5,
                                                           patience_rounds=2,
                                                           verbose=True,)

        if removed:
            print(f"\n  Trimmed {len(removed)} features. Using {len(kept)}.")
            X = X[kept]
        else:
            print("\n  No features removed — full set retained.")
    else:
        print("\n  [skip] Feature selection skipped.")

    # ══════════════════════════════════════════════════════════
    # STEP 5: HOLD-OUT EVALUATION (competition-style)
    # ══════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("HOLD-OUT EVALUATION (same fixed 8y/2y window)")
    print("═" * 60)

    holdout = evaluate_fixed_year_windows(
        X, y,
        start_date="2015-01-01",
        end_date="2025-01-01",   # same window used by fixed_8y2y scorer
        train_years=8,
        test_years=2,
        verbose=True,
    )

    imp = feature_importance(
        holdout["model"], list(X.columns),
        holdout["X_test"], holdout["y_test"],
    )
    print(f"\n  Top features by importance:")
    print(imp.head(10).to_string(index=False))
    imp.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    # ══════════════════════════════════════════════════════════
    # STEP 6: WALK-FORWARD CROSS-VALIDATION (rolling window)
    # ══════════════════════════════════════════════════════════
    wf_result = None
    if not args.no_walkforward:
        print("\n" + "═" * 60)
        print("WALK-FORWARD CROSS-VALIDATION (rolling window)")
        print("═" * 60)

        wf_result = evaluate_fixed_year_windows(X, y, verbose=True)

    # ══════════════════════════════════════════════════════════
    # STEP 7: EVALUATION PLOTS
    # ══════════════════════════════════════════════════════════
    if wf_result is None:
        # Create a minimal wf_result for plotting
        wf_result = {
            "pooled_auc": holdout["auc"],
            "mean_auc": holdout["auc"],
            "fold_results": None,
            "all_y_true": holdout["y_test"].values,
            "all_y_prob": holdout["y_prob"],
        }

    run_evaluation_plots(holdout, wf_result, imp)

    # ══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print("\n" + "═" * 60)
    print("RESULTS SUMMARY")
    print("═" * 60)
    print(f"  Features used       : {X.shape[1]}")
    print(f"  Hold-out AUC        : {holdout['auc']:.4f}")
    print(f"  Hold-out Accuracy   : {holdout['accuracy']:.4f}")
    if wf_result and wf_result.get("fold_results") is not None:
        print(f"  Walk-forward AUC    : {wf_result['pooled_auc']:.4f} (pooled)")
        print(f"  Walk-forward AUC    : {wf_result['mean_auc']:.4f} (mean)")
    print(f"  Baseline (random)   : 0.5000")
    print(f"  Runtime             : {elapsed:.1f}s")
    print(f"  Outputs saved to    : {OUTPUT_DIR}/")
    print("═" * 60)

    return holdout, wf_result


if __name__ == "__main__":
    main()
