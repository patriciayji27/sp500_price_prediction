"""
model.py — Gradient-boosted classifier + feature selection utilities.

Uses scikit-learn's HistGradientBoostingClassifier (same algorithm family as
LightGBM — histogram-based gradient boosting, native NaN handling).
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    roc_curve,
)

from config import (
    WALKFORWARD_MIN_TRAIN_DAYS,
    WALKFORWARD_MAX_TRAIN_DAYS,
    WALKFORWARD_TEST_DAYS,
    WALKFORWARD_STEP_DAYS,
    HOLDOUT_TEST_RATIO,
    ABLATION_IMPROVEMENT_THRESHOLD,
    MAX_BACKWARD_ROUNDS,
    RANDOM_SEED,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════
# Tuned for ~3000 rows, ~25 features, weak signal environment.
_MODEL_PARAMS = dict(
    max_iter=600,
    learning_rate=0.05,
    max_depth=6,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.3,
    max_features=0.8,
    early_stopping=True,
    n_iter_no_change=50,
    validation_fraction=0.10,
    random_state=RANDOM_SEED,
    verbose=0,
)


# ══════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════
def _prepare(X: pd.DataFrame, y: pd.Series):
    """
    Clean the feature matrix and target:
      1. Drop rows where target is NaN (last row, no future).
      2. Drop burn-in rows where >50% of features are NaN.
      3. Forward-fill remaining NaN.
      4. Back-fill any NaN still at the start.
    """
    mask = y.notna()
    X_c = X.loc[mask].copy()
    y_c = y.loc[mask].copy()

    # Drop rows where most features are still NaN (SMA-200 burn-in, etc.)
    valid = X_c.notna().mean(axis=1) > 0.5
    X_c = X_c.loc[valid]
    y_c = y_c.loc[valid]

    # Forward-fill then back-fill remaining NaN
    X_c = X_c.ffill().bfill()

    return X_c, y_c


def _ensure_sorted(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Ensure chronological order for time-based splits."""
    if not X.index.is_monotonic_increasing:
        X = X.sort_index()
        y = y.loc[X.index]
    return X, y


# ══════════════════════════════════════════════════════════════
# CORE TRAIN / PREDICT
# ══════════════════════════════════════════════════════════════
def _fit_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, HistGradientBoostingClassifier]:
    """Fit single model and return (labels, probabilities, model)."""
    model = HistGradientBoostingClassifier(**_MODEL_PARAMS)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return y_pred, y_prob, model


def _build_eval_output(
    *,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model: HistGradientBoostingClassifier,
    verbose_title: str,
    verbose_lines: list[str],
    verbose: bool,
) -> dict:
    """
    Standardized payload so holdout/fixed splits can both feed:
      - feature importance
      - plots
      - beam scorer / summary code
    """
    test_auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    if verbose:
        print(verbose_title)
        for line in verbose_lines:
            print(line)
        print(f"  Test AUC: {test_auc:.4f}")
        print(f"  Test Acc: {acc:.4f}")

    return {
        # scorer compatibility
        "test_auc": test_auc,
        "auc": test_auc,
        "accuracy": acc,

        # holdout-style payload (for feature importance / plots)
        "report": report,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "model": model,
        "X_test": X_test,

        # split metadata
        "train_start": X_train.index[0],
        "train_end": X_train.index[-1],
        "test_start": X_test.index[0],
        "test_end": X_test.index[-1],
        "train_size": len(X_train),
        "test_size": len(X_test),

        # compatibility with plotting code
        "all_y_true": y_test.to_numpy(),
        "all_y_prob": np.asarray(y_prob),
    }


# ══════════════════════════════════════════════════════════════
# HOLD-OUT EVALUATION
# ══════════════════════════════════════════════════════════════
def evaluate_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    test_ratio: float = HOLDOUT_TEST_RATIO,
    verbose: bool = True,
) -> dict:
    """
    Chronological hold-out: train on first (1-test_ratio) rows,
    test on the last test_ratio rows.
    """
    X_clean, y_clean = _prepare(X, y)
    X_clean, y_clean = _ensure_sorted(X_clean, y_clean)

    split_idx = int(len(X_clean) * (1 - test_ratio))
    if split_idx <= 0 or split_idx >= len(X_clean):
        raise ValueError(
            f"Invalid holdout split with test_ratio={test_ratio}. "
            f"Need 0 < split_idx < {len(X_clean)}, got {split_idx}."
        )

    X_train, X_test = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
    y_train, y_test = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]

    y_pred, y_prob, model = _fit_predict(X_train, y_train, X_test)

    return _build_eval_output(
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        model=model,
        verbose_title="Hold-out evaluation:",
        verbose_lines=[
            f"  Train: {X_train.index[0].date()} → {X_train.index[-1].date()}  ({len(X_train)} days)",
            f"  Test:  {X_test.index[0].date()} → {X_test.index[-1].date()}  ({len(X_test)} days)",
        ],
        verbose=verbose,
    )


# ══════════════════════════════════════════════════════════════
# FIXED YEAR SPLIT EVALUATION (8Y/2Y BY DEFAULT)
# ══════════════════════════════════════════════════════════════
def evaluate_fixed_year_windows(
    X: pd.DataFrame,
    y: pd.Series,
    start_date: str = "2015-01-01",
    end_date: str = "2025-01-01",
    train_years: int = 8,
    test_years: int = 2,
    verbose: bool = True,
) -> dict:
    """
    Fixed (non-rolling) split with explicit date boundaries.

    Default:
      - Use data in [2015-01-01, 2025-01-01)
      - Train: first 8 years  => [2015-01-01, 2023-01-01)
      - Test:  final 2 years  => [2023-01-01, 2025-01-01)
    """
    X_clean, y_clean = _prepare(X, y)

    if not isinstance(X_clean.index, pd.DatetimeIndex):
        raise TypeError("X must have a DatetimeIndex for date-based splitting.")
    X_clean, y_clean = _ensure_sorted(X_clean, y_clean)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    split_ts = start_ts + pd.DateOffset(years=train_years)

    expected_end_ts = split_ts + pd.DateOffset(years=test_years)
    if end_ts != expected_end_ts:
        raise ValueError(
            "end_date must equal start_date + (train_years + test_years). "
            f"Got end_date={end_ts.date()}, expected {expected_end_ts.date()}."
        )

    # Restrict to requested window [start, end)
    mask = (X_clean.index >= start_ts) & (X_clean.index < end_ts)
    Xw = X_clean.loc[mask]
    yw = y_clean.loc[Xw.index]

    if len(Xw) == 0:
        raise ValueError("No data in the requested [start_date, end_date) window.")

    # Build train/test by dates
    X_tr = Xw.loc[(Xw.index >= start_ts) & (Xw.index < split_ts)]
    y_tr = yw.loc[X_tr.index]

    X_te = Xw.loc[(Xw.index >= split_ts) & (Xw.index < end_ts)]
    y_te = yw.loc[X_te.index]

    if len(X_tr) == 0 or len(X_te) == 0:
        raise ValueError(
            f"Empty split. Train rows={len(X_tr)}, test rows={len(X_te)}. "
            f"Check coverage around split={split_ts.date()}."
        )

    # IMPORTANT: must run even when verbose=False (beam search scorer path)
    y_pred, y_prob, model = _fit_predict(X_tr, y_tr, X_te)

    return _build_eval_output(
        X_train=X_tr,
        X_test=X_te,
        y_test=y_te,
        y_pred=y_pred,
        y_prob=y_prob,
        model=model,
        verbose_title="Fixed year windows:",
        verbose_lines=[
            f"  Window: {start_ts.date()} → {end_ts.date()} (end exclusive)",
            f"  Train:  {start_ts.date()} → {split_ts.date()} ({len(X_tr)} rows)",
            f"  Test:   {split_ts.date()} → {end_ts.date()} ({len(X_te)} rows)",
        ],
        verbose=verbose,
    )


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD CROSS-VALIDATION (ROLLING WINDOW)
# ══════════════════════════════════════════════════════════════
def evaluate_walkforward(
    X: pd.DataFrame,
    y: pd.Series,
    min_train: int = WALKFORWARD_MIN_TRAIN_DAYS,
    max_train: int = WALKFORWARD_MAX_TRAIN_DAYS,
    test_days: int = WALKFORWARD_TEST_DAYS,
    step_days: int = WALKFORWARD_STEP_DAYS,
    verbose: bool = True,
) -> dict:
    """
    Rolling-window walk-forward validation.

    Each fold:
      • Train on rows [max(0, train_end-max_train) .. train_end)
      • Test on rows [train_end .. train_end+test_days)
      • Slide forward by step_days
    """
    X_clean, y_clean = _prepare(X, y)
    X_clean, y_clean = _ensure_sorted(X_clean, y_clean)

    n = len(X_clean)
    if min_train <= 0 or max_train < min_train or test_days <= 0 or step_days <= 0:
        raise ValueError("Invalid walk-forward parameters.")

    fold_results = []
    all_y_true = []
    all_y_prob = []

    train_end = min_train
    fold_num = 0

    while train_end + test_days <= n:
        fold_num += 1
        test_start = train_end
        test_end = min(train_end + test_days, n)

        # Rolling window: cap training to max_train days
        train_start = max(0, train_end - max_train)

        X_tr = X_clean.iloc[train_start:train_end]
        y_tr = y_clean.iloc[train_start:train_end]
        X_te = X_clean.iloc[test_start:test_end]
        y_te = y_clean.iloc[test_start:test_end]

        y_pred, y_prob, _ = _fit_predict(X_tr, y_tr, X_te)
        fold_auc = roc_auc_score(y_te, y_prob)
        fold_acc = accuracy_score(y_te, y_pred)

        fold_results.append(
            {
                "fold": fold_num,
                "train_start": X_tr.index[0],
                "train_end": X_tr.index[-1],
                "test_start": X_te.index[0],
                "test_end": X_te.index[-1],
                "train_size": len(X_tr),
                "test_size": len(X_te),
                "auc": fold_auc,
                "accuracy": fold_acc,
            }
        )

        all_y_true.extend(y_te.values)
        all_y_prob.extend(y_prob)

        if verbose:
            print(
                f"    Fold {fold_num:2d}: "
                f"train={X_tr.index[0].strftime('%Y-%m')}→{X_tr.index[-1].strftime('%Y-%m')} "
                f"({len(X_tr)}d) "
                f"test={X_te.index[0].strftime('%Y-%m')}→{X_te.index[-1].strftime('%Y-%m')} "
                f"({len(X_te)}d) "
                f"AUC={fold_auc:.4f}"
            )

        train_end += step_days

    if fold_num == 0:
        raise ValueError(
            f"No walk-forward folds created. Need at least {min_train + test_days} rows, have {n}."
        )

    pooled_auc = roc_auc_score(all_y_true, all_y_prob)
    mean_auc = float(np.mean([f["auc"] for f in fold_results]))

    if verbose:
        below_50 = sum(1 for f in fold_results if f["auc"] < 0.50)
        print(f"\n  Walk-forward pooled AUC:  {pooled_auc:.4f}")
        print(f"  Walk-forward mean AUC:    {mean_auc:.4f}  ({fold_num} folds)")
        print(f"  Folds below 0.50:         {below_50}/{fold_num}")

    return {
        "pooled_auc": pooled_auc,
        "mean_auc": mean_auc,
        "fold_results": pd.DataFrame(fold_results),
        "all_y_true": np.array(all_y_true),
        "all_y_prob": np.array(all_y_prob),
    }


# ══════════════════════════════════════════════════════════════
# BACKWARD FEATURE SELECTION
# ══════════════════════════════════════════════════════════════
def backward_elimination(
    X: pd.DataFrame,
    y: pd.Series,
    max_rounds: int = MAX_BACKWARD_ROUNDS,
    threshold: float = ABLATION_IMPROVEMENT_THRESHOLD,
    verbose: bool = True,
) -> tuple[list[str], list[str], float]:
    """
    Iteratively drop the feature whose removal improves hold-out AUC
    the most. Stops when no single drop improves AUC by ≥ threshold.
    """
    keep = list(X.columns)
    cur_auc = evaluate_holdout(X[keep], y, verbose=False)["auc"]
    removed = []

    if verbose:
        print(f"  Baseline ({len(keep)} features): AUC = {cur_auc:.4f}\n")

    for rd in range(1, max_rounds + 1):
        best_drop: Optional[str] = None
        best_auc = cur_auc

        for feat in keep:
            trial = [f for f in keep if f != feat]
            auc_val = evaluate_holdout(X[trial], y, verbose=False)["auc"]
            if auc_val > best_auc + threshold:
                best_auc = auc_val
                best_drop = feat

        if best_drop is None:
            if verbose:
                print(f"  Round {rd}: no drop improves AUC by ≥{threshold}. STOP.")
            break

        keep.remove(best_drop)
        removed.append(best_drop)
        cur_auc = best_auc

        if verbose:
            print(
                f"  Round {rd}: drop {best_drop:30s} → "
                f"AUC: {best_auc:.4f}  ({len(keep)} left)"
            )

    if verbose:
        print(f"\n  Final AUC: {cur_auc:.4f}")
        print(f"  Kept ({len(keep)}):    {keep}")
        print(f"  Removed ({len(removed)}): {removed}")

    return keep, removed, cur_auc


def backward_elimination_beam(
    X: pd.DataFrame,
    y: pd.Series,
    max_rounds: int = MAX_BACKWARD_ROUNDS,
    threshold: float = ABLATION_IMPROVEMENT_THRESHOLD,
    beam_width: int = 8,
    children_per_state: int = 3,
    min_features: int = 5,
    scorer: str = "holdout",   # "holdout" or "fixed_8y2y"
    patience_rounds: int = 2,
    verbose: bool = True,
) -> tuple[list[str], list[str], float]:
    """
    Beam-search backward elimination (order-aware).

    Why this helps:
      - Greedy backward elimination commits to ONE drop order.
      - With correlated features, the best subset may require a different
        early drop that looks slightly worse short-term.
      - Beam search keeps multiple candidate paths alive per round.

    Returns (kept_features, removed_features_in_order, best_auc_like_metric).

    Notes
    -----
    - Uses memoization: same subset is evaluated once, even if reached via
      different drop orders.
    - 'threshold' is used for global-improvement stop logic, not per-step pruning.
    """
    feature_order = list(X.columns)
    if len(feature_order) == 0:
        raise ValueError("X has no columns.")
    if min_features < 1:
        raise ValueError("min_features must be >= 1.")
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1.")
    if children_per_state < 1:
        raise ValueError("children_per_state must be >= 1.")

    # -------- scoring with cache --------
    score_cache: dict[tuple[str, ...], float] = {}

    def _score_subset(cols_tuple: tuple[str, ...]) -> float:
        if cols_tuple in score_cache:
            return score_cache[cols_tuple]

        X_sub = X.loc[:, list(cols_tuple)]

        if scorer == "holdout":
            res = evaluate_holdout(X_sub, y, verbose=False)
            score = float(res["auc"])
        elif scorer == "fixed_8y2y":
            res = evaluate_fixed_year_windows(X_sub, y, verbose=False)
            score = float(res["test_auc"])
        else:
            raise ValueError("scorer must be 'holdout' or 'fixed_8y2y'.")

        score_cache[cols_tuple] = score
        return score

    # -------- initial state --------
    start_keep = tuple(feature_order)
    start_score = _score_subset(start_keep)

    beam = [
        {
            "keep": start_keep,
            "score": start_score,
            "removed": [],
        }
    ]
    best_state = {
        "keep": start_keep,
        "score": start_score,
        "removed": [],
    }

    if verbose:
        print(
            f"  Beam baseline ({len(start_keep)} features): "
            f"{scorer} AUC = {start_score:.4f}"
        )

    rounds_without_global_improve = 0

    # -------- beam search over drop depth --------
    for rd in range(1, max_rounds + 1):
        candidate_map: dict[tuple[str, ...], dict] = {}

        for state in beam:
            keep = state["keep"]

            if len(keep) <= min_features:
                continue

            local_children = []
            for feat in keep:
                child_keep = tuple(f for f in keep if f != feat)
                if len(child_keep) < min_features:
                    continue

                child_score = _score_subset(child_keep)
                local_children.append(
                    {
                        "keep": child_keep,
                        "score": child_score,
                        "removed": state["removed"] + [feat],
                        "parent_score": state["score"],
                        "dropped_now": feat,
                    }
                )

            if not local_children:
                continue

            # Keep only top-k children per parent to maintain path diversity
            local_children.sort(key=lambda d: d["score"], reverse=True)
            for child in local_children[:children_per_state]:
                k = child["keep"]
                prev = candidate_map.get(k)
                if (prev is None) or (child["score"] > prev["score"]):
                    candidate_map[k] = child

        if not candidate_map:
            if verbose:
                print(f"  Round {rd}: no more valid candidates. STOP.")
            break

        candidates = list(candidate_map.values())
        candidates.sort(key=lambda d: d["score"], reverse=True)

        next_beam = candidates[:beam_width]
        beam = [
            {"keep": c["keep"], "score": c["score"], "removed": c["removed"]}
            for c in next_beam
        ]

        round_best = beam[0]

        # Global-best tracking with threshold (allows temporary dips)
        if round_best["score"] > best_state["score"] + threshold:
            best_state = {
                "keep": round_best["keep"],
                "score": round_best["score"],
                "removed": list(round_best["removed"]),
            }
            rounds_without_global_improve = 0
            improved = True
        else:
            rounds_without_global_improve += 1
            improved = False

        if verbose:
            print(
                f"  Round {rd:2d}: best in round = {round_best['score']:.4f} "
                f"({len(round_best['keep'])} feats), "
                f"global best = {best_state['score']:.4f} "
                f"({len(best_state['keep'])} feats)"
                + ("  [improved]" if improved else "")
            )
            for i, st in enumerate(beam[: min(3, len(beam))], start=1):
                last_drop = st["removed"][-1] if st["removed"] else "-"
                print(
                    f"      beam#{i}: score={st['score']:.4f}, "
                    f"n={len(st['keep'])}, last_drop={last_drop}"
                )

        if rounds_without_global_improve >= patience_rounds:
            if verbose:
                print(
                    f"  Stop: no global improvement by ≥{threshold} for "
                    f"{patience_rounds} round(s)."
                )
            break

    kept = list(best_state["keep"])
    removed = list(best_state["removed"])
    best_score = float(best_state["score"])

    if verbose:
        print(f"\n  Final best {scorer} AUC: {best_score:.4f}")
        print(f"  Kept ({len(kept)}): {kept}")
        print(f"  Removed ({len(removed)}): {removed}")
        print(f"  Unique subsets evaluated: {len(score_cache)}")

    return kept, removed, best_score


# ══════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
def feature_importance(
    model: HistGradientBoostingClassifier,
    feature_names: list[str],
    X_eval: pd.DataFrame = None,
    y_eval: pd.Series = None,
) -> pd.DataFrame:
    """
    Compute feature importance via permutation importance on held-out data.

    Output columns: feature, gain, split, gain_pct
    (matches what visualize.py expects)
    """
    if X_eval is not None and y_eval is not None:
        X_eval_clean = X_eval.ffill().bfill()

        result = permutation_importance(
            model,
            X_eval_clean,
            y_eval,
            n_repeats=15,
            random_state=RANDOM_SEED,
            scoring="roc_auc",
            n_jobs=-1,
        )
        raw_importance = np.maximum(result.importances_mean, 0)
    else:
        raw_importance = np.ones(len(feature_names))

    total = raw_importance.sum()
    if total > 0:
        gain_pct = (raw_importance / total) * 100
        est_splits = (raw_importance / total) * 1000
    else:
        gain_pct = np.zeros(len(raw_importance))
        est_splits = np.zeros(len(raw_importance))

    imp = pd.DataFrame(
        {
            "feature": feature_names,
            "gain": raw_importance,
            "split": est_splits.astype(int),
            "gain_pct": np.round(gain_pct, 2),
        }
    )
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    return imp


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_all_data
    from features import build_features

    data = load_all_data()
    X, y = build_features(**data)

    print("\n" + "=" * 60)
    print("HOLD-OUT EVALUATION")
    print("=" * 60)
    result = evaluate_holdout(X, y)

    print("\nFeature importance (permutation):")
    imp = feature_importance(
        result["model"], list(X.columns),
        result["X_test"], result["y_test"],
    )
    print(imp.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("FIXED 8Y/2Y EVALUATION")
    print("=" * 60)
    result = evaluate_fixed_year_windows(
        X, y,
        start_date="2015-01-01",
        end_date="2025-01-01",
        train_years=8,
        test_years=2,
        verbose=True,
        )
