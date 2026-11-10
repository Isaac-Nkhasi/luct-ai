"""Phase 9 — evaluation metrics and plots.

These helpers are UI-agnostic: they return numbers and matplotlib figures so the
notebooks, the report and (later) the dashboard can reuse the exact same
definitions. The dashboard must show stored/computed metrics — never
hand-written numbers — which is why everything is computed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

try:  # scikit-learn >= 1.4
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
except ImportError:  # pragma: no cover - fallback for older scikit-learn
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    def root_mean_squared_error(y_true, y_pred):  # type: ignore[misc]
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(
    y_true: Sequence[float] | pd.Series,
    y_pred: Sequence[float] | pd.Series,
) -> dict[str, float | int]:
    """Return MAE, RMSE, R^2 and the number of scored records.

    MAE and RMSE are in the target currency (maloti); R^2 is unitless.
    """
    y_true_arr = np.asarray(y_true, dtype="float64")
    y_pred_arr = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr, y_pred_arr = y_true_arr[mask], y_pred_arr[mask]
    if y_true_arr.size == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    return {
        "n": int(y_true_arr.size),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(root_mean_squared_error(y_true_arr, y_pred_arr)),
        "r2": float(r2_score(y_true_arr, y_pred_arr)) if y_true_arr.size > 1 else float("nan"),
    }


def comparison_table(
    predictions: Mapping[str, tuple[Sequence[float], Sequence[float]]],
) -> pd.DataFrame:
    """Build the model comparison table.

    ``predictions`` maps a model name to ``(y_true, y_pred)``.
    """
    rows = []
    for name, (y_true, y_pred) in predictions.items():
        metrics = regression_metrics(y_true, y_pred)
        rows.append({"model": name, **metrics})
    return pd.DataFrame(rows, columns=["model", "n", "mae", "rmse", "r2"])


def error_by_band(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    bands: Sequence[object],
) -> pd.DataFrame:
    """MAE/RMSE/count per price band (or any grouping), sorted by band."""
    frame = pd.DataFrame(
        {"y_true": np.asarray(y_true, dtype="float64"),
         "y_pred": np.asarray(y_pred, dtype="float64"),
         "band": np.asarray(bands, dtype=object)}
    )
    rows = []
    for band, group in frame.groupby("band", dropna=False, observed=True):
        metrics = regression_metrics(group["y_true"], group["y_pred"])
        rows.append({"band": str(band), "n": metrics["n"],
                     "mae": metrics["mae"], "rmse": metrics["rmse"]})
    return pd.DataFrame(rows, columns=["band", "n", "mae", "rmse"])


@dataclass
class LearningCurve:
    train_sizes: np.ndarray
    train_mae: np.ndarray
    train_mae_std: np.ndarray
    val_mae: np.ndarray
    val_mae_std: np.ndarray

    @property
    def is_empty(self) -> bool:
        return self.train_sizes.size == 0


def learning_curve_data(
    estimator,
    X,
    y,
    *,
    groups: Sequence[object] | None = None,
    cv_splits: int = 5,
    train_sizes: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
) -> LearningCurve:
    """Compute a learning curve scored by MAE (lower is better).

    Uses grouped CV when ``groups`` is supplied so the curve is not inflated by
    the same laptop spec appearing on both sides of a fold.
    """
    from sklearn.model_selection import GroupKFold, KFold, learning_curve

    empty = LearningCurve(
        np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    )
    y_arr = np.asarray(y, dtype="float64")
    if y_arr.size < 10:
        return empty

    cv_groups = None
    if groups is not None and pd.Series(groups).nunique() >= 2:
        n_groups = pd.Series(groups).nunique()
        splitter = GroupKFold(n_splits=min(cv_splits, n_groups))
        cv_groups = groups
    else:
        splitter = KFold(n_splits=min(cv_splits, max(2, y_arr.size // 2)), shuffle=True,
                         random_state=42)

    try:
        sizes, train_scores, val_scores = learning_curve(
            estimator, X, y_arr, groups=cv_groups, cv=splitter,
            scoring="neg_mean_absolute_error", train_sizes=list(train_sizes), n_jobs=None,
        )
    except ValueError:
        # Not enough samples per fold for a stable curve.
        return empty
    return LearningCurve(
        train_sizes=sizes,
        train_mae=-train_scores.mean(axis=1),
        train_mae_std=train_scores.std(axis=1),
        val_mae=-val_scores.mean(axis=1),
        val_mae_std=val_scores.std(axis=1),
    )


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def actual_vs_predicted_plot(
    y_true: Sequence[float], y_pred: Sequence[float], *, title: str = "Actual vs predicted"
):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    y_true_arr, y_pred_arr = np.asarray(y_true, float), np.asarray(y_pred, float)
    ax.scatter(y_true_arr, y_pred_arr, s=18, alpha=0.6, edgecolor="none")
    if y_true_arr.size:
        low = float(min(y_true_arr.min(), y_pred_arr.min()))
        high = float(max(y_true_arr.max(), y_pred_arr.max()))
        ax.plot([low, high], [low, high], color="#d64545", linewidth=1.2, linestyle="--",
                label="perfect prediction")
        ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("Actual price (LSL)")
    ax.set_ylabel("Predicted price (LSL)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def error_by_band_plot(band_table: pd.DataFrame, *, title: str = "Prediction error by price band"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    if not band_table.empty:
        ax.bar(band_table["band"].astype(str), band_table["mae"], color="#4c78a8")
        ax.set_ylabel("MAE (LSL)")
    ax.set_xlabel("Price band")
    ax.set_title(title)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def learning_curve_plot(curve: LearningCurve, *, title: str = "Learning curve"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    if not curve.is_empty:
        ax.plot(curve.train_sizes, curve.train_mae, marker="o", label="Training MAE")
        ax.fill_between(curve.train_sizes, curve.train_mae - curve.train_mae_std,
                        curve.train_mae + curve.train_mae_std, alpha=0.15)
        ax.plot(curve.train_sizes, curve.val_mae, marker="s", label="Validation MAE")
        ax.fill_between(curve.train_sizes, curve.val_mae - curve.val_mae_std,
                        curve.val_mae + curve.val_mae_std, alpha=0.15)
        ax.legend(fontsize=8)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("MAE (LSL)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def feature_importance_plot(importances: pd.Series, *, top_n: int = 20,
                            title: str = "Feature importance"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, max(3.0, 0.28 * min(top_n, len(importances)))))
    if not importances.empty:
        top = importances.sort_values(ascending=True).tail(top_n)
        ax.barh(top.index.astype(str), top.to_numpy(), color="#59a14f")
        ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def save_figure(fig, path: str | Path, *, dpi: int = 140) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
