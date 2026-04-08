"""
model.py
--------
Builds, evaluates and saves a COVID-19 case forecasting model for India.

Architecture
    Baseline : Naive (last-known 7-day avg)
    Model 1  : Random Forest Regressor
    Model 2  : Gradient Boosting Regressor (sklearn)
    Evaluation: Time-series walk-forward split on last 90 days as test set.

Features used
    Lag features (7, 14, 21 days), rolling mean/std, growth rate,
    day-of-week, month, vaccination coverage, stringency index,
    and Google Mobility indicators.

Run:
    python src/model.py
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "..", "figures")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE  = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#888888"]
TEST_DAYS = 90          # held-out test window
FORECAST_DAYS = 30      # forward forecast horizon


# ─────────────────────────────────────────────────────────────────────────────
def load() -> pd.DataFrame:
    path = os.path.join(PROC_DIR, "covid_india_processed.csv")
    if not os.path.exists(path):
        log.info("Processed file missing – running data_preparation …")
        import sys; sys.path.insert(0, os.path.dirname(__file__))
        from data_preparation import run
        return run()
    return pd.read_csv(path, parse_dates=["date"])


def build_feature_matrix(df: pd.DataFrame):
    """
    Select and clean feature matrix X and target y.
    Returns X (DataFrame), y (Series), feature_names.
    """
    feature_cols = [
        # Lag features
        "cases_lag7", "cases_lag14", "cases_lag21",
        # Rolling stats
        "cases_roll14_mean", "cases_roll14_std",
        # Growth dynamics
        "growth_rate_7d",
        # Temporal
        "day_of_week", "month", "is_weekend", "week_of_year",
        # Epidemiological
        "positivity_rate", "stringency_index",
        # Vaccination
        "vacc_coverage",
        # Mobility
        "retail_and_recreation_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    target = "cases_7d_avg"
    sub = df[["date", target] + feature_cols].copy()

    # Fill remaining NaN in features with median
    for col in feature_cols:
        sub[col] = sub[col].fillna(sub[col].median())

    sub = sub.dropna(subset=[target]).reset_index(drop=True)
    X = sub[feature_cols]
    y = sub[target]
    dates = sub["date"]
    return X, y, dates, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
def mape(y_true, y_pred, eps=1.0):
    """Mean Absolute Percentage Error (avoids divide-by-zero)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true > eps
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mp   = mape(y_true, y_pred)
    metrics = dict(MAE=round(mae, 2), RMSE=round(rmse, 2), R2=round(r2, 4), MAPE=round(mp, 2))
    log.info(f"{name:30s} | MAE={mae:,.0f}  RMSE={rmse:,.0f}  R²={r2:.4f}  MAPE={mp:.1f}%")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame):
    X, y, dates, feature_cols = build_feature_matrix(df)
    n = len(X)
    split = n - TEST_DAYS

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    d_train, d_test = dates.iloc[:split], dates.iloc[split:]

    log.info(f"Train: {d_train.iloc[0].date()} → {d_train.iloc[-1].date()}  ({split} days)")
    log.info(f"Test : {d_test.iloc[0].date()}  → {d_test.iloc[-1].date()}  ({TEST_DAYS} days)")

    # ── Naive baseline: predict last known 7-day avg ──────────────────────────
    naive_pred = np.full(len(y_test), y_train.iloc[-1])
    metrics = {"Naive Baseline": evaluate("Naive Baseline", y_test.values, naive_pred)}

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("model", RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            max_features="sqrt", random_state=42, n_jobs=-1,
        ))
    ])
    rf_pipe.fit(X_train, y_train)
    rf_pred = rf_pipe.predict(X_test).clip(0)
    metrics["Random Forest"] = evaluate("Random Forest", y_test.values, rf_pred)

    # ── Gradient Boosting ─────────────────────────────────────────────────────
    gb_pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        ))
    ])
    gb_pipe.fit(X_train, y_train)
    gb_pred = gb_pipe.predict(X_test).clip(0)
    metrics["Gradient Boosting"] = evaluate("Gradient Boosting", y_test.values, gb_pred)

    results = {
        "dates_test": d_test,
        "y_test": y_test,
        "y_train": y_train,
        "d_train": d_train,
        "naive_pred": naive_pred,
        "rf_pred": rf_pred,
        "gb_pred": gb_pred,
        "rf_pipe": rf_pipe,
        "gb_pipe": gb_pipe,
        "X_test": X_test,
        "feature_cols": feature_cols,
        "metrics": metrics,
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
def feature_importance_plot(pipe, feature_cols, model_name="gradient_boosting"):
    model = pipe.named_steps["model"]
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:15]
    top_features = [feature_cols[i] for i in idx]
    top_imp      = importances[idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(range(len(top_features)), top_imp[::-1], color=PALETTE[0], alpha=0.85)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([f.replace("_percent_change_from_baseline", "\n(mobility)") for f in top_features[::-1]], fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title(f"Top-15 Feature Importances – {model_name.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"05_feature_importance_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {path}")


def plot_predictions(results: dict):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    d_test = results["dates_test"]
    y_test = results["y_test"]

    # ── Panel 1: full series + test window ───────────────────────────────────
    ax = axes[0]
    all_dates = pd.concat([results["d_train"], d_test])
    all_y     = pd.concat([results["y_train"], y_test])
    ax.plot(all_dates, all_y, color="grey", lw=1, alpha=0.5, label="Actual (all)")
    ax.axvspan(d_test.iloc[0], d_test.iloc[-1], alpha=0.07, color="orange", label="Test window")
    ax.plot(d_test, results["rf_pred"],  color=PALETTE[1], lw=2, ls="--", label="Random Forest")
    ax.plot(d_test, results["gb_pred"],  color=PALETTE[2], lw=2, ls="-",  label="Gradient Boosting")
    ax.plot(d_test, y_test.values,       color=PALETTE[0], lw=2,           label="Actual (test)")
    ax.set_title("COVID-19 India – Model Predictions vs Actual (Full Timeline)", fontsize=12, fontweight="bold")
    ax.set_ylabel("7-day Avg New Cases")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # ── Panel 2: zoomed test window ──────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(d_test, y_test.values,       color=PALETTE[0], lw=2.5,         label="Actual")
    ax2.plot(d_test, results["naive_pred"],color="grey",     lw=1.5, ls=":", label="Naive Baseline")
    ax2.plot(d_test, results["rf_pred"],   color=PALETTE[1], lw=2, ls="--",  label="Random Forest")
    ax2.plot(d_test, results["gb_pred"],   color=PALETTE[2], lw=2,           label="Gradient Boosting")
    ax2.set_title("Test Window – Zoomed View", fontsize=12, fontweight="bold")
    ax2.set_ylabel("7-day Avg New Cases")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "06_predictions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {path}")


def plot_residuals(results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    y_test = results["y_test"].values
    for ax, preds, name, color in [
        (axes[0], results["rf_pred"],  "Random Forest",     PALETTE[1]),
        (axes[1], results["gb_pred"],  "Gradient Boosting", PALETTE[2]),
    ]:
        resid = y_test - preds
        ax.scatter(preds, resid, alpha=0.4, s=20, color=color)
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Residual", fontsize=10)
        ax.set_title(f"{name} – Residuals", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "07_residuals.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {path}")


def plot_metrics_comparison(metrics: dict):
    """Bar chart comparing model metrics."""
    models = list(metrics.keys())
    metric_names = ["MAE", "RMSE", "R2", "MAPE"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    colors = [PALETTE[4], PALETTE[1], PALETTE[2]]

    for ax, mn in zip(axes, metric_names):
        vals = [metrics[m].get(mn, 0) for m in models]
        bars = ax.bar(models, vals, color=colors, alpha=0.85, width=0.5)
        ax.set_title(mn, fontsize=11, fontweight="bold")
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f"{val:.2f}", ha="center", fontsize=8)
        if mn == "R2":
            ax.set_ylim(-0.2, 1.1)
    fig.suptitle("Model Performance Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "08_metrics_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {path}")


def save_metrics(metrics: dict):
    path = os.path.join(OUT_DIR, "model_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved → {path}")

    # Also save as CSV
    rows = []
    for model, m in metrics.items():
        rows.append({"Model": model, **m})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "model_metrics.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
def run():
    df = load()
    log.info(f"Dataset: {len(df):,} rows | {df['date'].min().date()} → {df['date'].max().date()}")

    results = train_and_evaluate(df)

    feature_importance_plot(results["rf_pipe"], results["feature_cols"], "random_forest")
    feature_importance_plot(results["gb_pipe"], results["feature_cols"], "gradient_boosting")
    plot_predictions(results)
    plot_residuals(results)
    plot_metrics_comparison(results["metrics"])
    save_metrics(results["metrics"])

    log.info("\n=== Final Model Metrics ===")
    for model, m in results["metrics"].items():
        log.info(f"  {model}: {m}")

    return results


if __name__ == "__main__":
    run()
