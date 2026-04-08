"""
eda.py
------
Exploratory Data Analysis and visualisation for COVID-19 India dataset.

Run:
    python src/eda.py
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)


def load() -> pd.DataFrame:
    path = os.path.join(PROC_DIR, "covid_india_processed.csv")
    if not os.path.exists(path):
        log.info("Processed file not found – running data_preparation …")
        from data_preparation import run
        return run()
    df = pd.read_csv(path); df["date"] = pd.to_datetime(df["date"]); return df


# ─────────────────────────────────────────────────────────────────────────────
def plot_temporal_overview(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("COVID-19 India – Temporal Overview", fontsize=15, fontweight="bold")

    # Cases
    axes[0].fill_between(df["date"], df["new_cases"], alpha=0.3, color=PALETTE[0])
    axes[0].plot(df["date"], df["cases_7d_avg"], color=PALETTE[0], lw=2, label="7-day avg")
    axes[0].set_ylabel("New Cases", fontsize=11)
    axes[0].legend(loc="upper left")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

    # Deaths
    axes[1].fill_between(df["date"], df["new_deaths"].fillna(0), alpha=0.3, color=PALETTE[1])
    axes[1].plot(df["date"], df["deaths_7d_avg"].fillna(0), color=PALETTE[1], lw=2, label="7-day avg")
    axes[1].set_ylabel("New Deaths", fontsize=11)
    axes[1].legend(loc="upper left")

    # Stringency / Mobility
    if "stringency_index" in df.columns:
        ax3 = axes[2]
        ax3.plot(df["date"], df["stringency_index"], color=PALETTE[3], lw=2, label="Stringency Index")
        ax3.set_ylabel("Stringency Index", fontsize=11)
        ax3.legend(loc="upper left")
        ax3.set_ylim(0, 100)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "01_temporal_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {path}")


def plot_wave_analysis(df: pd.DataFrame):
    """Annotate the three pandemic waves."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(df["date"], df["cases_7d_avg"], alpha=0.25, color=PALETTE[0])
    ax.plot(df["date"], df["cases_7d_avg"], color=PALETTE[0], lw=2)

    waves = [
        ("Wave 1\n(Sep 2020)", "2020-09-15", 0.65),
        ("Wave 2 – Delta\n(May 2021)", "2021-05-10", 0.98),
        ("Wave 3 – Omicron\n(Jan 2022)", "2022-01-25", 0.72),
    ]
    ymax = df["cases_7d_avg"].max()
    for label, dt, frac in waves:
        x = pd.Timestamp(dt)
        ax.axvline(x, ls="--", color="grey", alpha=0.6)
        ax.text(x, ymax * frac, label, ha="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_title("COVID-19 India – Pandemic Waves", fontsize=13, fontweight="bold")
    ax.set_ylabel("7-day Avg New Cases")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "02_wave_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {path}")


def plot_mobility_correlation(df: pd.DataFrame):
    mob_cols = [c for c in df.columns if "percent_change" in c]
    if not mob_cols:
        return
    short = {c: c.replace("_percent_change_from_baseline", "").replace("_", " ").title()
              for c in mob_cols}
    sub = df[["cases_7d_avg"] + mob_cols].dropna()
    sub = sub.rename(columns=short)
    sub = sub.rename(columns={"cases_7d_avg": "Cases (7d avg)"})

    fig, axes = plt.subplots(1, len(mob_cols), figsize=(5 * len(mob_cols), 4))
    if len(mob_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, [short[c] for c in mob_cols]):
        ax.scatter(sub[col], sub["Cases (7d avg)"], alpha=0.25, s=10, color=PALETTE[0])
        m, b = np.polyfit(sub[col].fillna(0), sub["Cases (7d avg)"].fillna(0), 1)
        xr = np.linspace(sub[col].min(), sub[col].max(), 100)
        ax.plot(xr, m * xr + b, color=PALETTE[1], lw=2)
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Cases (7d avg)")
        r = np.corrcoef(sub[col].fillna(0), sub["Cases (7d avg)"].fillna(0))[0, 1]
        ax.set_title(f"r = {r:.2f}", fontsize=10)

    fig.suptitle("Mobility vs COVID Cases (India)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "03_mobility_correlation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {path}")


def plot_feature_heatmap(df: pd.DataFrame):
    feature_cols = [
        "cases_7d_avg", "cases_lag7", "cases_lag14", "cases_lag21",
        "growth_rate_7d", "positivity_rate", "stringency_index",
        "vacc_coverage",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    corr = df[feature_cols].dropna().corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5, square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "04_feature_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {path}")


def run():
    df = load()
    log.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")
    log.info(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    plot_temporal_overview(df)
    plot_wave_analysis(df)
    plot_mobility_correlation(df)
    plot_feature_heatmap(df)
    log.info("EDA complete.")
    return df


if __name__ == "__main__":
    run()
