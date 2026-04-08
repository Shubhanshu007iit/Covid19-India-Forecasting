"""
data_preparation.py
-------------------
Downloads and pre-processes the COVID-19 dataset (Our World in Data) 
for India at the state/national level, merges Google Mobility data,
and produces a single cleaned Parquet file ready for modelling.

Run:
    python src/data_preparation.py
"""

import os
import io
import logging
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

# ── Source URLs ────────────────────────────────────────────────────────────────
OWID_URL = (
    "https://covid.ourworldindata.org/data/owid-covid-data.csv"
)
MOBILITY_URL = (
    "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
)


# ─────────────────────────────────────────────────────────────────────────────
def _download_or_cache(url: str, fname: str) -> pd.DataFrame:
    """Download CSV if not cached; return DataFrame."""
    fpath = os.path.join(RAW_DIR, fname)
    if os.path.exists(fpath):
        log.info(f"Loading cached file: {fname}")
        return pd.read_csv(fpath, low_memory=False)
    log.info(f"Downloading {fname} …")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, fpath)
        df = pd.read_csv(fpath, low_memory=False)
        log.info(f"Saved {fname} ({len(df):,} rows)")
        return df
    except Exception as exc:
        log.warning(f"Download failed ({exc}). Generating synthetic data for {fname}.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_covid(start="2020-01-30", end="2023-12-31") -> pd.DataFrame:
    """
    Generate realistic synthetic India COVID-19 data when network is unavailable.
    Models three pandemic waves (Alpha, Delta, Omicron) at national level.
    """
    log.info("Generating synthetic COVID-19 data for India …")
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    rng = np.random.default_rng(42)

    # --- three wave kernels (log-normal shaped) --------------------------------
    def wave(peak_day, sigma, height):
        t = np.arange(n)
        return height * np.exp(-((t - peak_day) ** 2) / (2 * sigma ** 2))

    # Wave 1: ~Apr-Jun 2020  Wave 2 (Delta): ~Apr-Jun 2021  Wave 3 (Omicron): Jan-Feb 2022
    day0 = pd.Timestamp(start)
    p1 = (pd.Timestamp("2020-09-15") - day0).days
    p2 = (pd.Timestamp("2021-05-10") - day0).days
    p3 = (pd.Timestamp("2022-01-20") - day0).days

    cases_smooth = (
        wave(p1, 45, 95_000) +
        wave(p2, 35, 400_000) +
        wave(p3, 28, 280_000)
    ).clip(0)

    noise = rng.negative_binomial(10, 0.5, n) * 50
    new_cases = (cases_smooth + noise).clip(0).round().astype(int)

    # Deaths ~1.2% CFR with 14-day lag
    lag = 14
    deaths_smooth = np.concatenate([np.zeros(lag), cases_smooth[:-lag]]) * 0.012
    new_deaths = (deaths_smooth + rng.negative_binomial(5, 0.5, n) * 2).clip(0).round().astype(int)

    # Tests: ~15x cases with weekend dip
    tests = (new_cases * 15 * (0.8 + 0.2 * rng.random(n))).round().astype(int)
    tests[dates.dayofweek >= 5] = (tests[dates.dayofweek >= 5] * 0.7).astype(int)

    vaccinations = np.zeros(n)
    vacc_start = (pd.Timestamp("2021-01-16") - day0).days
    if vacc_start < n:
        ramp = np.linspace(0, 4_500_000, n - vacc_start)
        vaccinations[vacc_start:] = ramp + rng.normal(0, 100_000, n - vacc_start)
        vaccinations = vaccinations.clip(0)

    df = pd.DataFrame({
        "date": dates,
        "location": "India",
        "iso_code": "IND",
        "new_cases": new_cases,
        "new_deaths": new_deaths,
        "new_tests": tests,
        "new_vaccinations": vaccinations.round().astype(int),
        "population": 1_380_004_385,
        "stringency_index": np.nan,   # filled below
    })

    # Crude stringency: high near peaks, tapers
    str_smooth = (
        wave(p1, 60, 80) +
        wave(p2, 40, 60) +
        wave(p3, 35, 50)
    ).clip(0, 100)
    df["stringency_index"] = str_smooth.round(1)
    return df


def generate_synthetic_mobility() -> pd.DataFrame:
    """Generate synthetic Google Mobility data for India."""
    log.info("Generating synthetic mobility data for India …")
    dates = pd.date_range("2020-02-15", "2022-10-15", freq="D")
    n = len(dates)
    rng = np.random.default_rng(99)
    day0 = pd.Timestamp("2020-02-15")

    def wave(peak_day, sigma, depth):
        t = np.arange(n)
        return depth * np.exp(-((t - peak_day) ** 2) / (2 * sigma ** 2))

    p1 = (pd.Timestamp("2020-04-15") - day0).days
    p2 = (pd.Timestamp("2021-05-10") - day0).days

    base = np.zeros(n)
    retail = base - wave(p1, 50, 70) - wave(p2, 40, 45) + rng.normal(0, 3, n)
    transit = base - wave(p1, 50, 65) - wave(p2, 40, 40) + rng.normal(0, 3, n)
    workplaces = base - wave(p1, 50, 55) - wave(p2, 40, 35) + rng.normal(0, 3, n)
    residential = base + wave(p1, 50, 25) + wave(p2, 40, 18) + rng.normal(0, 2, n)

    # Weekend effect
    dow = dates.dayofweek.values
    retail[dow >= 5] += 5
    workplaces[dow >= 5] -= 20

    df = pd.DataFrame({
        "date": dates,
        "country_region": "India",
        "country_region_code": "IN",
        "sub_region_1": np.nan,
        "retail_and_recreation_percent_change_from_baseline": retail.clip(-100, 50).round(1),
        "transit_stations_percent_change_from_baseline": transit.clip(-100, 50).round(1),
        "workplaces_percent_change_from_baseline": workplaces.clip(-100, 50).round(1),
        "residential_percent_change_from_baseline": residential.clip(-30, 50).round(1),
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
def load_covid_data() -> pd.DataFrame:
    raw = _download_or_cache(OWID_URL, "owid_covid.csv")
    if raw is None:
        return generate_synthetic_covid()

    india = raw[raw["iso_code"] == "IND"].copy()
    keep = [
        "date", "location", "iso_code",
        "new_cases", "new_deaths", "new_tests",
        "new_vaccinations", "population", "stringency_index",
    ]
    india = india[[c for c in keep if c in india.columns]].copy()
    india["date"] = pd.to_datetime(india["date"])
    return india.sort_values("date").reset_index(drop=True)


def load_mobility_data() -> pd.DataFrame:
    raw = _download_or_cache(MOBILITY_URL, "google_mobility.csv")
    if raw is None:
        return generate_synthetic_mobility()

    mob = raw[
        (raw["country_region_code"] == "IN") &
        (raw["sub_region_1"].isna())
    ].copy()
    mob_cols = [
        "date", "country_region", "country_region_code",
        "retail_and_recreation_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
    ]
    mob = mob[[c for c in mob_cols if c in mob.columns]].copy()
    mob["date"] = pd.to_datetime(mob["date"])
    return mob.sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
def preprocess(covid: pd.DataFrame, mobility: pd.DataFrame) -> pd.DataFrame:
    """
    Merge COVID + Mobility data, impute, and engineer features.
    Returns a clean DataFrame indexed by date.
    """
    log.info("Merging datasets …")
    df = covid.merge(mobility, on="date", how="left")

    # ── Clip negatives (reporting corrections) ────────────────────────────────
    for col in ["new_cases", "new_deaths", "new_tests", "new_vaccinations"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # ── 7-day rolling smoothing for case series ────────────────────────────────
    df = df.sort_values("date").reset_index(drop=True)
    df["cases_7d_avg"] = df["new_cases"].rolling(7, min_periods=1).mean()
    df["deaths_7d_avg"] = df["new_deaths"].rolling(7, min_periods=1).mean()

    # ── Positivity rate ────────────────────────────────────────────────────────
    df["positivity_rate"] = np.where(
        df.get("new_tests", pd.Series(np.nan, index=df.index)).fillna(0) > 0,
        df["new_cases"] / df["new_tests"].replace(0, np.nan),
        np.nan,
    )
    df["positivity_rate"] = df["positivity_rate"].clip(0, 1)

    # ── Temporal features ──────────────────────────────────────────────────────
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # ── Lag features (7, 14, 21 days) ────────────────────────────────────────
    for lag in [7, 14, 21]:
        df[f"cases_lag{lag}"] = df["cases_7d_avg"].shift(lag)

    # ── Rolling statistics ────────────────────────────────────────────────────
    df["cases_roll14_mean"] = df["cases_7d_avg"].shift(1).rolling(14, min_periods=7).mean()
    df["cases_roll14_std"] = df["cases_7d_avg"].shift(1).rolling(14, min_periods=7).std()

    # ── Growth rate ───────────────────────────────────────────────────────────
    df["growth_rate_7d"] = df["cases_7d_avg"].pct_change(7).replace([np.inf, -np.inf], np.nan)

    # ── Vaccination coverage (cumulative % of pop) ────────────────────────────
    if "new_vaccinations" in df.columns and "population" in df.columns:
        df["vacc_cumulative"] = df["new_vaccinations"].fillna(0).cumsum()
        df["vacc_coverage"] = df["vacc_cumulative"] / df["population"].fillna(1.4e9)
    else:
        df["vacc_coverage"] = 0.0

    # ── Mobility: forward-fill (weekends/holidays have no data) ───────────────
    mob_cols = [c for c in df.columns if "percent_change" in c]
    for col in mob_cols:
        df[col] = df[col].ffill().bfill().fillna(0)

    # ── Stringency ────────────────────────────────────────────────────────────
    if "stringency_index" in df.columns:
        df["stringency_index"] = df["stringency_index"].ffill().bfill().fillna(0)

    # ── Drop rows without target ──────────────────────────────────────────────
    df = df.dropna(subset=["cases_7d_avg"]).reset_index(drop=True)

    out_path = os.path.join(PROC_DIR, "covid_india_processed.csv")
    df.to_csv(out_path.replace(".parquet", ".csv"), index=False)
    log.info(f"Processed data saved → {out_path}  ({len(df):,} rows, {df.shape[1]} cols)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def run():
    covid = load_covid_data()
    mobility = load_mobility_data()
    df = preprocess(covid, mobility)
    log.info("Data preparation complete.")
    return df


if __name__ == "__main__":
    run()
