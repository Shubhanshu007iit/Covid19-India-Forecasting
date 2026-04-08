"""
main.py
-------
Entry point – runs the full COVID-19 forecasting pipeline:
  1. Data preparation
  2. EDA / visualisations
  3. Model training & evaluation

Usage:
    python main.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def main():
    log.info("=" * 60)
    log.info("  COVID-19 India Forecasting Pipeline")
    log.info("=" * 60)

    # Step 1: Data preparation
    log.info("\n[1/3] Data Preparation …")
    from data_preparation import run as prep_run
    df = prep_run()

    # Step 2: EDA
    log.info("\n[2/3] Exploratory Data Analysis …")
    from eda import run as eda_run
    eda_run()

    # Step 3: Model
    log.info("\n[3/3] Model Training & Evaluation …")
    from model import run as model_run
    results = model_run()

    log.info("\n" + "=" * 60)
    log.info("  Pipeline complete.")
    log.info("  Figures  → figures/")
    log.info("  Outputs  → outputs/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
