from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_parquet(path: Path) -> pd.DataFrame:
    logger.info("Loading parquet: %s", path)
    return pd.read_parquet(path)


def load_benchmark_csv(path: Path, trading_day_col: str = "TradingDay") -> pd.DataFrame:
    """
    Expected columns:
      - TradingDay
      - benchmark_return_column (e.g., next_ret)
    """
    logger.info("Loading benchmark CSV: %s", path)
    df = pd.read_csv(path)
    df[trading_day_col] = pd.to_datetime(df[trading_day_col])
    df = df.sort_values(trading_day_col).set_index(trading_day_col)
    return df


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_series_csv(series: pd.Series, path: Path, encoding: str = "utf-8-sig") -> None:
    logger.info("Saving series to: %s", path)
    series.to_csv(path, encoding=encoding)


def save_df_csv(df: pd.DataFrame, path: Path, encoding: str = "utf-8-sig") -> None:
    logger.info("Saving dataframe to: %s", path)
    df.to_csv(path, encoding=encoding, index=True)


def save_df_csv_noindex(df: pd.DataFrame, path: Path, encoding: str = "utf-8-sig") -> None:
    logger.info("Saving dataframe (no index) to: %s", path)
    df.to_csv(path, encoding=encoding, index=False)
