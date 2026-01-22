from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class BacktestConfig:
    """
    Backtest configuration (sanitized, no sensitive paths).

    Notes:
    - Provide file paths relative to repo or via environment variables in scripts.
    - All dates are interpreted in local timezone-naive datetime.
    """
    start_date: datetime
    end_date: datetime

    # Inputs
    benchmark_csv: Optional[Path] = None          # CSV with columns: TradingDay, <benchmark_return_column>
    benchmark_return_column: str = "next_ret"     # e.g. "next_ret"

    # Outputs
    output_dir: Path = Path("outputs")
    version_tag: str = "v1"

    # Costs
    transaction_fee_rate: float = 0.001  # per unit turnover

    # Trading calendar assumption
    trading_days_per_year: int = 242

    # Output filenames
    metrics_filename: str = "backtest_metrics.csv"
    yearly_filename: str = "yearly_performance.csv"
    selected_stocks_filename: str = "selected_stocks.csv"
