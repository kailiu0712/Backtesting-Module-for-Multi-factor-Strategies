from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import BacktestConfig
from io import load_benchmark_csv, safe_mkdir, save_series_csv, save_df_csv, save_df_csv_noindex
from metrics import (
    cumulative_return_simple,
    annual_return,
    annual_volatility,
    information_ratio_1,
    information_ratio_2,
    win_probability,
    max_drawdown_simple,
    profit_loss_ratio,
    format_metrics,
)
from plotting import plot_net_value, plot_daily_holdings_count, plot_pnl_and_value

logger = logging.getLogger(__name__)


def _compute_portfolio_returns(selected_df: pd.DataFrame) -> pd.Series:
    # sum(weights * next_ret) per day
    return selected_df.groupby("TradingDay").apply(lambda x: (x["next_ret"] * x["weights"]).sum())


def _compute_turnover(all_df: pd.DataFrame) -> pd.Series:
    # absolute weight change per security, summed by day, divided by 2
    all_df = all_df.copy()
    all_df["position_change"] = all_df.groupby("SecuCode")["weights"].diff().abs()
    return all_df.groupby("TradingDay")["position_change"].sum() / 2.0


def _build_selected_stocks_table(original_df: pd.DataFrame) -> pd.DataFrame:
    selected_df = original_df[original_df["weights"] > 0].copy()

    def combine(group: pd.DataFrame) -> str:
        return ", ".join([f"{code}:{w:.4f}" for code, w in zip(group["SecuCode"], group["weights"])])

    out = selected_df.groupby("TradingDay").apply(combine).reset_index()
    out.columns = ["TradingDay", "selected_stocks"]
    return out


def run_backtest(
    df: pd.DataFrame,
    cfg: BacktestConfig,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - formatted_metrics: pd.Series (strings)
      - yearly_performance: pd.DataFrame
      - result_timeseries: pd.DataFrame indexed by TradingDay
      - original_df_filtered: pd.DataFrame filtered to date range (for diagnostics/exports)
    """
    required = ["TradingDay", "SecuCode", "next_ret", "weights"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for backtest: {missing}")

    logger.info("Filtering to backtest window...")
    x = df.copy()
    x["TradingDay"] = pd.to_datetime(x["TradingDay"])
    x = x[(x["TradingDay"] >= cfg.start_date) & (x["TradingDay"] <= cfg.end_date)].copy()

    selected = x[x["weights"] > 0].copy()
    logger.info("Selected rows: %d", len(selected))

    logger.info("Computing portfolio returns...")
    portfolio_returns = _compute_portfolio_returns(selected)  # indexed by TradingDay

    logger.info("Computing turnover and applying transaction costs...")
    turnover = _compute_turnover(x)
    adjusted_returns = portfolio_returns - cfg.transaction_fee_rate * turnover

    daily_n_holdings = selected.groupby("TradingDay").size()

    logger.info("Building equity curves (simple return cumulation)...")
    strategy_value = 1.0 + adjusted_returns.cumsum()

    # Benchmark
    benchmark = None
    if cfg.benchmark_csv is not None:
        bench_df = load_benchmark_csv(cfg.benchmark_csv)
        if cfg.benchmark_return_column not in bench_df.columns:
            raise ValueError(
                f"Benchmark column '{cfg.benchmark_return_column}' not found in {cfg.benchmark_csv}"
            )
        benchmark = bench_df.loc[(bench_df.index >= cfg.start_date) & (bench_df.index <= cfg.end_date)].copy()
        benchmark["PctChange"] = benchmark[cfg.benchmark_return_column]

    if benchmark is not None and not benchmark.empty:
        excess_daily = adjusted_returns - benchmark["PctChange"]
        base_value = 1.0 + benchmark["PctChange"].cumsum()
        excess_value = strategy_value - base_value
        excess_value_rel = excess_value / base_value.replace(0.0, np.nan)
    else:
        excess_daily = pd.Series(index=adjusted_returns.index, data=np.nan)
        base_value = None
        excess_value = None
        excess_value_rel = None

    # Build timeseries table
    result = pd.DataFrame(
        {
            "strategy_return": adjusted_returns,
            "strategy_value": strategy_value,
            "portfolio_return": portfolio_returns,
            "turnover": turnover,
            "n_holdings": daily_n_holdings,
        }
    )
    if base_value is not None:
        result["base_value"] = base_value
        result["excess_value"] = excess_value
        result["excess_value_relative"] = excess_value_rel

    # Metrics
    tdpy = cfg.trading_days_per_year
    metrics_raw = {
        "cum_ret": cumulative_return_simple(adjusted_returns),
        "ann_ret": annual_return(adjusted_returns, tdpy),
        "ann_vol": annual_volatility(adjusted_returns, tdpy),

        "cum_ret_excess": cumulative_return_simple(excess_daily.dropna()) if excess_daily.notna().any() else 0.0,
        "ann_ret_excess": annual_return(excess_daily.dropna(), tdpy) if excess_daily.notna().any() else 0.0,
        "ann_vol_excess": annual_volatility(excess_daily.dropna(), tdpy) if excess_daily.notna().any() else 0.0,

        "ir1_excess": information_ratio_1(excess_daily.dropna(), tdpy) if excess_daily.notna().any() else 0.0,
        "ir2_excess": information_ratio_2(excess_daily.dropna(), tdpy) if excess_daily.notna().any() else 0.0,
        "win_excess": win_probability(excess_daily.dropna()) if excess_daily.notna().any() else 0.0,
        "mdd_excess": max_drawdown_simple(excess_daily.dropna()) if excess_daily.notna().any() else 0.0,
        "pl_excess": profit_loss_ratio(excess_daily.dropna()) if excess_daily.notna().any() else 0.0,

        "avg_n_holdings": float(daily_n_holdings.mean()) if len(daily_n_holdings) > 0 else 0.0,
        "avg_turnover": float(turnover.mean()) if len(turnover) > 0 else 0.0,
    }
    metrics_fmt = format_metrics(metrics_raw)

    # Yearly performance (simple yearly sum)
    logger.info("Computing yearly performance...")
    with tqdm(total=2, desc="Yearly performance") as pbar:
        strat_y = adjusted_returns.groupby(adjusted_returns.index.year).sum().round(4)
        pbar.update(1)
        if benchmark is not None and not benchmark.empty:
            bench_y = benchmark.groupby(benchmark.index.year)["PctChange"].sum().round(4)
            excess_y = (strat_y - bench_y).round(4)
            yearly = pd.DataFrame({"strategy": strat_y, "benchmark": bench_y, "excess": excess_y})
            pbar.update(1)
        else:
            yearly = pd.DataFrame({"strategy": strat_y})
            pbar.update(1)
    yearly.index.name = "year"

    return metrics_fmt, yearly, result, x


def save_backtest_artifacts(
    metrics: pd.Series,
    yearly: pd.DataFrame,
    result: pd.DataFrame,
    original_df: pd.DataFrame,
    cfg: BacktestConfig,
) -> None:
    """
    Save metrics/yearly CSV and standard plots.
    """
    safe_mkdir(cfg.output_dir)

    metrics_path = cfg.output_dir / cfg.metrics_filename
    yearly_path = cfg.output_dir / cfg.yearly_filename

    save_series_csv(metrics, metrics_path)
    save_df_csv(yearly, yearly_path)

    # plots
    plot_net_value(
        result,
        cfg.output_dir / f"net_value_{cfg.version_tag}.png",
        title=f"Net Value ({cfg.version_tag})",
    )
    plot_daily_holdings_count(
        original_df,
        cfg.output_dir / f"holdings_count_{cfg.version_tag}.png",
        title=f"# Holdings ({cfg.version_tag})",
    )
    plot_pnl_and_value(
        result,
        cfg.output_dir / f"pnl_and_value_{cfg.version_tag}.png",
        title=f"Daily PnL & Net Value ({cfg.version_tag})",
    )

    # selected holdings table
    selected_tbl = _build_selected_stocks_table(original_df)
    save_df_csv_noindex(selected_tbl, cfg.output_dir / f"{cfg.selected_stocks_filename.replace('.csv','')}_{cfg.version_tag}.csv")
