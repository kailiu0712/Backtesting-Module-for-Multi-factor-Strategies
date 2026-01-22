from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_net_value(df: pd.DataFrame, out_path: Path, title: str = "Net Value") -> None:
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(df.index, df["strategy_value"], label="Strategy", linewidth=1.8)
    if "base_value" in df.columns:
        ax1.plot(df.index, df["base_value"], label="Benchmark", linewidth=1.8)

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Net Value")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    if "excess_value" in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(df.index, df["excess_value"], label="Excess (Abs)", linewidth=1.3)
        ax2.plot(df.index, df["excess_value_relative"], label="Excess (Rel)", linewidth=1.3)
        ax2.set_ylabel("Excess")
        ax2.legend(loc="upper right")

    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def plot_daily_holdings_count(original_df: pd.DataFrame, out_path: Path, title: str = "# Holdings") -> None:
    daily_count = original_df[original_df["weights"] > 0].groupby("TradingDay").size()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_count.index, daily_count.values, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def plot_pnl_and_value(df: pd.DataFrame, out_path: Path, title: str = "Daily PnL and Net Value") -> None:
    fig, ax1 = plt.subplots(figsize=(16, 8))

    ax1.bar(df.index, df["strategy_return"], alpha=0.5, label="Daily PnL")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("PnL")
    ax1.grid(True, alpha=0.3)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(df.index, df["strategy_value"], label="Net Value", linewidth=1.8)
    ax2.set_ylabel("Net Value")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved: %s", out_path)
