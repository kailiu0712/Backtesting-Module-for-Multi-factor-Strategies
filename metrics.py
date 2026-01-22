from __future__ import annotations

import numpy as np
import pandas as pd


def cumulative_return_simple(returns: pd.Series) -> float:
    """Simple cumulative return (sum of daily returns)."""
    r = returns.dropna()
    return float(r.sum())


def annual_return(returns: pd.Series, trading_days_per_year: int) -> float:
    return float(returns.mean() * trading_days_per_year)


def annual_volatility(returns: pd.Series, trading_days_per_year: int) -> float:
    return float(returns.std() * np.sqrt(trading_days_per_year))


def max_drawdown_simple(returns: pd.Series) -> float:
    """
    Max drawdown under simple-return cumulation:
      equity = 1 + cumsum(r)
    """
    r = returns.fillna(0.0)
    equity = 1.0 + r.cumsum()
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0.0, np.nan)
    return float(dd.max())


def information_ratio_1(excess_returns: pd.Series, trading_days_per_year: int) -> float:
    mu = excess_returns.mean() * trading_days_per_year
    sig = excess_returns.std() * np.sqrt(trading_days_per_year)
    return float(mu / sig) if sig != 0 else 0.0


def information_ratio_2(excess_returns: pd.Series, trading_days_per_year: int) -> float:
    """
    Variant IR with drawdown penalty:
      (annual_mean - max_dd/4) / annual_vol
    """
    mdd = max_drawdown_simple(excess_returns)
    mu = excess_returns.mean() * trading_days_per_year
    sig = excess_returns.std() * np.sqrt(trading_days_per_year)
    return float((mu - mdd / 4.0) / sig) if sig != 0 else 0.0


def win_probability(returns: pd.Series) -> float:
    nz = returns[returns != 0]
    if nz.empty:
        return 0.0
    return float((nz > 0).mean())


def profit_loss_ratio(returns: pd.Series) -> float:
    profits = returns[returns > 0]
    losses = returns[returns < 0]
    if profits.empty or losses.empty:
        return 0.0
    return float(profits.mean() / losses.abs().mean())


def format_metrics(metrics: dict) -> pd.Series:
    """
    Produce a compact, presentation-ready metrics Series in English.
    """
    return pd.Series({
        "Cumulative Return (Adj)": f"{metrics['cum_ret']:.6f}",
        "Annual Return (Adj)": f"{metrics['ann_ret']*100:.2f}%",
        "Annual Volatility (Adj)": f"{metrics['ann_vol']*100:.2f}%",
        "Cumulative Return (Excess)": f"{metrics['cum_ret_excess']:.6f}",
        "Annual Return (Excess)": f"{metrics['ann_ret_excess']*100:.2f}%",
        "Annual Volatility (Excess)": f"{metrics['ann_vol_excess']*100:.2f}%",
        "IR1 (Excess)": f"{metrics['ir1_excess']:.6f}",
        "IR2 (Excess)": f"{metrics['ir2_excess']:.6f}",
        "Win Rate (Excess)": f"{metrics['win_excess']*100:.2f}%",
        "Max Drawdown (Excess)": f"{metrics['mdd_excess']*100:.2f}%",
        "P/L Ratio (Excess)": f"{metrics['pl_excess']:.6f}",
        "Avg #Holdings": f"{metrics['avg_n_holdings']:.2f}",
        "Avg Turnover": f"{metrics['avg_turnover']*100:.2f}%",
    })
