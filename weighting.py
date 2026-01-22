from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

logger = logging.getLogger(__name__)


REQUIRED_TRADE_COLS = [
    "TradeStatus",
    "SwingStatus",
    "StopTradeStatus3",
    "StopTradeStatus5",
    "IpoStatus",
]


def compute_weights_with_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily portfolio weights given:
      - selection signal column: `select` in {0,1}
      - tradability constraints: TradeStatus, SwingStatus, StopTradeStatus3, StopTradeStatus5, IpoStatus

    Behavior:
      - If a stock is selected and tradable, it can receive allocation.
      - If a stock is not tradable today but was held yesterday, we keep yesterday's weight.
      - Remaining weight is reallocated equally across tradable+selected positions.

    Adds columns:
      - weights: final portfolio weight
      - w_prev: previous day weights aligned to each row (diagnostic)
      - locked_weight: weights that are carried due to non-tradable constraint (diagnostic)
    """
    missing = [c for c in REQUIRED_TRADE_COLS + ["TradingDay", "SecuCode", "select"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for weighting: {missing}")

    logger.info("Computing weights with trading constraints...")

    out = df.copy()
    out["TradingDay"] = pd.to_datetime(out["TradingDay"])

    dates = np.array(sorted(out["TradingDay"].unique()))
    secu_codes = out["SecuCode"].astype(int).values

    trade_mask = (
        (out["TradeStatus"] == 1)
        & (out["SwingStatus"] == 1)
        & (out["StopTradeStatus3"] == 1)
        & (out["StopTradeStatus5"] == 0)
        & (out["IpoStatus"] == 1)
    ).values

    select_arr = out["select"].values.astype(np.int8)

    weights = np.zeros(len(out), dtype=np.float64)
    w_prev_aligned = np.zeros(len(out), dtype=np.float64)
    locked_weight_aligned = np.zeros(len(out), dtype=np.float64)

    @njit
    def _daily_weight_update(day_indices, prev_weights, prev_codes, trade_mask_arr, select_arr, secu_codes_arr):
        n = len(day_indices)
        daily = np.zeros(n, dtype=np.float64)

        tradable = trade_mask_arr[day_indices]
        selected = select_arr[day_indices] == 1
        valid = tradable & selected
        valid_count = np.sum(valid)

        if valid_count == 0:
            # No tradable selected stocks => if there are non-tradable carryovers, they remain;
            # otherwise all zero.
            if len(prev_weights) == 0:
                return daily

        if len(prev_weights) == 0:
            # First day: equal weight across tradable+selected
            if valid_count > 0:
                daily[valid] = 1.0 / valid_count
            return daily

        # Carry over weights for non-tradable positions if held yesterday
        remaining = 1.0
        for i in range(n):
            if not tradable[i]:
                code = secu_codes_arr[day_indices[i]]
                for j in range(len(prev_codes)):
                    if prev_codes[j] == code and prev_weights[j] > 0:
                        daily[i] = prev_weights[j]
                        remaining -= prev_weights[j]
                        break

        # Reallocate remaining weight equally among tradable+selected
        if remaining > 0 and np.sum(valid) > 0:
            daily[valid] += remaining / np.sum(valid)

        return daily

    prev_weights = np.array([], dtype=np.float64)
    prev_codes = np.array([], dtype=np.int64)

    for d in tqdm(dates, desc="Weighting by day", unit="day"):
        idx = np.where(out["TradingDay"].values == d)[0]
        daily_w = _daily_weight_update(idx, prev_weights, prev_codes, trade_mask, select_arr, secu_codes)

        weights[idx] = daily_w
        w_prev_aligned[idx] = daily_w  # aligned diagnostic; note "prev" here means previous after update in your original
        locked_weight_aligned[idx] = np.where(trade_mask[idx], 0.0, daily_w)

        prev_weights = daily_w.copy()
        prev_codes = secu_codes[idx]

    out["weights"] = weights
    out["w_prev"] = w_prev_aligned
    out["locked_weight"] = locked_weight_aligned
    return out
