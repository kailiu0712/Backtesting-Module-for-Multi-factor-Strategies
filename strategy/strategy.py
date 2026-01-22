from __future__ import annotations

import pandas as pd


def build_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Low-beta stock pool selection strategy.

    Expected inputs (columns used):

    Outputs (adds/overwrites):

    Selection rule (per trading day, X universe):

    """
    out = df.copy()
    out["TradingDay"] = pd.to_datetime(out["TradingDay"])

    # Initialize output columns
    out["select"] = 0
    out["score"] = 0.0

    mask = out[""] > 0
    sub = out.loc[mask].copy()

    # Ensure numeric
    sub[""] = sub[""].astype(float)

    g = sub.groupby("TradingDay")

    ep_q = g[""].transform(lambda x: x.quantile(0.7))
    bp_q = g[""].transform(lambda x: x.quantile(0.5))
    div_q = g[""].transform(lambda x: x.quantile(0.7))
    eg_q = g[""].transform(lambda x: x.quantile(0.7))
    opg_q = g[""].transform(lambda x: x.quantile(0.7))
    ocfp_q = g[""].transform(lambda x: x.quantile(0.6))
    sfhp_q = g[""].transform(lambda x: x.quantile(0.8))

    scores = (
        1.0 * (sub[""] >= ep_q)
        + 1.0 * (sub[""] >= bp_q)
        + 0.5 * (sub[""] > 3)
        + 0.5 * (sub[""] > 0)
        + 1.0 * (sub[""] >= div_q)
        + 1.0 * (sub[""] >= eg_q)
        + 0.5 * (sub[""] >= opg_q)
        + 0.5 * (sub[""] > 0)
        + 1.0 * (sub[""] >= ocfp_q)
        + 0.5 * (sub[""] >= sfhp_q)
    )

    sub["score"] = scores
    sub["select"] = (scores >= 6.0).astype(int)

    out.loc[mask, "select"] = sub["select"].values
    out.loc[mask, "score"] = sub["score"].values
    return out
