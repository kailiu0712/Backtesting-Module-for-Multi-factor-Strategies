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

    factor1 = g[""].transform(lambda x: x.quantile(0.5))
    factor2 = g[""].transform(lambda x: x.quantile(0.5))
    factor3 = g[""].transform(lambda x: x.quantile(0.5))
    factor4 = g[""].transform(lambda x: x.quantile(0.5))
    factor5 = g[""].transform(lambda x: x.quantile(0.5))
    factor6 = g[""].transform(lambda x: x.quantile(0.5))
    factor7 = g[""].transform(lambda x: x.quantile(0.5))

    scores = (
        1.0 * (sub[""] >= factor1)
        + 1.0 * (sub[""] >= factor2)
        + 0.5 * (sub[""] > 3)
        + 0.5 * (sub[""] > 0)
        + 1.0 * (sub[""] >= factor3)
        + 1.0 * (sub[""] >= factor4)
        + 0.5 * (sub[""] >= factor5)
        + 0.5 * (sub[""] > 0)
        + 1.0 * (sub[""] >= factor6)
        + 0.5 * (sub[""] >= factor7)
    )

    sub["score"] = scores
    sub["select"] = (scores >= 6.0).astype(int)

    out.loc[mask, "select"] = sub["select"].values
    out.loc[mask, "score"] = sub["score"].values
    return out
