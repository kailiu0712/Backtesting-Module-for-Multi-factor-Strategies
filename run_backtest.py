from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from logging_utils import setup_logging
from config import BacktestConfig
from io import load_parquet
from strategy import build_lowbeta_dividend_strategy
from weighting import compute_weights_with_constraints
from backtester import run_backtest, save_backtest_artifacts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_parquet", type=str, required=True, help="Input parquet containing merged features + next_ret + trading flags.")
    p.add_argument("--benchmark_csv", type=str, default=None, help="Optional benchmark CSV.")
    p.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--out", type=str, default="outputs", help="Output directory")
    p.add_argument("--version", type=str, default="v1", help="Version tag for outputs")
    p.add_argument("--fee", type=float, default=0.001, help="Transaction fee rate per turnover")
    p.add_argument("--bench_col", type=str, default="next_ret", help="Benchmark return column name")
    return p.parse_args()


def main():
    setup_logging()

    args = parse_args()
    df = load_parquet(Path(args.data_parquet))

    # 1) Strategy: select + score
    df = build_lowbeta_dividend_strategy(df)

    # 2) Weighting under tradability constraints
    df = compute_weights_with_constraints(df)

    cfg = BacktestConfig(
        start_date=datetime.strptime(args.start, "%Y-%m-%d"),
        end_date=datetime.strptime(args.end, "%Y-%m-%d"),
        benchmark_csv=Path(args.benchmark_csv) if args.benchmark_csv else None,
        benchmark_return_column=args.bench_col,
        output_dir=Path(args.out),
        version_tag=args.version,
        transaction_fee_rate=args.fee,
    )

    metrics, yearly, result, original_df = run_backtest(df, cfg)
    save_backtest_artifacts(metrics, yearly, result, original_df, cfg)

    print("\nMetrics:\n", metrics.to_string())
    print("\nYearly Performance:\n", yearly.to_string())


if __name__ == "__main__":
    main()
