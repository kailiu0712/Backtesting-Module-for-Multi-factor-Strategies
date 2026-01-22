# Quant Backtest Toolkit (Daily Frequency)

A lightweight, modular Python project for daily-frequency equity strategy backtesting. It supports: (1) cross-sectional stock selection strategies, (2) constrained portfolio weight assignment with tradability rules, (3) transaction-cost-aware return calculation with turnover, (4) benchmark-relative evaluation, and (5) standardized artifact export (metrics tables + yearly performance + plots + selected holdings table).

This repository is organized as a small Python package (`src/quant_backtest`) plus a CLI entrypoint (`scripts/run_backtest.py`) so you can run experiments reproducibly and extend the codebase cleanly.

---

## Features

- **Strategy module**: implement selection logic that outputs `select` and optionally `score`
- **Weighting module**: compute daily portfolio weights with tradability constraints (carry non-tradable holdings)
- **Backtester**:
  - daily portfolio returns via `sum(weights * next_ret)`
  - turnover calculation via per-security weight diffs
  - transaction-fee adjustment
  - benchmark-relative excess returns (optional)
  - performance metrics + yearly performance table
- **Artifacts**:
  - CSV exports: metrics, yearly performance, selected holdings
  - Plots: net value, holdings count, daily PnL + net value
- **Sanitized configuration**:
  - no hard-coded machine paths
  - no embedded credentials
  - configurable via CLI flags

---

## Repository Structure

```text
quant_backtest/
  README.md
  .env.example
  requirements.txt
  src/
    quant_backtest/
      __init__.py
      config.py
      logging_utils.py
      io.py
      metrics.py
      weighting.py
      plotting.py
      backtester.py
      strategy/
        __init__.py
        lowbeta_dividend.py
  scripts/
    run_backtest.py
