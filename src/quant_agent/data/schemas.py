"""Pydantic models defining the data contracts for every fetched payload.

These are the typed boundary between raw provider responses (yfinance, FRED)
and the rest of the pipeline. Validation happens here so downstream models can
assume clean inputs.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class Fundamentals(BaseModel):
    """Fundamental snapshot for one ticker.

    Carries the current-period figures used by the DCF and Altman Z-Score plus
    the prior-period figures Piotroski needs for its year-over-year criteria.
    Every field is optional; downstream models degrade gracefully when a line
    item is unavailable from the data provider.
    """

    ticker: str
    name: str | None = None              # company display name (for securities)

    # --- Summary / market figures (from .info) --------------------------- #
    net_income: float | None = None
    free_cash_flow: float | None = None
    total_debt: float | None = None
    total_assets: float | None = None
    current_ratio: float | None = None
    roa: float | None = None
    gross_margin: float | None = None
    shares_outstanding: float | None = None
    dividend_yield: float | None = None
    beta: float | None = None
    analyst_target_price: float | None = None
    sector: str | None = None

    # --- Derived growth -------------------------------------------------- #
    fcf_cagr_3y: float | None = None     # trailing 3y free-cash-flow CAGR

    # --- Current-period statement line items ----------------------------- #
    revenue: float | None = None
    gross_profit: float | None = None
    ebit: float | None = None
    operating_cash_flow: float | None = None
    total_liabilities: float | None = None
    long_term_debt: float | None = None
    current_assets: float | None = None
    current_liabilities: float | None = None
    retained_earnings: float | None = None

    # --- Prior-period figures (for Piotroski year-over-year deltas) ------- #
    net_income_prior: float | None = None
    total_assets_prior: float | None = None
    revenue_prior: float | None = None
    gross_profit_prior: float | None = None
    long_term_debt_prior: float | None = None
    current_assets_prior: float | None = None
    current_liabilities_prior: float | None = None
    operating_cash_flow_prior: float | None = None
    shares_outstanding_prior: float | None = None


class MacroSnapshot(BaseModel):
    """Macro environment on the run date (annualized decimals where rates)."""

    risk_free_rate: float = Field(..., description="3M T-bill yield (DGS3MO)")
    cpi_yoy: float | None = None
    sofr: float | None = None
    is_fallback: bool = False


class OptionsChain(BaseModel):
    """Options chain across the nearest N monthly expirations.

    ``data`` holds one row per (expiration, strike, type) with bid/ask/IV.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ticker: str
    spot: float
    data: pd.DataFrame


class TickerData(BaseModel):
    """Complete fetched payload for a single ticker on the run date."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ticker: str
    as_of: date
    prices: pd.DataFrame  # 3y daily OHLCV, DatetimeIndex
    fundamentals: Fundamentals
    options: OptionsChain | None = None
    market_prices: pd.DataFrame  # SPY OHLCV for CAPM regression
    macro: MacroSnapshot
