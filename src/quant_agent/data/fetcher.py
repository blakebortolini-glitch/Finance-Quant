"""Data fetchers: yfinance (equities, fundamentals, options) + FRED (macro).

All network calls go through ``_with_retry`` for resilience. Results are routed
through the parquet cache so we never refetch within a trading day. If FRED is
unavailable (no key, or request fails), macro falls back to
``FALLBACK_RISK_FREE_RATE`` from config.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date
from typing import TypeVar

import pandas as pd
import yfinance as yf
from loguru import logger

from quant_agent.config import (
    FALLBACK_RISK_FREE_RATE,
    FETCH_MAX_RETRIES,
    FETCH_RETRY_BACKOFF_SEC,
    FRED_API_KEY,
    FRED_CPI_SERIES,
    FRED_RISK_FREE_SERIES,
    FRED_SOFR_SERIES,
    MARKET_PROXY,
    OPTIONS_N_EXPIRATIONS,
    PRICE_HISTORY_YEARS,
)
from quant_agent.data import cache
from quant_agent.data.schemas import (
    Fundamentals,
    MacroSnapshot,
    OptionsChain,
    TickerData,
)

T = TypeVar("T")


def _with_retry(fn: Callable[[], T], what: str) -> T:
    """Run ``fn`` with exponential backoff; re-raise after the last attempt."""
    last_exc: Exception | None = None
    for attempt in range(1, FETCH_MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - retry any transient failure
            last_exc = exc
            wait = FETCH_RETRY_BACKOFF_SEC * attempt
            logger.warning(
                f"{what}: attempt {attempt}/{FETCH_MAX_RETRIES} failed ({exc}); "
                f"retrying in {wait:.1f}s"
            )
            if attempt < FETCH_MAX_RETRIES:
                time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def _safe(info: dict, key: str) -> float | None:
    """Pull a numeric field from yfinance .info, tolerating missing/None."""
    val = info.get(key)
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #
def _fetch_price_history(ticker: str) -> pd.DataFrame:
    """Raw 3y daily OHLCV from yfinance, normalized columns + DatetimeIndex."""
    period = f"{PRICE_HISTORY_YEARS}y"
    df = _with_retry(
        lambda: yf.Ticker(ticker).history(period=period, auto_adjust=True),
        f"prices[{ticker}]",
    )
    if df.empty:
        raise ValueError(f"No price history returned for {ticker}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df


def fetch_prices(ticker: str, as_of: date) -> pd.DataFrame:
    """Fetch 3y daily OHLCV, cached. Returns DataFrame with DatetimeIndex."""
    key = f"{ticker}_prices"
    cached = cache.get(key, as_of)
    if cached is not None:
        return cached
    df = _fetch_price_history(ticker)
    cache.put(key, as_of, df)
    return df


# --------------------------------------------------------------------------- #
# Fundamentals
# --------------------------------------------------------------------------- #
def fetch_fundamentals(ticker: str, as_of: date) -> Fundamentals:
    """Fetch the fundamental snapshot for a ticker (cached as a 1-row frame)."""
    key = f"{ticker}_fundamentals"
    cached = cache.get(key, as_of)
    if cached is not None:
        return Fundamentals(**cached.iloc[0].where(pd.notna(cached.iloc[0]), None).to_dict())

    tk = yf.Ticker(ticker)
    info = _with_retry(lambda: tk.info, f"fundamentals[{ticker}]")
    fundamentals = Fundamentals(
        ticker=ticker,
        name=info.get("shortName") or info.get("longName"),
        net_income=_safe(info, "netIncomeToCommon"),
        free_cash_flow=_safe(info, "freeCashflow"),
        total_debt=_safe(info, "totalDebt"),
        current_ratio=_safe(info, "currentRatio"),
        roa=_safe(info, "returnOnAssets"),
        gross_margin=_safe(info, "grossMargins"),
        shares_outstanding=_safe(info, "sharesOutstanding"),
        dividend_yield=_safe(info, "dividendYield"),
        beta=_safe(info, "beta"),
        analyst_target_price=_safe(info, "targetMeanPrice"),
        sector=info.get("sector"),
    )
    _populate_statement_fields(tk, fundamentals)

    frame = pd.DataFrame([fundamentals.model_dump()])
    cache.put(key, as_of, frame)
    return fundamentals


def _row(df: pd.DataFrame | None, candidates: list[str], col: int) -> float | None:
    """First matching row label's value at column ``col`` (0=current, 1=prior)."""
    if df is None or df.empty or df.shape[1] <= col:
        return None
    for label in candidates:
        if label in df.index:
            try:
                val = df.loc[label].iloc[col]
                return float(val) if pd.notna(val) else None
            except (TypeError, ValueError, IndexError):
                continue
    return None


def _populate_statement_fields(tk: yf.Ticker, f: Fundamentals) -> None:
    """Fill current/prior statement line items from yfinance, best-effort."""
    try:
        income = tk.income_stmt
        balance = tk.balance_sheet
        cash = tk.cashflow
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"statements[{f.ticker}] unavailable: {exc}")
        return

    rev = ["Total Revenue", "Operating Revenue"]
    gp = ["Gross Profit"]
    ebit = ["EBIT", "Operating Income", "Total Operating Income As Reported"]
    ni = ["Net Income", "Net Income Common Stockholders"]
    ta = ["Total Assets"]
    tl = ["Total Liabilities Net Minority Interest", "Total Liabilities"]
    ltd = ["Long Term Debt"]
    ca = ["Current Assets", "Total Current Assets"]
    cl = ["Current Liabilities", "Total Current Liabilities"]
    re = ["Retained Earnings"]
    ocf = ["Operating Cash Flow", "Total Cash From Operating Activities"]
    sh = ["Ordinary Shares Number", "Share Issued"]

    # Current period (column 0).
    f.revenue = _row(income, rev, 0)
    f.gross_profit = _row(income, gp, 0)
    f.ebit = _row(income, ebit, 0)
    if f.total_assets is None:
        f.total_assets = _row(balance, ta, 0)
    f.total_liabilities = _row(balance, tl, 0)
    f.long_term_debt = _row(balance, ltd, 0)
    f.current_assets = _row(balance, ca, 0)
    f.current_liabilities = _row(balance, cl, 0)
    f.retained_earnings = _row(balance, re, 0)
    f.operating_cash_flow = _row(cash, ocf, 0)

    # Prior period (column 1) for Piotroski year-over-year criteria.
    f.net_income_prior = _row(income, ni, 1)
    f.revenue_prior = _row(income, rev, 1)
    f.gross_profit_prior = _row(income, gp, 1)
    f.total_assets_prior = _row(balance, ta, 1)
    f.long_term_debt_prior = _row(balance, ltd, 1)
    f.current_assets_prior = _row(balance, ca, 1)
    f.current_liabilities_prior = _row(balance, cl, 1)
    f.operating_cash_flow_prior = _row(cash, ocf, 1)
    f.shares_outstanding_prior = _row(balance, sh, 1)

    f.fcf_cagr_3y = _fcf_cagr(cash)


def _fcf_cagr(cash: pd.DataFrame | None) -> float | None:
    """Trailing free-cash-flow CAGR from the cashflow statement.

    Uses the most recent year and the oldest available year (up to 3 back) of
    the "Free Cash Flow" row: CAGR = (FCF_latest / FCF_oldest)^(1/years) - 1.
    Returns None if FCF is missing or non-positive at either endpoint (a CAGR
    across a sign change is meaningless).
    """
    fcf_labels = ["Free Cash Flow"]
    latest = _row(cash, fcf_labels, 0)
    if latest is None or cash is None:
        return None

    n_cols = cash.shape[1]
    oldest_col = min(3, n_cols - 1)
    if oldest_col < 1:
        return None
    oldest = _row(cash, fcf_labels, oldest_col)
    if oldest is None or oldest <= 0 or latest <= 0:
        return None

    years = oldest_col
    return float((latest / oldest) ** (1.0 / years) - 1.0)


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #
def fetch_options(ticker: str, as_of: date) -> OptionsChain | None:
    """Fetch nearest N expirations with full strike ladder (bid/ask/IV)."""
    key = f"{ticker}_options"
    cached = cache.get(key, as_of)
    spot = float(fetch_prices(ticker, as_of)["Close"].iloc[-1])
    if cached is not None:
        return OptionsChain(ticker=ticker, spot=spot, data=cached)

    try:
        tk = yf.Ticker(ticker)
        expirations = _with_retry(lambda: list(tk.options), f"options-exp[{ticker}]")
        if not expirations:
            logger.info(f"No options chain available for {ticker}")
            return None

        frames: list[pd.DataFrame] = []
        for exp in expirations[:OPTIONS_N_EXPIRATIONS]:
            chain = _with_retry(lambda e=exp: tk.option_chain(e), f"options[{ticker} {exp}]")
            for opt_type, leg in (("call", chain.calls), ("put", chain.puts)):
                leg = leg[["strike", "bid", "ask", "impliedVolatility"]].copy()
                leg.insert(0, "expiration", exp)
                leg.insert(1, "type", opt_type)
                leg = leg.rename(columns={"impliedVolatility": "iv"})
                frames.append(leg)

        data = pd.concat(frames, ignore_index=True)
        cache.put(key, as_of, data)
        return OptionsChain(ticker=ticker, spot=spot, data=data)
    except Exception as exc:  # noqa: BLE001 - options are optional; degrade gracefully
        logger.warning(f"Options fetch failed for {ticker}: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Macro (FRED)
# --------------------------------------------------------------------------- #
def fetch_macro(as_of: date) -> MacroSnapshot:
    """Fetch macro snapshot from FRED, falling back to config on failure."""
    if not FRED_API_KEY or FRED_API_KEY == "your_fred_api_key_here":
        logger.warning("FRED_API_KEY not set; using fallback risk-free rate.")
        return MacroSnapshot(risk_free_rate=FALLBACK_RISK_FREE_RATE, is_fallback=True)

    try:
        from fredapi import Fred

        fred = Fred(api_key=FRED_API_KEY)

        rf_pct = _with_retry(
            lambda: fred.get_series(FRED_RISK_FREE_SERIES).dropna().iloc[-1],
            "fred[DGS3MO]",
        )
        sofr_pct = _latest_or_none(fred, FRED_SOFR_SERIES)
        cpi_yoy = _cpi_yoy(fred)

        return MacroSnapshot(
            risk_free_rate=float(rf_pct) / 100.0,
            cpi_yoy=cpi_yoy,
            sofr=sofr_pct / 100.0 if sofr_pct is not None else None,
            is_fallback=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"FRED fetch failed ({exc}); using fallback risk-free rate.")
        return MacroSnapshot(risk_free_rate=FALLBACK_RISK_FREE_RATE, is_fallback=True)


def _latest_or_none(fred, series_id: str) -> float | None:
    try:
        return float(fred.get_series(series_id).dropna().iloc[-1])
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"fred[{series_id}] unavailable: {exc}")
        return None


def _cpi_yoy(fred) -> float | None:
    """Year-over-year CPI inflation as a decimal."""
    try:
        cpi = fred.get_series(FRED_CPI_SERIES).dropna()
        latest = cpi.iloc[-1]
        year_ago = cpi.asof(cpi.index[-1] - pd.DateOffset(years=1))
        return float(latest / year_ago - 1.0)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"cpi_yoy unavailable: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #
def fetch_ticker(ticker: str, as_of: date | None = None) -> TickerData:
    """Aggregate all payloads for one ticker into a TickerData contract."""
    as_of = as_of or date.today()
    logger.info(f"Fetching {ticker} as of {as_of}")

    prices = fetch_prices(ticker, as_of)
    fundamentals = fetch_fundamentals(ticker, as_of)
    options = fetch_options(ticker, as_of)
    market_prices = fetch_prices(MARKET_PROXY, as_of)
    macro = fetch_macro(as_of)

    return TickerData(
        ticker=ticker,
        as_of=as_of,
        prices=prices,
        fundamentals=fundamentals,
        options=options,
        market_prices=market_prices,
        macro=macro,
    )


__all__ = [
    "fetch_prices",
    "fetch_fundamentals",
    "fetch_options",
    "fetch_macro",
    "fetch_ticker",
]
