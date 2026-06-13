"""Data fetchers: FMP (primary) + yfinance (fallback for 402 tickers) + FRED (macro).

Provider strategy
-----------------
FMP stable API is tried first for prices, fundamentals, and options. If FMP
returns HTTP 402 (Payment Required — ticker not in plan coverage), the same
call is automatically retried via yfinance so small-cap / non-Starter tickers
work transparently. The cached result uses the same key regardless of which
provider won, so the fallback doesn't re-trigger on subsequent same-day calls.

FMP stable API endpoints used (https://financialmodelingprep.com/stable/):
  /profile                    — beta, sector, company name, dividend
  /historical-price-eod/full  — daily OHLCV (split-adjusted close)
  /income-statement           — revenue, EBIT, net income (annual, 2 periods)
  /balance-sheet-statement    — assets, liabilities, debt (annual, 2 periods)
  /cash-flow-statement        — operating CF, FCF (annual, 4 periods for CAGR)
  /price-target-consensus     — analyst consensus price target
  /options/chain              — options chain (graceful fallback if unavailable)

yfinance fallback used when FMP returns 402:
  yf.Ticker(t).history(...)   — prices
  yf.Ticker(t).info           — profile / fundamentals summary
  tk.income_stmt / balance_sheet / cashflow — financial statements
  tk.options / tk.option_chain(exp) — options chain
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date, timedelta
from typing import TypeVar

import pandas as pd
import requests
import yfinance as yf
from loguru import logger

from quant_agent.config import (
    FALLBACK_RISK_FREE_RATE,
    FETCH_MAX_RETRIES,
    FETCH_RETRY_BACKOFF_SEC,
    FMP_API_KEY,
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

_FMP_BASE = "https://financialmodelingprep.com/stable"


# --------------------------------------------------------------------------- #
# Shared utilities
# --------------------------------------------------------------------------- #

class _FMP402Error(Exception):
    """FMP returned HTTP 402 — ticker not covered by the current plan.

    Raised immediately; never retried (billing limits are not transient).
    """


def _with_retry(fn: Callable[[], T], what: str) -> T:
    """Run ``fn`` with exponential backoff; re-raise after the last attempt.

    ``_FMP402Error`` is re-raised immediately without retrying — a 402 from
    FMP means the ticker is not in the subscribed plan, not a transient fault.
    """
    last_exc: Exception | None = None
    for attempt in range(1, FETCH_MAX_RETRIES + 1):
        try:
            return fn()
        except _FMP402Error:
            raise  # billing limit — not transient, fall through to yfinance
        except Exception as exc:  # noqa: BLE001
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


def _fmp(path: str, params: dict | None = None) -> list | dict:
    """GET a single FMP stable endpoint; raise ``_FMP402Error`` on 402."""
    if not FMP_API_KEY:
        raise RuntimeError(
            "FMP_API_KEY not set — add it to .env or Render environment variables"
        )
    p: dict = {"apikey": FMP_API_KEY, **(params or {})}
    resp = requests.get(f"{_FMP_BASE}{path}", params=p, timeout=15)
    if resp.status_code == 402:
        raise _FMP402Error(f"FMP 402 for {path} (ticker not in plan coverage)")
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "Error Message" in data:
        raise ValueError(f"FMP error for {path}: {data['Error Message']}")
    return data  # type: ignore[return-value]


def _safe_float(d: dict, key: str) -> float | None:
    """Pull a numeric value from a dict, tolerating missing / None / strings."""
    val = d.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _first(raw: list | dict) -> dict:
    """Return first element of a list-or-single-dict FMP response."""
    if isinstance(raw, list):
        return raw[0] if raw else {}
    return raw if isinstance(raw, dict) else {}


# =========================================================================== #
# FMP implementations
# =========================================================================== #

def _fetch_price_history_fmp(ticker: str) -> pd.DataFrame:
    """Daily split-adjusted OHLCV from FMP stable API."""
    to_date   = date.today()
    from_date = to_date - timedelta(days=int(PRICE_HISTORY_YEARS * 365 + 60))

    raw = _with_retry(
        lambda: _fmp(
            "/historical-price-eod/full",
            {"symbol": ticker, "from": from_date.isoformat(), "to": to_date.isoformat()},
        ),
        f"prices[{ticker}]",
    )
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"No price history returned for {ticker} via FMP")

    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    missing = [c for c in ("Open", "High", "Low", "Close", "Volume") if c not in df.columns]
    if missing:
        raise ValueError(f"FMP price response missing columns {missing} for {ticker}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "date"
    return df


def _build_fundamentals_fmp(ticker: str) -> Fundamentals:
    """Fetch profile + three financial statements from FMP stable API."""
    profile_raw = _with_retry(
        lambda: _fmp("/profile", {"symbol": ticker}), f"profile[{ticker}]"
    )
    income_raw = _with_retry(
        lambda: _fmp("/income-statement", {"symbol": ticker, "period": "annual", "limit": "2"}),
        f"income[{ticker}]",
    )
    balance_raw = _with_retry(
        lambda: _fmp("/balance-sheet-statement",
                     {"symbol": ticker, "period": "annual", "limit": "2"}),
        f"balance[{ticker}]",
    )
    cash_raw = _with_retry(
        lambda: _fmp("/cash-flow-statement",
                     {"symbol": ticker, "period": "annual", "limit": "4"}),
        f"cashflow[{ticker}]",
    )
    analyst_target = _fetch_analyst_target_fmp(ticker)

    profile = _first(profile_raw)
    inc = income_raw  if isinstance(income_raw,  list) else []
    bal = balance_raw if isinstance(balance_raw, list) else []
    csh = cash_raw    if isinstance(cash_raw,    list) else []

    inc0 = inc[0] if len(inc) > 0 else {}
    inc1 = inc[1] if len(inc) > 1 else {}
    bal0 = bal[0] if len(bal) > 0 else {}
    bal1 = bal[1] if len(bal) > 1 else {}
    csh0 = csh[0] if len(csh) > 0 else {}
    csh1 = csh[1] if len(csh) > 1 else {}

    net_income   = _safe_float(inc0, "netIncome")
    revenue      = _safe_float(inc0, "revenue")
    gross_profit = _safe_float(inc0, "grossProfit")
    total_assets = _safe_float(bal0, "totalAssets")
    cur_assets   = _safe_float(bal0, "totalCurrentAssets")
    cur_liabs    = _safe_float(bal0, "totalCurrentLiabilities")

    roa          = (net_income / total_assets
                    if net_income is not None and total_assets else None)
    gross_margin = (gross_profit / revenue
                    if gross_profit is not None and revenue else None)
    current_ratio = (cur_assets / cur_liabs
                     if cur_assets is not None and cur_liabs else None)

    last_div = _safe_float(profile, "lastDividend")
    price    = _safe_float(profile, "price")
    dividend_yield = (last_div / price if last_div is not None and price else None)

    return Fundamentals(
        ticker=ticker,
        name=profile.get("companyName"),
        net_income=net_income,
        free_cash_flow=_safe_float(csh0, "freeCashFlow"),
        total_debt=_safe_float(bal0, "totalDebt"),
        total_assets=total_assets,
        current_ratio=current_ratio,
        roa=roa,
        gross_margin=gross_margin,
        shares_outstanding=_safe_float(inc0, "weightedAverageShsOut"),
        dividend_yield=dividend_yield,
        beta=_safe_float(profile, "beta"),
        analyst_target_price=analyst_target,
        sector=profile.get("sector"),
        revenue=revenue,
        gross_profit=gross_profit,
        ebit=_safe_float(inc0, "ebit"),
        operating_cash_flow=_safe_float(csh0, "operatingCashFlow"),
        total_liabilities=_safe_float(bal0, "totalLiabilities"),
        long_term_debt=_safe_float(bal0, "longTermDebt"),
        current_assets=cur_assets,
        current_liabilities=cur_liabs,
        retained_earnings=_safe_float(bal0, "retainedEarnings"),
        net_income_prior=_safe_float(inc1, "netIncome"),
        revenue_prior=_safe_float(inc1, "revenue"),
        gross_profit_prior=_safe_float(inc1, "grossProfit"),
        total_assets_prior=_safe_float(bal1, "totalAssets"),
        long_term_debt_prior=_safe_float(bal1, "longTermDebt"),
        current_assets_prior=_safe_float(bal1, "totalCurrentAssets"),
        current_liabilities_prior=_safe_float(bal1, "totalCurrentLiabilities"),
        operating_cash_flow_prior=_safe_float(csh1, "operatingCashFlow"),
        shares_outstanding_prior=_safe_float(inc1, "weightedAverageShsOut"),
        fcf_cagr_3y=_compute_fcf_cagr_fmp(csh),
    )


def _fetch_analyst_target_fmp(ticker: str) -> float | None:
    """FMP price-target consensus. Returns None on any failure (including 402)."""
    try:
        raw = _with_retry(
            lambda: _fmp("/price-target-consensus", {"symbol": ticker}),
            f"analyst-target[{ticker}]",
        )
        d = _first(raw) if isinstance(raw, list) else (raw if isinstance(raw, dict) else {})
        return _safe_float(d, "targetConsensus")
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"analyst-target[{ticker}] via FMP unavailable: {exc}")
        return None


def _compute_fcf_cagr_fmp(cash_statements: list[dict]) -> float | None:
    """Trailing FCF CAGR from up to 4 FMP annual cash flow dicts (newest-first)."""
    fcf_values = [
        _safe_float(s, "freeCashFlow")
        for s in cash_statements
        if _safe_float(s, "freeCashFlow") is not None
    ]
    if len(fcf_values) < 2:
        return None
    latest, oldest, years = fcf_values[0], fcf_values[-1], len(fcf_values) - 1
    if oldest <= 0 or latest <= 0:
        return None
    return float((latest / oldest) ** (1.0 / years) - 1.0)


def _fetch_options_fmp(ticker: str, spot: float, as_of: date, cache_key: str) -> OptionsChain | None:
    """Options chain from FMP. Returns None if unavailable (plan or empty)."""
    raw = _with_retry(
        lambda: _fmp("/options/chain", {"symbol": ticker}),
        f"options[{ticker}]",
    )
    options_list = raw if isinstance(raw, list) else []
    if not options_list:
        logger.info(f"No options data for {ticker} via FMP")
        return None

    expirations = sorted({
        r.get("expirationDate", r.get("expiration", ""))
        for r in options_list
        if r.get("expirationDate") or r.get("expiration")
    })[:OPTIONS_N_EXPIRATIONS]

    frames: list[pd.DataFrame] = []
    for exp in expirations:
        for opt_type in ("call", "put"):
            leg_rows = [
                r for r in options_list
                if (r.get("expirationDate") == exp or r.get("expiration") == exp)
                and r.get("putCall", r.get("type", "")).lower() == opt_type
            ]
            if not leg_rows:
                continue
            leg = pd.DataFrame([{
                "expiration": exp,
                "type": opt_type,
                "strike": _safe_float(r, "strike") or 0.0,
                "bid":    _safe_float(r, "bid")    or 0.0,
                "ask":    _safe_float(r, "ask")    or 0.0,
                "iv":     _safe_float(r, "impliedVolatility") or 0.0,
            } for r in leg_rows])
            frames.append(leg)

    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    cache.put(cache_key, as_of, df)
    return OptionsChain(ticker=ticker, spot=spot, data=df)


# =========================================================================== #
# yfinance fallback implementations
# =========================================================================== #

def _yf_safe(info: dict, key: str) -> float | None:
    """Pull a numeric field from yfinance .info, tolerating missing/None."""
    val = info.get(key)
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _yf_row(df: pd.DataFrame | None, candidates: list[str], col: int) -> float | None:
    """First matching row label's value at column ``col`` in a yfinance statement."""
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


def _fetch_price_history_yf(ticker: str) -> pd.DataFrame:
    """Raw PRICE_HISTORY_YEARS daily OHLCV from yfinance, normalized columns."""
    period = f"{PRICE_HISTORY_YEARS}y"
    df = _with_retry(
        lambda: yf.Ticker(ticker).history(period=period, auto_adjust=True),
        f"prices-yf[{ticker}]",
    )
    if df.empty:
        raise ValueError(f"yfinance returned no price history for {ticker}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df


def _build_fundamentals_yf(ticker: str) -> Fundamentals:
    """Fetch fundamental snapshot via yfinance (fallback for FMP 402 tickers)."""
    tk   = yf.Ticker(ticker)
    info = _with_retry(lambda: tk.info, f"fundamentals-yf[{ticker}]")

    f = Fundamentals(
        ticker=ticker,
        name=info.get("shortName") or info.get("longName"),
        net_income=_yf_safe(info, "netIncomeToCommon"),
        free_cash_flow=_yf_safe(info, "freeCashflow"),
        total_debt=_yf_safe(info, "totalDebt"),
        total_assets=_yf_safe(info, "totalAssets"),
        current_ratio=_yf_safe(info, "currentRatio"),
        roa=_yf_safe(info, "returnOnAssets"),
        gross_margin=_yf_safe(info, "grossMargins"),
        shares_outstanding=_yf_safe(info, "sharesOutstanding"),
        dividend_yield=_yf_safe(info, "dividendYield"),
        beta=_yf_safe(info, "beta"),
        analyst_target_price=_yf_safe(info, "targetMeanPrice"),
        sector=info.get("sector"),
    )
    _yf_populate_statements(tk, f)
    return f


def _yf_populate_statements(tk: yf.Ticker, f: Fundamentals) -> None:
    """Fill current/prior statement line items from yfinance, best-effort."""
    try:
        income  = tk.income_stmt
        balance = tk.balance_sheet
        cash    = tk.cashflow
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"yfinance statements[{f.ticker}] unavailable: {exc}")
        return

    rev  = ["Total Revenue", "Operating Revenue"]
    gp   = ["Gross Profit"]
    ebit = ["EBIT", "Operating Income", "Total Operating Income As Reported"]
    ni   = ["Net Income", "Net Income Common Stockholders"]
    ta   = ["Total Assets"]
    tl   = ["Total Liabilities Net Minority Interest", "Total Liabilities"]
    ltd  = ["Long Term Debt"]
    ca   = ["Current Assets", "Total Current Assets"]
    cl   = ["Current Liabilities", "Total Current Liabilities"]
    re   = ["Retained Earnings"]
    ocf  = ["Operating Cash Flow", "Total Cash From Operating Activities"]
    sh   = ["Ordinary Shares Number", "Share Issued"]

    f.revenue             = _yf_row(income,  rev,  0)
    f.gross_profit        = _yf_row(income,  gp,   0)
    f.ebit                = _yf_row(income,  ebit, 0)
    if f.total_assets is None:
        f.total_assets    = _yf_row(balance, ta,   0)
    f.total_liabilities   = _yf_row(balance, tl,   0)
    f.long_term_debt      = _yf_row(balance, ltd,  0)
    f.current_assets      = _yf_row(balance, ca,   0)
    f.current_liabilities = _yf_row(balance, cl,   0)
    f.retained_earnings   = _yf_row(balance, re,   0)
    f.operating_cash_flow = _yf_row(cash,    ocf,  0)

    f.net_income_prior          = _yf_row(income,  ni,  1)
    f.revenue_prior             = _yf_row(income,  rev, 1)
    f.gross_profit_prior        = _yf_row(income,  gp,  1)
    f.total_assets_prior        = _yf_row(balance, ta,  1)
    f.long_term_debt_prior      = _yf_row(balance, ltd, 1)
    f.current_assets_prior      = _yf_row(balance, ca,  1)
    f.current_liabilities_prior = _yf_row(balance, cl,  1)
    f.operating_cash_flow_prior = _yf_row(cash,    ocf, 1)
    f.shares_outstanding_prior  = _yf_row(balance, sh,  1)

    f.fcf_cagr_3y = _compute_fcf_cagr_yf(cash)


def _compute_fcf_cagr_yf(cash: pd.DataFrame | None) -> float | None:
    """Trailing FCF CAGR from yfinance cashflow DataFrame (columns = periods)."""
    fcf_labels = ["Free Cash Flow"]
    latest = _yf_row(cash, fcf_labels, 0)
    if latest is None or cash is None:
        return None
    n_cols     = cash.shape[1]
    oldest_col = min(3, n_cols - 1)
    if oldest_col < 1:
        return None
    oldest = _yf_row(cash, fcf_labels, oldest_col)
    if oldest is None or oldest <= 0 or latest <= 0:
        return None
    years = oldest_col
    return float((latest / oldest) ** (1.0 / years) - 1.0)


def _fetch_options_yf(ticker: str, spot: float, as_of: date, cache_key: str) -> OptionsChain | None:
    """Options chain from yfinance (fallback)."""
    try:
        tk          = yf.Ticker(ticker)
        expirations = _with_retry(lambda: list(tk.options), f"options-exp-yf[{ticker}]")
        if not expirations:
            logger.info(f"No options chain available for {ticker} via yfinance")
            return None

        frames: list[pd.DataFrame] = []
        for exp in expirations[:OPTIONS_N_EXPIRATIONS]:
            chain = _with_retry(
                lambda e=exp: tk.option_chain(e), f"options-yf[{ticker} {exp}]"
            )
            for opt_type, leg in (("call", chain.calls), ("put", chain.puts)):
                leg = leg[["strike", "bid", "ask", "impliedVolatility"]].copy()
                leg.insert(0, "expiration", exp)
                leg.insert(1, "type", opt_type)
                leg = leg.rename(columns={"impliedVolatility": "iv"})
                frames.append(leg)

        if not frames:
            return None
        data = pd.concat(frames, ignore_index=True)
        cache.put(cache_key, as_of, data)
        return OptionsChain(ticker=ticker, spot=spot, data=data)

    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Options fetch via yfinance failed for {ticker}: {exc}")
        return None


# =========================================================================== #
# Public API — FMP first, yfinance on 402
# =========================================================================== #

def fetch_prices(ticker: str, as_of: date) -> pd.DataFrame:
    """Fetch PRICE_HISTORY_YEARS daily OHLCV, cached. DatetimeIndex.

    Uses FMP; falls back to yfinance if FMP returns 402.
    """
    key = f"{ticker}_prices"
    cached = cache.get(key, as_of)
    if cached is not None:
        return cached

    try:
        df = _fetch_price_history_fmp(ticker)
    except _FMP402Error:
        logger.info(f"prices[{ticker}]: FMP 402 — falling back to yfinance")
        df = _fetch_price_history_yf(ticker)

    cache.put(key, as_of, df)
    return df


def fetch_fundamentals(ticker: str, as_of: date) -> Fundamentals:
    """Fetch the fundamental snapshot for a ticker (cached as a 1-row frame).

    Uses FMP; falls back to yfinance if FMP returns 402.
    """
    key = f"{ticker}_fundamentals"
    cached = cache.get(key, as_of)
    if cached is not None:
        return Fundamentals(**cached.iloc[0].where(pd.notna(cached.iloc[0]), None).to_dict())

    try:
        f = _build_fundamentals_fmp(ticker)
    except _FMP402Error:
        logger.info(f"fundamentals[{ticker}]: FMP 402 — falling back to yfinance")
        f = _build_fundamentals_yf(ticker)

    frame = pd.DataFrame([f.model_dump()])
    cache.put(key, as_of, frame)
    return f


def fetch_options(ticker: str, as_of: date) -> OptionsChain | None:
    """Fetch nearest N expirations with full strike ladder (bid/ask/IV).

    Uses FMP; falls back to yfinance if FMP returns 402. Both providers
    degrade gracefully to None on any other failure (Black-Scholes skipped).
    """
    key  = f"{ticker}_options"
    cached = cache.get(key, as_of)
    spot = float(fetch_prices(ticker, as_of)["Close"].iloc[-1])
    if cached is not None:
        return OptionsChain(ticker=ticker, spot=spot, data=cached)

    try:
        return _fetch_options_fmp(ticker, spot, as_of, key)
    except _FMP402Error:
        logger.info(f"options[{ticker}]: FMP 402 — falling back to yfinance")
        return _fetch_options_yf(ticker, spot, as_of, key)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Options fetch failed for {ticker} (IV signal skipped): {exc}")
        return None


# --------------------------------------------------------------------------- #
# Macro (FRED) — unchanged; FMP does not replace FRED macro data
# --------------------------------------------------------------------------- #

def fetch_macro(as_of: date) -> MacroSnapshot:
    """Fetch macro snapshot from FRED, falling back to config on failure."""
    if not FRED_API_KEY or FRED_API_KEY == "your_fred_api_key_here":
        logger.warning("FRED_API_KEY not set; using fallback risk-free rate.")
        return MacroSnapshot(risk_free_rate=FALLBACK_RISK_FREE_RATE, is_fallback=True)

    try:
        from fredapi import Fred

        fred = Fred(api_key=FRED_API_KEY)
        rf_pct   = _with_retry(
            lambda: fred.get_series(FRED_RISK_FREE_SERIES).dropna().iloc[-1],
            "fred[DGS3MO]",
        )
        sofr_pct = _latest_or_none(fred, FRED_SOFR_SERIES)
        cpi_yoy  = _cpi_yoy(fred)

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
        cpi      = fred.get_series(FRED_CPI_SERIES).dropna()
        latest   = cpi.iloc[-1]
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

    prices        = fetch_prices(ticker, as_of)
    fundamentals  = fetch_fundamentals(ticker, as_of)
    options       = fetch_options(ticker, as_of)
    market_prices = fetch_prices(MARKET_PROXY, as_of)
    macro         = fetch_macro(as_of)

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
