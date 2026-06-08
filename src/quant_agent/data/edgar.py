"""SEC EDGAR data layer (edgartools): Form 4 insider trades + 13F holdings.

Free, keyless. SEC requires a contact identity on every request, set once via
``set_identity`` from ``config.EDGAR_IDENTITY``. Results are cached to parquet:
insider data daily (filings arrive daily), 13F data on a weekly bucket
(institutional reports are quarterly, so refetching daily is wasteful).
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from loguru import logger

from quant_agent.config import (
    EDGAR_CACHE_TTL_DAYS,
    EDGAR_IDENTITY,
    INSIDER_LOOKBACK_DAYS,
    TRACKED_FUNDS,
)
from quant_agent.data import cache

_INSIDER_MAX_FILINGS = 60          # cap parsing per ticker within the window
_identity_set = False


def _ensure_identity() -> None:
    global _identity_set
    if not _identity_set:
        from edgar import set_identity

        set_identity(EDGAR_IDENTITY)
        _identity_set = True


def _week_key(as_of: date) -> date:
    """Monday-anchored week bucket so weekly-TTL caches survive across days."""
    return as_of - timedelta(days=as_of.weekday())


# --------------------------------------------------------------------------- #
# Form 4 insider transactions
# --------------------------------------------------------------------------- #
_INSIDER_COLS = ["Date", "Code", "Transaction Type", "Shares", "Price", "Value",
                 "Insider", "Position", "Ticker", "FilingDate", "Is10b5_1"]


def fetch_insider_transactions(ticker: str, as_of: date) -> pd.DataFrame:
    """Form 4 transactions for ``ticker`` over the trailing lookback window.

    Returns a DataFrame (possibly empty) with the canonical insider columns.
    """
    key = f"{ticker}_insider"
    cached = cache.get(key, as_of)
    if cached is not None:
        return cached

    _ensure_identity()
    cutoff = as_of - timedelta(days=INSIDER_LOOKBACK_DAYS)
    frames: list[pd.DataFrame] = []
    try:
        from edgar import Company

        filings = Company(ticker).get_filings(form="4")
        for filing in filings:
            fdate = pd.Timestamp(filing.filing_date).date()
            if fdate < cutoff:
                break  # filings are newest-first
            try:
                obj = filing.obj()
                df = obj.to_dataframe()
                if df is not None and not df.empty:
                    df = df.copy()
                    df["FilingDate"] = pd.Timestamp(fdate)
                    # 10b5-1 plans are disclosed in Form 4 footnote text.
                    df["Is10b5_1"] = "10b5" in str(obj.footnotes).lower()
                    frames.append(df)
            except Exception as exc:  # noqa: BLE001 - skip an unparseable filing
                logger.debug(f"insider[{ticker}] filing parse skipped: {exc}")
            if len(frames) >= _INSIDER_MAX_FILINGS:
                break
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning(f"insider fetch failed for {ticker}: {exc}")

    if not frames:
        result = pd.DataFrame(columns=_INSIDER_COLS)
    else:
        combined = pd.concat(frames, ignore_index=True)
        keep = [c for c in _INSIDER_COLS if c in combined.columns]
        result = combined[keep].copy()

    cache.put(key, as_of, result)
    return result


# --------------------------------------------------------------------------- #
# 13F institutional holdings
# --------------------------------------------------------------------------- #
def _aggregate_holdings(infotable: pd.DataFrame, period: str) -> pd.DataFrame:
    """Collapse a 13F infotable to one row per ticker (long equity only)."""
    df = infotable.copy()
    if "Ticker" not in df.columns:
        return pd.DataFrame(columns=["Ticker", "value", "shares", "period"])
    df = df[df["Ticker"].notna() & (df["Ticker"].astype(str).str.len() > 0)]
    if "PutCall" in df.columns:  # exclude options, keep direct long stock
        df = df[df["PutCall"].fillna("").astype(str).str.strip() == ""]
    df["value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0.0)
    shares_col = "SharesPrnAmount" if "SharesPrnAmount" in df.columns else "Shares"
    df["shares"] = pd.to_numeric(df.get(shares_col), errors="coerce").fillna(0.0)
    agg = df.groupby("Ticker", as_index=False).agg(value=("value", "sum"),
                                                    shares=("shares", "sum"))
    agg["period"] = period
    return agg


def fetch_fund_holdings(name: str, cik: int, as_of: date) -> dict | None:
    """Latest + prior 13F holdings for one fund, aggregated by ticker.

    Returns dict(name, cik, latest: df, prior: df|None, period_latest,
    period_prior) or None if unavailable.
    """
    wk = _week_key(as_of)
    key_latest = f"{cik}_13f_latest"
    key_prior = f"{cik}_13f_prior"
    cached_latest = cache.get(key_latest, wk, ttl_days=EDGAR_CACHE_TTL_DAYS)
    cached_prior = cache.get(key_prior, wk, ttl_days=EDGAR_CACHE_TTL_DAYS)
    if cached_latest is not None:
        period_latest = cached_latest["period"].iloc[0] if not cached_latest.empty else "?"
        period_prior = (
            cached_prior["period"].iloc[0]
            if cached_prior is not None and not cached_prior.empty else None
        )
        return {
            "name": name, "cik": cik,
            "latest": cached_latest, "prior": cached_prior,
            "period_latest": period_latest, "period_prior": period_prior,
        }

    _ensure_identity()
    try:
        from edgar import Company

        filings = Company(cik).get_filings(form="13F-HR")
        if len(filings) == 0:
            return None
        latest_obj = filings[0].obj()
        period_latest = str(getattr(latest_obj, "report_period", "?"))
        latest = _aggregate_holdings(latest_obj.infotable, period_latest)

        prior = None
        period_prior = None
        if len(filings) > 1:
            prior_obj = filings[1].obj()
            period_prior = str(getattr(prior_obj, "report_period", "?"))
            prior = _aggregate_holdings(prior_obj.infotable, period_prior)

        cache.put(key_latest, wk, latest)
        if prior is not None:
            cache.put(key_prior, wk, prior)
        return {
            "name": name, "cik": cik, "latest": latest, "prior": prior,
            "period_latest": period_latest, "period_prior": period_prior,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"13F fetch failed for {name} (CIK {cik}): {exc}")
        return None


def fetch_all_fund_holdings(as_of: date) -> list[dict]:
    """Fetch holdings for every tracked fund (skipping failures)."""
    out = []
    for name, cik in TRACKED_FUNDS.items():
        logger.info(f"Fetching 13F: {name} (CIK {cik})")
        data = fetch_fund_holdings(name, cik, as_of)
        if data is not None:
            out.append(data)
    return out


__all__ = [
    "fetch_insider_transactions",
    "fetch_fund_holdings",
    "fetch_all_fund_holdings",
]
