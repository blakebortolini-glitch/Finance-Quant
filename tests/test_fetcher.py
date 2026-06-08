"""Unit tests for fetcher helpers that don't require network access."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_agent.config import FINANCIAL_SECTORS
from quant_agent.data import fetcher


def _cashflow(fcf_by_year: list[float]) -> pd.DataFrame:
    """Build a yfinance-style cashflow frame (newest year first)."""
    cols = pd.to_datetime([f"{2025 - i}-12-31" for i in range(len(fcf_by_year))])
    return pd.DataFrame({c: [v] for c, v in zip(cols, fcf_by_year, strict=True)},
                        index=["Free Cash Flow"])


def test_fcf_cagr_three_year():
    # 100 -> 172.8 over 3 years = 20% CAGR.
    cash = _cashflow([172.8, 140.0, 120.0, 100.0])
    assert fetcher._fcf_cagr(cash) == pytest.approx(0.20, abs=1e-6)


def test_fcf_cagr_handles_two_years_only():
    # Only 2 columns -> 1-year CAGR over the available span.
    cash = _cashflow([110.0, 100.0])
    assert fetcher._fcf_cagr(cash) == pytest.approx(0.10, abs=1e-6)


def test_fcf_cagr_none_on_sign_change():
    cash = _cashflow([120.0, 50.0, -10.0, -50.0])  # oldest negative
    assert fetcher._fcf_cagr(cash) is None


def test_fcf_cagr_none_when_missing():
    assert fetcher._fcf_cagr(None) is None
    empty = pd.DataFrame(index=["Other"], data={pd.Timestamp("2025-12-31"): [1.0]})
    assert fetcher._fcf_cagr(empty) is None


def test_financial_sectors_membership():
    assert "Financial Services" in FINANCIAL_SECTORS  # banks, insurers
    assert "Real Estate" in FINANCIAL_SECTORS         # REITs
    assert "Technology" not in FINANCIAL_SECTORS


def test_row_helper_picks_first_matching_label():
    df = pd.DataFrame(
        {pd.Timestamp("2025-12-31"): [10.0, np.nan], pd.Timestamp("2024-12-31"): [8.0, 5.0]},
        index=["EBIT", "Other"],
    )
    assert fetcher._row(df, ["Missing", "EBIT"], 0) == pytest.approx(10.0)
    assert fetcher._row(df, ["EBIT"], 1) == pytest.approx(8.0)
    assert fetcher._row(df, ["Nope"], 0) is None
