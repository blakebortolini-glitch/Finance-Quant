"""Tests for DCF, Piotroski F-Score, and Altman Z-Score."""

from __future__ import annotations

import pytest

from quant_agent.data.schemas import Fundamentals
from quant_agent.models import valuation


@pytest.fixture
def strong() -> Fundamentals:
    """A fundamentally improving, healthy firm (should score high)."""
    return Fundamentals(
        ticker="STRONG",
        net_income=1000,
        free_cash_flow=900,
        total_debt=500,
        total_assets=8000,
        shares_outstanding=100,
        revenue=5000,
        gross_profit=2000,
        ebit=1200,
        operating_cash_flow=1100,
        total_liabilities=3000,
        long_term_debt=400,
        current_assets=2500,
        current_liabilities=1500,
        retained_earnings=2200,
        net_income_prior=800,
        total_assets_prior=7800,
        revenue_prior=4500,
        gross_profit_prior=1700,
        long_term_debt_prior=500,
        current_assets_prior=2100,
        current_liabilities_prior=1600,
        operating_cash_flow_prior=950,
        shares_outstanding_prior=100,
    )


def test_wacc_formula():
    # rf + beta*MRP + size_premium (size premium default 0).
    assert valuation.wacc(1.0, 0.04, 0.055) == pytest.approx(0.095)


def test_dcf_positive_for_cash_generative_firm(strong):
    iv = valuation.dcf_intrinsic_value(strong, beta=1.1, rf=0.04, market_premium=0.055)
    assert iv is not None and iv > 0


def test_fcf_growth_uses_clamped_cagr():
    from quant_agent.config import (
        DCF_DEFAULT_FCF_GROWTH,
        DCF_FCF_GROWTH_CAP,
        DCF_FCF_GROWTH_FLOOR,
    )

    # In-band CAGR is used as-is.
    f = Fundamentals(ticker="X", fcf_cagr_3y=0.12)
    assert valuation.fcf_growth_rate(f) == pytest.approx(0.12)
    # Above cap -> capped.
    assert valuation.fcf_growth_rate(Fundamentals(ticker="X", fcf_cagr_3y=0.55)) == pytest.approx(
        DCF_FCF_GROWTH_CAP
    )
    # Below floor (incl. negative) -> floored.
    assert valuation.fcf_growth_rate(Fundamentals(ticker="X", fcf_cagr_3y=-0.30)) == pytest.approx(
        DCF_FCF_GROWTH_FLOOR
    )
    # Missing CAGR -> default (which is within bounds).
    assert valuation.fcf_growth_rate(Fundamentals(ticker="X")) == pytest.approx(
        DCF_DEFAULT_FCF_GROWTH
    )


def test_dcf_growth_higher_yields_higher_value(strong):
    low = valuation.dcf_intrinsic_value(strong, 1.1, 0.04, 0.055, fcf_growth=0.02)
    high = valuation.dcf_intrinsic_value(strong, 1.1, 0.04, 0.055, fcf_growth=0.20)
    assert high > low


def test_dcf_none_without_fcf():
    f = Fundamentals(ticker="X", free_cash_flow=None, shares_outstanding=100)
    assert valuation.dcf_intrinsic_value(f, 1.0, 0.04, 0.055) is None


def test_dcf_none_when_wacc_below_terminal_growth(strong):
    # Negative beta + low rf can push WACC under terminal growth -> None.
    assert valuation.dcf_intrinsic_value(strong, beta=-2.0, rf=0.0, market_premium=0.01) is None


def test_piotroski_strong_firm_scores_high(strong):
    score = valuation.piotroski_f_score(strong)
    assert 7 <= score <= 9


def test_piotroski_missing_data_is_conservative():
    # No prior-year data -> YoY criteria cannot pass; only level criteria can.
    f = Fundamentals(
        ticker="X",
        net_income=100,
        total_assets=1000,
        operating_cash_flow=120,
    )
    score = valuation.piotroski_f_score(f)
    # ROA>0, CFO>0, CFO>NI -> 3; everything requiring priors -> 0.
    assert score == 3


def test_altman_safe_firm_above_threshold(strong):
    z = valuation.altman_z_score(strong, market_cap=12000)
    assert z is not None and z > 2.99


def test_altman_none_without_total_assets():
    f = Fundamentals(ticker="X", total_assets=None)
    assert valuation.altman_z_score(f, market_cap=1000) is None
