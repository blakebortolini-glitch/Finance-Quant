"""Fundamental valuation: DCF, Piotroski F-Score, Altman Z-Score.

DCF:        5-year FCF forecast + Gordon terminal value, discounted at
            WACC = rf + beta*MRP + size_premium, terminal growth 3%.
Piotroski:  0-9 from 9 binary fundamental criteria (>=7 strong). Criteria that
            require a missing line item score 0 (conservative).
Altman Z:   Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5 (>2.99 safe).
"""

from __future__ import annotations

from quant_agent.config import (
    DCF_DEFAULT_FCF_GROWTH,
    DCF_FCF_GROWTH_CAP,
    DCF_FCF_GROWTH_FLOOR,
    DCF_FORECAST_YEARS,
    DCF_SIZE_PREMIUM,
    DCF_TERMINAL_GROWTH,
)
from quant_agent.data.schemas import Fundamentals


def fcf_growth_rate(fundamentals: Fundamentals) -> float:
    """Per-ticker FCF growth from the trailing 3y CAGR, clamped to config bounds.

    Falls back to ``DCF_DEFAULT_FCF_GROWTH`` when the CAGR is unavailable, then
    clamps the result to [DCF_FCF_GROWTH_FLOOR, DCF_FCF_GROWTH_CAP].
    """
    cagr = fundamentals.fcf_cagr_3y
    g = cagr if cagr is not None else DCF_DEFAULT_FCF_GROWTH
    return float(min(DCF_FCF_GROWTH_CAP, max(DCF_FCF_GROWTH_FLOOR, g)))


def wacc(beta: float, rf: float, market_premium: float) -> float:
    """Discount rate: rf + beta*MRP + size premium (a CAPM cost of equity)."""
    return float(rf + beta * market_premium + DCF_SIZE_PREMIUM)


def dcf_intrinsic_value(
    fundamentals: Fundamentals,
    beta: float,
    rf: float,
    market_premium: float,
    fcf_growth: float | None = None,
) -> float | None:
    """Discounted cash flow intrinsic value per share.

    Projects free cash flow ``DCF_FORECAST_YEARS`` forward at ``fcf_growth``
    (defaults to the per-ticker clamped 3y FCF CAGR via ``fcf_growth_rate``),
    adds a Gordon-growth terminal value at ``DCF_TERMINAL_GROWTH``, discounts at
    WACC, subtracts net debt, and divides by shares outstanding.

    Returns None if FCF or shares are unavailable, or if WACC <= terminal g.
    """
    fcf0 = fundamentals.free_cash_flow
    shares = fundamentals.shares_outstanding
    if not fcf0 or not shares or fcf0 <= 0 or shares <= 0:
        return None

    if fcf_growth is None:
        fcf_growth = fcf_growth_rate(fundamentals)

    r = wacc(beta, rf, market_premium)
    g = DCF_TERMINAL_GROWTH
    if r <= g:
        return None  # Gordon terminal value diverges

    pv = 0.0
    fcf = fcf0
    for year in range(1, DCF_FORECAST_YEARS + 1):
        fcf *= 1.0 + fcf_growth
        pv += fcf / (1.0 + r) ** year

    terminal_fcf = fcf * (1.0 + g)
    terminal_value = terminal_fcf / (r - g)
    pv += terminal_value / (1.0 + r) ** DCF_FORECAST_YEARS

    net_debt = (fundamentals.total_debt or 0.0)
    equity_value = pv - net_debt
    return float(equity_value / shares)


def piotroski_f_score(f: Fundamentals) -> int:
    """Piotroski F-Score (0-9). Missing-data criteria score 0 (conservative).

    Profitability (4): ROA>0, CFO>0, ROA up YoY, CFO>NetIncome (accruals).
    Leverage/Liquidity (3): LT-debt ratio down YoY, current ratio up YoY,
        no share dilution.
    Efficiency (2): gross margin up YoY, asset turnover up YoY.
    """
    score = 0

    # --- Profitability ---------------------------------------------------- #
    roa_now = _roa(f.net_income, f.total_assets)
    roa_prior = _roa(f.net_income_prior, f.total_assets_prior)
    if roa_now is not None and roa_now > 0:
        score += 1
    if f.operating_cash_flow is not None and f.operating_cash_flow > 0:
        score += 1
    if roa_now is not None and roa_prior is not None and roa_now > roa_prior:
        score += 1
    if (
        f.operating_cash_flow is not None
        and f.net_income is not None
        and f.operating_cash_flow > f.net_income
    ):
        score += 1

    # --- Leverage / liquidity -------------------------------------------- #
    ltd_now = _ratio(f.long_term_debt, f.total_assets)
    ltd_prior = _ratio(f.long_term_debt_prior, f.total_assets_prior)
    if ltd_now is not None and ltd_prior is not None and ltd_now < ltd_prior:
        score += 1
    cr_now = _ratio(f.current_assets, f.current_liabilities)
    cr_prior = _ratio(f.current_assets_prior, f.current_liabilities_prior)
    if cr_now is not None and cr_prior is not None and cr_now > cr_prior:
        score += 1
    if (
        f.shares_outstanding is not None
        and f.shares_outstanding_prior is not None
        and f.shares_outstanding <= f.shares_outstanding_prior
    ):
        score += 1

    # --- Operating efficiency -------------------------------------------- #
    gm_now = _ratio(f.gross_profit, f.revenue)
    gm_prior = _ratio(f.gross_profit_prior, f.revenue_prior)
    if gm_now is not None and gm_prior is not None and gm_now > gm_prior:
        score += 1
    at_now = _ratio(f.revenue, f.total_assets)
    at_prior = _ratio(f.revenue_prior, f.total_assets_prior)
    if at_now is not None and at_prior is not None and at_now > at_prior:
        score += 1

    return score


def altman_z_score(f: Fundamentals, market_cap: float) -> float | None:
    """Altman Z-Score for public firms. Returns None if total assets missing."""
    ta = f.total_assets
    if not ta or ta <= 0:
        return None

    working_capital = None
    if f.current_assets is not None and f.current_liabilities is not None:
        working_capital = f.current_assets - f.current_liabilities

    x1 = (working_capital / ta) if working_capital is not None else 0.0
    x2 = (f.retained_earnings / ta) if f.retained_earnings is not None else 0.0
    x3 = (f.ebit / ta) if f.ebit is not None else 0.0
    total_liab = f.total_liabilities or f.total_debt
    x4 = (market_cap / total_liab) if total_liab else 0.0
    x5 = (f.revenue / ta) if f.revenue is not None else 0.0

    return float(1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5)


def _roa(net_income: float | None, total_assets: float | None) -> float | None:
    return _ratio(net_income, total_assets)


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


__all__ = [
    "wacc",
    "fcf_growth_rate",
    "dcf_intrinsic_value",
    "piotroski_f_score",
    "altman_z_score",
]


if __name__ == "__main__":  # pragma: no cover
    demo = Fundamentals(
        ticker="DEMO",
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
        total_assets_prior=7500,
        revenue_prior=4500,
        gross_profit_prior=1700,
        long_term_debt_prior=450,
        current_assets_prior=2200,
        current_liabilities_prior=1500,
        operating_cash_flow_prior=950,
        shares_outstanding_prior=100,
    )
    iv = dcf_intrinsic_value(demo, beta=1.1, rf=0.04, market_premium=0.055)
    print(f"DCF intrinsic/share = {iv:.2f}")
    print(f"Piotroski F-Score   = {piotroski_f_score(demo)}/9")
    print(f"Altman Z-Score      = {altman_z_score(demo, market_cap=12000):.2f}")
