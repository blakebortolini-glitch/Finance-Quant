"""Tests for risk metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_agent.models import risk


@pytest.fixture
def rets() -> np.ndarray:
    # Clear positive drift relative to noise so Sharpe is unambiguously positive.
    rng = np.random.default_rng(0)
    return rng.normal(0.0012, 0.010, 2000)


def test_sharpe_positive_for_positive_drift(rets):
    assert risk.sharpe_ratio(rets, rf=0.0) > 0


def test_sortino_at_least_sharpe_for_symmetric(rets):
    # Sortino uses only downside dev, so it's >= Sharpe for symmetric returns.
    assert risk.sortino_ratio(rets, 0.02) >= risk.sharpe_ratio(rets, 0.02) - 1e-9


def test_var_is_negative_and_99_worse_than_95(rets):
    v95 = risk.value_at_risk(rets, 0.95)
    v99 = risk.value_at_risk(rets, 0.99)
    assert v95 < 0 and v99 < 0
    assert v99 < v95  # 99% VaR is a deeper loss


def test_monte_carlo_var_close_to_historical(rets):
    hist = risk.value_at_risk(rets, 0.95, method="historical")
    mc = risk.value_at_risk(rets, 0.95, method="monte_carlo")
    assert mc == pytest.approx(hist, abs=0.005)


def test_cvar_worse_than_var(rets):
    var = risk.value_at_risk(rets, 0.95)
    cvar = risk.conditional_var(rets, 0.95)
    assert cvar <= var  # expected shortfall is at least as bad as VaR


def test_var_unknown_method_raises(rets):
    with pytest.raises(ValueError):
        risk.value_at_risk(rets, 0.95, method="bogus")


def test_max_drawdown_known_series():
    prices = np.array([100.0, 120.0, 60.0, 80.0, 130.0])  # peak 120 -> trough 60 = -50%
    out = risk.max_drawdown(prices)
    assert out["max_drawdown"] == pytest.approx(-0.5)
    assert out["peak_idx"] == 1
    assert out["trough_idx"] == 2
    assert out["recovery_days"] == 2  # index 4 (130 >= 120) is 2 steps past trough


def test_max_drawdown_never_recovers():
    prices = np.array([100.0, 90.0, 80.0, 70.0])
    out = risk.max_drawdown(prices)
    assert math.isnan(out["recovery_days"])
