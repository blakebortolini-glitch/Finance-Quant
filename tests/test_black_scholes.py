"""Tests for Black-Scholes pricing, Greeks, implied vol, and the vol signal."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from quant_agent.models import black_scholes as bs


def test_put_call_parity():
    S, K, T, r, sigma = 100.0, 95.0, 0.75, 0.04, 0.28
    lhs = bs.bs_call(S, K, T, r, sigma) - bs.bs_put(S, K, T, r, sigma)
    rhs = S - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_call_known_value():
    # ATM, r=0: call = S(2N(sigma sqrt(T)/2) - 1)
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.0, 0.20
    from scipy.stats import norm

    expected = S * (2 * norm.cdf(sigma * math.sqrt(T) / 2) - 1)
    assert bs.bs_call(S, K, T, r, sigma) == pytest.approx(expected, abs=1e-6)


def test_implied_vol_roundtrip():
    S, K, T, r, sigma = 100.0, 110.0, 0.5, 0.05, 0.35
    for opt in ("call", "put"):
        price = bs.bs_price(S, K, T, r, sigma, opt)
        recovered = bs.implied_volatility(price, S, K, T, r, opt)
        assert recovered == pytest.approx(sigma, abs=1e-4)


def test_implied_vol_out_of_bounds_returns_nan():
    # Price above the no-arbitrage upper bound (spot) for a call.
    assert math.isnan(bs.implied_volatility(150.0, 100.0, 100.0, 0.5, 0.05, "call"))


def test_greeks_signs_and_gamma_symmetry():
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.25
    c = bs.greeks(S, K, T, r, sigma, "call")
    p = bs.greeks(S, K, T, r, sigma, "put")
    assert 0 < c["delta"] < 1
    assert -1 < p["delta"] < 0
    assert c["gamma"] > 0 and c["vega"] > 0
    assert c["gamma"] == pytest.approx(p["gamma"])  # gamma identical call/put
    assert c["theta"] < 0  # long call decays


def test_vol_signal_classification():
    assert bs.vol_signal(0.30, 0.40)["label"] == "vol premium"   # IV >> realized
    assert bs.vol_signal(0.40, 0.30)["label"] == "vol discount"  # IV << realized
    assert bs.vol_signal(0.30, 0.31)["label"] == "vol neutral"
    assert bs.vol_signal(0.30, float("nan"))["label"] == "no-options"


def test_atm_implied_vol_picks_nearest_strike():
    chain = pd.DataFrame(
        {
            "expiration": ["2026-07-17"] * 3,
            "type": ["call"] * 3,
            "strike": [90.0, 100.0, 110.0],
            "iv": [0.45, 0.33, 0.41],
        }
    )
    assert bs.atm_implied_vol(chain, spot=101.0) == pytest.approx(0.33)
