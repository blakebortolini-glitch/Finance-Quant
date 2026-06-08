"""Tests for CAPM beta regression, required return, and the ER blend."""

from __future__ import annotations

import numpy as np
import pytest

from quant_agent.models import capm


def test_estimate_beta_recovers_known_slope():
    rng = np.random.default_rng(0)
    mkt = rng.normal(0.001, 0.02, 500)
    true_beta, true_alpha = 1.4, 0.0008
    stock = true_alpha + true_beta * mkt + rng.normal(0, 0.003, 500)
    fit = capm.estimate_beta(stock, mkt)
    assert fit["beta"] == pytest.approx(true_beta, abs=0.05)
    assert fit["alpha"] == pytest.approx(true_alpha, abs=0.002)
    assert 0.0 <= fit["r_squared"] <= 1.0


def test_expected_return_formula():
    assert capm.expected_return(beta=1.0, rf=0.05, market_premium=0.06) == pytest.approx(0.11)
    assert capm.expected_return(beta=0.0, rf=0.05, market_premium=0.06) == pytest.approx(0.05)


def test_actual_expected_return_blend_weights():
    # All present -> weighted average with config weights (0.40/0.35/0.25).
    blended = capm.actual_expected_return(0.20, 0.10, 0.05)
    expected = 0.40 * 0.20 + 0.35 * 0.10 + 0.25 * 0.05
    assert blended == pytest.approx(expected)


def test_actual_expected_return_renormalizes_when_missing():
    # Only analyst + gbm -> weights renormalized over the two present.
    blended = capm.actual_expected_return(0.20, 0.10, None)
    expected = (0.40 * 0.20 + 0.35 * 0.10) / (0.40 + 0.35)
    assert blended == pytest.approx(expected)


def test_actual_expected_return_raises_when_all_missing():
    with pytest.raises(ValueError):
        capm.actual_expected_return(None, None, None)


def test_alpha_signal_sign():
    assert capm.alpha_signal(0.15, 0.10) > 0   # undervalued
    assert capm.alpha_signal(0.08, 0.12) < 0   # overvalued


def test_estimate_beta_shape_mismatch_raises():
    with pytest.raises(ValueError):
        capm.estimate_beta(np.ones(10), np.ones(9))
