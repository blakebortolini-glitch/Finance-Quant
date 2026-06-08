"""Tests for Kelly position sizing."""

from __future__ import annotations

import pytest

from quant_agent.config import KELLY_MAX_POSITION
from quant_agent.models import kelly


def test_kelly_fraction_classic_even_money():
    # p=0.6, b=1 -> f* = (0.6*2 - 1)/1 = 0.2
    assert kelly.kelly_fraction(0.6, 1.0) == pytest.approx(0.2)


def test_kelly_fraction_negative_edge():
    # p=0.4, b=1 -> negative -> do not bet long
    assert kelly.kelly_fraction(0.4, 1.0) < 0


def test_kelly_fraction_invalid_prob_raises():
    with pytest.raises(ValueError):
        kelly.kelly_fraction(1.5, 1.0)


def test_kelly_fraction_nonpositive_ratio_is_zero():
    assert kelly.kelly_fraction(0.6, 0.0) == 0.0
    assert kelly.kelly_fraction(0.6, float("inf")) == 0.0


def test_continuous_kelly_formula():
    # (mu - r)/sigma^2 = (0.18 - 0.05)/0.09 ~ 1.444
    assert kelly.continuous_kelly(0.18, 0.30**2, 0.05) == pytest.approx(0.13 / 0.09)


def test_continuous_kelly_zero_variance_raises():
    with pytest.raises(ValueError):
        kelly.continuous_kelly(0.1, 0.0, 0.05)


def test_fractional_kelly_applies_quarter_and_cap():
    assert kelly.fractional_kelly(0.40, fraction=0.25) == pytest.approx(0.10)
    # Huge raw Kelly clamps to the hard cap.
    assert kelly.fractional_kelly(10.0) == pytest.approx(KELLY_MAX_POSITION)


def test_fractional_kelly_floors_negative_at_zero():
    assert kelly.fractional_kelly(-0.5) == 0.0
