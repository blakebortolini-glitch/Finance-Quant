"""Smoke tests confirming the package imports and config is internally consistent.

Real model tests (test_gbm, test_black_scholes, test_capm, ...) land in
Milestone 1 as each module is implemented.
"""

from __future__ import annotations

from quant_agent.config import (
    ER_BLEND_ANALYST_TARGET,
    ER_BLEND_EARNINGS_YIELD,
    ER_BLEND_GBM_DRIFT,
    SCORING_WEIGHTS,
)


def test_package_imports() -> None:
    import quant_agent  # noqa: F401

    assert quant_agent.__version__


def test_scoring_weights_sum_to_one() -> None:
    SCORING_WEIGHTS.validate()  # raises if not 1.0


def test_er_blend_weights_sum_to_one() -> None:
    total = ER_BLEND_ANALYST_TARGET + ER_BLEND_GBM_DRIFT + ER_BLEND_EARNINGS_YIELD
    assert abs(total - 1.0) < 1e-9
