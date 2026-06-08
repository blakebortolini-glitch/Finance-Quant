"""GARCH(1,1) one-day-ahead volatility forecast (via ``arch``).

Fits GARCH(1,1) to daily percentage log-returns and forecasts the h-day-ahead
conditional variance, returned as an *annualized* volatility. The forecast
feeds Black-Scholes for forward-looking option valuation.

Returns are scaled to percent (x100) before fitting, which is the convention
the ``arch`` optimizer is numerically happiest with; the resulting variance is
unscaled (/100^2) before annualizing.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from arch import arch_model
from loguru import logger

from quant_agent.config import GARCH_FORECAST_HORIZON, GARCH_P, GARCH_Q, TRADING_DAYS_PER_YEAR

FloatArray = npt.NDArray[np.float64]


def forecast_volatility(
    returns: FloatArray, horizon: int = GARCH_FORECAST_HORIZON
) -> float:
    """Fit GARCH(1,1) and return the annualized h-day-ahead vol forecast.

    Args:
        returns: 1-D array of daily simple/log returns (decimals), shape (n,).
        horizon: forecast horizon in trading days.

    Returns:
        Annualized volatility (decimal). Returns NaN if the fit fails.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 100:
        logger.debug(f"GARCH: only {r.size} returns; need >=100 for a stable fit")
        return float("nan")

    try:
        model = arch_model(r * 100.0, vol="GARCH", p=GARCH_P, q=GARCH_Q, mean="Constant")
        fit = model.fit(disp="off")
        forecast = fit.forecast(horizon=horizon, reindex=False)
        # Variance of the final forecast step, in percent^2 -> decimal^2.
        daily_var = float(forecast.variance.iloc[-1, -1]) / (100.0**2)
        return float(np.sqrt(daily_var * TRADING_DAYS_PER_YEAR))
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning(f"GARCH fit failed: {exc}")
        return float("nan")


__all__ = ["forecast_volatility"]


if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(0)
    # Synthetic returns with mild volatility clustering.
    n = 1000
    vol = 0.01 * np.ones(n)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        vol[t] = np.sqrt(1e-6 + 0.1 * (vol[t - 1] * eps[t - 1]) ** 2 + 0.85 * vol[t - 1] ** 2)
    rets = vol * eps
    print(f"GARCH(1,1) annualized 1d-ahead vol forecast = {forecast_volatility(rets):.2%}")
