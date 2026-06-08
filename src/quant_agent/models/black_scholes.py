"""Black-Scholes closed-form pricing, Greeks, and implied volatility.

PDE:  dV/dt + 1/2 sigma^2 S^2 d2V/dS2 + r S dV/dS - r V = 0

Closed-form European solution:
    d1 = (ln(S/K) + (r + sigma^2/2) T) / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)
    Call = S N(d1) - K e^{-rT} N(d2)
    Put  = K e^{-rT} N(-d2) - S N(-d1)

Critical signal: realized vol (GBM sigma) vs ATM implied vol.
    IV - sigma > 5pp  -> "vol premium"  (options overpriced)
    sigma - IV > 5pp  -> "vol discount" (options underpriced)
"""

from __future__ import annotations

import math
from typing import Literal

import pandas as pd
from scipy import stats

from quant_agent.config import (
    IV_INITIAL_GUESS,
    IV_MAX_ITER,
    IV_TOLERANCE,
    VOL_PREMIUM_THRESHOLD_PP,
)

OptionType = Literal["call", "put"]

_N = stats.norm.cdf
_n = stats.norm.pdf


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European call price."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return float(S * _N(d1) - K * math.exp(-r * T) * _N(d2))


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European put price."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return float(K * math.exp(-r * T) * _N(-d2) - S * _N(-d1))


def bs_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType
) -> float:
    """Dispatch to call/put pricing by option type."""
    return bs_call(S, K, T, r, sigma) if option_type == "call" else bs_put(S, K, T, r, sigma)


def greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType
) -> dict[str, float]:
    """Return Delta, Gamma, Vega, Theta, Rho.

    Vega and Rho are per 1.00 (100%) change in vol/rate; Theta is per year.
    Divide Vega/Rho by 100 and Theta by 252 for per-1pp / per-day conventions.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    sqrt_t = math.sqrt(T)
    disc = math.exp(-r * T)
    gamma = _n(d1) / (S * sigma * sqrt_t)
    vega = S * _n(d1) * sqrt_t

    if option_type == "call":
        delta = _N(d1)
        theta = -(S * _n(d1) * sigma) / (2 * sqrt_t) - r * K * disc * _N(d2)
        rho = K * T * disc * _N(d2)
    else:
        delta = _N(d1) - 1.0
        theta = -(S * _n(d1) * sigma) / (2 * sqrt_t) + r * K * disc * _N(-d2)
        rho = -K * T * disc * _N(-d2)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
) -> float:
    """Implied vol via Newton-Raphson with a bisection fallback.

    Returns NaN if the price violates no-arbitrage bounds or fails to converge.
    """
    disc = math.exp(-r * T)
    intrinsic = max(S - K * disc, 0.0) if option_type == "call" else max(K * disc - S, 0.0)
    upper_bound = S if option_type == "call" else K * disc
    if not (intrinsic - 1e-9 <= market_price <= upper_bound + 1e-9):
        return float("nan")

    # Newton-Raphson seeded at the config guess.
    sigma = IV_INITIAL_GUESS
    for _ in range(IV_MAX_ITER):
        price = bs_price(S, K, T, r, sigma, option_type)
        vega = S * _n(_d1_d2(S, K, T, r, sigma)[0]) * math.sqrt(T)
        diff = price - market_price
        if abs(diff) < IV_TOLERANCE:
            return float(sigma)
        if vega < 1e-12:
            break  # flat — hand off to bisection
        sigma -= diff / vega
        if sigma <= 0 or sigma > 10:
            break  # diverged — hand off to bisection

    # Bisection fallback on a wide bracket.
    lo, hi = 1e-6, 10.0
    for _ in range(IV_MAX_ITER):
        mid = 0.5 * (lo + hi)
        diff = bs_price(S, K, T, r, mid, option_type) - market_price
        if abs(diff) < IV_TOLERANCE:
            return float(mid)
        if diff > 0:
            hi = mid
        else:
            lo = mid
    return float("nan")


def iv_surface(options_chain: pd.DataFrame) -> pd.DataFrame:
    """Extract implied vol across strikes and expirations.

    Expects columns: expiration, type, strike, iv. Returns the chain with any
    non-positive / missing IVs dropped, sorted by (expiration, strike).
    """
    df = options_chain.copy()
    df = df[df["iv"].notna() & (df["iv"] > 0)]
    return df.sort_values(["expiration", "strike"]).reset_index(drop=True)


def atm_implied_vol(options_chain: pd.DataFrame, spot: float) -> float:
    """ATM implied vol from the nearest expiration, strike closest to spot."""
    df = iv_surface(options_chain)
    if df.empty:
        return float("nan")
    nearest_exp = sorted(df["expiration"].unique())[0]
    leg = df[df["expiration"] == nearest_exp].copy()
    leg["dist"] = (leg["strike"] - spot).abs()
    atm = leg.loc[leg["dist"].idxmin()]
    return float(atm["iv"])


def vol_signal(realized_sigma: float, implied_vol: float) -> dict[str, float | str]:
    """Compare realized vs implied vol; classify premium/discount/neutral.

    Returns a dict with the gap (IV - realized, in decimal) and a label.
    """
    if math.isnan(implied_vol):
        return {"gap": float("nan"), "label": "no-options"}
    gap = implied_vol - realized_sigma
    if gap > VOL_PREMIUM_THRESHOLD_PP:
        label = "vol premium"        # options overpriced -> mildly bearish
    elif gap < -VOL_PREMIUM_THRESHOLD_PP:
        label = "vol discount"       # options underpriced -> mildly bullish
    else:
        label = "vol neutral"
    return {"gap": float(gap), "label": label}


__all__ = [
    "bs_call",
    "bs_put",
    "bs_price",
    "greeks",
    "implied_volatility",
    "iv_surface",
    "atm_implied_vol",
    "vol_signal",
]


if __name__ == "__main__":  # pragma: no cover
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.30
    call = bs_call(S, K, T, r, sigma)
    recovered = implied_volatility(call, S, K, T, r, "call")
    g = greeks(S, K, T, r, sigma, "call")
    print(f"call={call:.4f}  IV(recovered)={recovered:.4f} (true {sigma})")
    print(f"greeks={ {k: round(v, 4) for k, v in g.items()} }")
    # Put-call parity check: C - P = S - K e^{-rT}
    parity = bs_call(S, K, T, r, sigma) - bs_put(S, K, T, r, sigma)
    print(f"parity LHS={parity:.4f}  RHS={S - K*math.exp(-r*T):.4f}")
