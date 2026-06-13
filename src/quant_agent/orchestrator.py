"""Pipeline coordinator: fetch -> model -> score -> report.

Two-pass design. Pass 1 fetches each ticker and computes the per-ticker
signals (GBM upside, CAPM alpha, GARCH-vs-implied vol, DCF, Piotroski, Altman),
collecting daily returns and expected returns along the way. Pass 2 solves the
watchlist-wide max-Sharpe (Markowitz) portfolio so each ticker's optimal weight
can enter as the seventh signal, then scores every ticker.

Per-ticker failures are logged and skipped so one bad symbol never aborts the
run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

from quant_agent.config import (
    CAPM_BETA_LOOKBACK_YEARS,
    FINANCIAL_SECTORS,
    GBM_DRIFT_WINDOW,
    GBM_HORIZON_DAYS,
    GBM_N_SIMS,
    GBM_RANDOM_SEED,
    GBM_UPSIDE_THRESHOLD,
    MARKET_RISK_PREMIUM,
    REPORTS_DIR,
    SMART_MONEY_ENABLED,
    TRADING_DAYS_PER_YEAR,
)
from quant_agent.data import edgar, supabase_reader, supabase_writer
from quant_agent.data.fetcher import fetch_ticker
from quant_agent.data.schemas import TickerData
from quant_agent.models import black_scholes as bs
from quant_agent.models import (
    capm,
    garch,
    gbm,
    insider,
    institutional,
    kelly,
    markowitz,
    risk,
    valuation,
)
from quant_agent.scoring.engine import (
    _WEIGHT_OF,
    ScoreResult,
    normalize_alpha,
    normalize_altman,
    normalize_dcf,
    normalize_insider,
    normalize_institutional,
    normalize_markowitz,
    normalize_piotroski,
    normalize_upside,
    normalize_vol,
    score_ticker,
)

VERDICT_HISTORY_PATH = REPORTS_DIR / "verdict_history.json"


@dataclass
class RunResult:
    as_of: date
    results: list[ScoreResult]
    portfolio: dict = field(default_factory=dict)  # max-Sharpe weights & stats


def _load_history() -> dict[str, dict]:
    """Load the last-recorded verdict per ticker (ticker -> {as_of, verdict})."""
    if not VERDICT_HISTORY_PATH.exists():
        return {}
    try:
        return json.loads(VERDICT_HISTORY_PATH.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not read verdict history: {exc}")
        return {}


def _save_history(results: list[ScoreResult], as_of: date, history: dict[str, dict]) -> None:
    """Persist the current verdicts, merging into the existing history."""
    for r in results:
        history[r.ticker] = {
            "as_of": as_of.isoformat(),
            "verdict": r.verdict.value,
            "composite": r.composite,
        }
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        VERDICT_HISTORY_PATH.write_text(json.dumps(history, indent=2, sort_keys=True))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not write verdict history: {exc}")


@dataclass
class _Analysis:
    """Interim per-ticker analysis carried between the two passes."""

    ticker: str
    price: float
    signals: dict[str, float]          # normalized, pre-Markowitz
    metrics: dict
    daily_returns: pd.Series
    expected_return: float             # annualized, for Markowitz


def _weekly_returns(prices: pd.DataFrame, years: int) -> pd.Series:
    """Weekly (Friday) close-to-close returns over the trailing ``years``."""
    weekly = prices["Close"].resample("W-FRI").last().dropna()
    cutoff = weekly.index.max() - pd.DateOffset(years=years)
    weekly = weekly[weekly.index >= cutoff]
    return weekly.pct_change().dropna()


def _earnings_yield(td: TickerData, price: float) -> float | None:
    """Trailing earnings yield = (net_income / shares) / price."""
    f = td.fundamentals
    if f.net_income and f.shares_outstanding and price > 0:
        return (f.net_income / f.shares_outstanding) / price
    return None


def _analyze(td: TickerData, funds: list[dict], smart_money_enabled: bool) -> _Analysis:
    """Run the per-ticker model set and assemble its (pre-Markowitz) signals."""
    close = td.prices["Close"].to_numpy(dtype=np.float64)
    S0 = float(close[-1])
    prev_close = float(close[-2]) if close.size >= 2 else None
    daily_returns = td.prices["Close"].pct_change().dropna()

    # --- GBM: drift/vol estimate + Monte Carlo upside probability ---------- #
    mu, sigma = gbm.estimate_drift_volatility(close, window=GBM_DRIFT_WINDOW)
    T = GBM_HORIZON_DAYS / TRADING_DAYS_PER_YEAR
    paths = gbm.simulate_paths(
        S0, mu, sigma, T=T, steps=GBM_HORIZON_DAYS, n_sims=GBM_N_SIMS, seed=GBM_RANDOM_SEED
    )
    upside_prob = gbm.prob_above(paths, S0 * GBM_UPSIDE_THRESHOLD)
    path_stats = gbm.path_statistics(paths, S0)
    ud_ratio = gbm.upside_downside_ratio(paths, S0)

    # --- CAPM: beta, required return, actual-ER blend, alpha --------------- #
    stock_r = _weekly_returns(td.prices, CAPM_BETA_LOOKBACK_YEARS)
    mkt_r = _weekly_returns(td.market_prices, CAPM_BETA_LOOKBACK_YEARS)
    aligned = pd.concat([stock_r, mkt_r], axis=1, keys=["stock", "mkt"]).dropna()
    fit = capm.estimate_beta(aligned["stock"].to_numpy(), aligned["mkt"].to_numpy())
    beta = fit["beta"]
    rf = td.macro.risk_free_rate
    required_er = capm.expected_return(beta, rf, MARKET_RISK_PREMIUM)

    analyst_ret = None
    if td.fundamentals.analyst_target_price:
        analyst_ret = td.fundamentals.analyst_target_price / S0 - 1.0
    actual_er = capm.actual_expected_return(
        analyst_target_return=analyst_ret,
        gbm_drift=mu,
        earnings_yield=_earnings_yield(td, S0),
    )
    alpha = capm.alpha_signal(actual_er, required_er)

    # --- GARCH forward vol -> Black-Scholes vol signal --------------------- #
    garch_sigma = garch.forecast_volatility(daily_returns.to_numpy())
    forward_sigma = garch_sigma if np.isfinite(garch_sigma) else sigma
    atm_iv = bs.atm_implied_vol(td.options.data, td.options.spot) if td.options else float("nan")
    vsig = bs.vol_signal(realized_sigma=forward_sigma, implied_vol=atm_iv)

    # --- Fundamental valuation: DCF, Piotroski, Altman -------------------- #
    intrinsic = valuation.dcf_intrinsic_value(td.fundamentals, beta, rf, MARKET_RISK_PREMIUM)
    f_score = valuation.piotroski_f_score(td.fundamentals)
    market_cap = (td.fundamentals.shares_outstanding or 0.0) * S0
    # Altman Z is unreliable for financials/REITs — skip it for those sectors
    # entirely so the signal is treated as missing and weights renormalize.
    sector = td.fundamentals.sector
    is_financial = sector in FINANCIAL_SECTORS
    z_score = None if is_financial else valuation.altman_z_score(td.fundamentals, market_cap)

    # --- Risk metrics (trailing, daily) ----------------------------------- #
    r_arr = daily_returns.to_numpy()
    risk_metrics = {
        "sharpe": risk.sharpe_ratio(r_arr, rf),
        "sortino": risk.sortino_ratio(r_arr, rf),
        "var_95": risk.value_at_risk(r_arr, 0.95),
        "var_99": risk.value_at_risk(r_arr, 0.99),
        "cvar_95": risk.conditional_var(r_arr, 0.95),
        "max_dd": risk.max_drawdown(close)["max_drawdown"],
    }

    # --- Kelly position sizing (continuous + discrete) -------------------- #
    kelly_cont = kelly.continuous_kelly(actual_er, sigma**2, rf)
    kelly_disc = kelly.kelly_fraction(upside_prob, ud_ratio if np.isfinite(ud_ratio) else 0.0)
    kelly_sized = kelly.fractional_kelly(kelly_cont)

    # --- Assemble normalized signals (Markowitz added in pass 2) ---------- #
    signals: dict[str, float] = {
        "capm_alpha": normalize_alpha(alpha),
        "gbm_upside": normalize_upside(upside_prob),
        "piotroski": normalize_piotroski(f_score),
    }
    if np.isfinite(vsig["gap"]):
        signals["vol_premium"] = normalize_vol(float(vsig["gap"]))
    if intrinsic is not None:
        signals["dcf_intrinsic"] = normalize_dcf(intrinsic, S0)
    if z_score is not None:
        signals["altman_z"] = normalize_altman(z_score)

    # --- Smart Money layer (insider Form 4 + 13F institutional) ----------- #
    # Computed and reported always (when enabled); folded into the composite
    # only when its config weight is > 0 ("composite-ready, off by default").
    smart_money = _smart_money_layer(td, funds, smart_money_enabled)
    for name in ("insider", "institutional"):
        sm = smart_money.get(name)
        if sm and sm["has_signal"] and _WEIGHT_OF.get(name, 0.0) > 0:
            signals[name] = sm["normalized"]

    metrics = {
        "mu": mu, "sigma": sigma, "garch_sigma": garch_sigma, "forward_sigma": forward_sigma,
        "beta": beta, "r_squared": fit["r_squared"], "rf": rf,
        "required_er": required_er, "actual_er": actual_er, "alpha": alpha,
        "upside_prob": upside_prob, "ud_ratio": ud_ratio,
        "atm_iv": atm_iv, "vol_gap": vsig["gap"], "vol_label": vsig["label"],
        "dcf_intrinsic": intrinsic, "dcf_fcf_growth": valuation.fcf_growth_rate(td.fundamentals),
        "piotroski": f_score, "altman_z": z_score,
        "sector": sector, "is_financial": is_financial,
        "name": td.fundamentals.name, "prev_close": prev_close,
        "market_cap": market_cap,
        "kelly_continuous": kelly_cont, "kelly_discrete": kelly_disc,
        "kelly_sized": kelly_sized,
        "horizon_days": GBM_HORIZON_DAYS,
        "smart_money": smart_money,
        **risk_metrics,
        **{f"gbm_{k}": v for k, v in path_stats.items()},
    }

    return _Analysis(
        ticker=td.ticker,
        price=round(S0, 2),
        signals=signals,
        metrics=metrics,
        daily_returns=daily_returns,
        expected_return=actual_er,
    )


def _smart_money_layer(td: TickerData, funds: list[dict], enabled: bool) -> dict:
    """Compute the insider + institutional signals for one ticker.

    Returns a dict with 'insider' and 'institutional' sub-dicts (each carrying
    a normalized [-1,1] score and raw counts) plus a combined 'score'. When the
    layer is disabled, returns an empty/neutral structure.
    """
    if not enabled:
        return {"enabled": False}

    ins = insider.insider_signal(
        edgar.fetch_insider_transactions(td.ticker, td.as_of), as_of=td.as_of
    )
    ins["normalized"] = normalize_insider(ins["score"])

    inst = institutional.institutional_signal(td.ticker, funds)
    inst["normalized"] = normalize_institutional(inst["score"])

    combined_parts = [s["normalized"] for s in (ins, inst) if s["has_signal"]]
    combined = float(np.mean(combined_parts)) if combined_parts else 0.0
    return {"enabled": True, "insider": ins, "institutional": inst, "score": combined}


def _solve_portfolio(analyses: list[_Analysis], rf: float) -> dict:
    """Max-Sharpe portfolio across the given analyses (best-effort).

    Callers pass only the watchlist analyses so the efficient frontier stays
    meaningful even when the scored universe is expanded with user holdings.
    """
    if len(analyses) < 2:
        return {}
    returns = pd.concat(
        {a.ticker: a.daily_returns for a in analyses}, axis=1
    ).dropna()
    if returns.shape[0] < 30 or returns.shape[1] < 2:
        return {}
    try:
        cov = markowitz.covariance_matrix(returns)
        er = pd.Series({a.ticker: a.expected_return for a in analyses})[returns.columns]
        result = markowitz.max_sharpe_portfolio(er, cov, rf)
        result["frontier"] = markowitz.efficient_frontier(er, cov)
        result["min_variance"] = markowitz.min_variance_portfolio(cov)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Markowitz portfolio optimization failed: {exc}")
        return {}


def run(
    tickers: list[str],
    as_of: date | None = None,
    detail: str = "summary",
    changed_only: bool = False,
    smart_money: bool | None = None,
    include_holdings: bool = False,
    watchlist: list[str] | None = None,
) -> RunResult:
    """Execute the pipeline for the given tickers (failures logged & skipped).

    The *scored universe* is ``tickers``, optionally unioned with every
    user-held ticker (``include_holdings``). ``watchlist`` is the curated set
    used to tag verdicts (``in_watchlist``) and to scope the Markowitz
    portfolio; it defaults to ``tickers`` when not given.
    """
    as_of = as_of or date.today()
    smart_money_enabled = SMART_MONEY_ENABLED if smart_money is None else smart_money
    watchlist_set = {t.upper() for t in (watchlist if watchlist is not None else tickers)}

    # Scored universe = requested tickers (+ all user holdings under option A).
    scored = list(dict.fromkeys(t.upper() for t in tickers))
    if include_holdings:
        for t in supabase_reader.fetch_held_tickers():
            if t not in scored:
                scored.append(t)
    logger.info(
        f"Scoring {len(scored)} tickers ({len(watchlist_set & set(scored))} watchlist "
        f"+ {len(scored) - len(watchlist_set & set(scored))} holdings-only)"
    )

    # --- Plaid auto-sync: refresh all users' holdings before scoring -------- #
    # Runs silently on failure so one bad Plaid token never aborts the run.
    try:
        from quant_agent.data import plaid_sync
        plaid_sync.sync_all_plaid_connections()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Plaid auto-sync failed (non-fatal, scoring continues): {exc}")

    # --- Smart Money: fetch tracked-fund 13F holdings once for the run ----- #
    funds: list[dict] = []
    if smart_money_enabled:
        funds = edgar.fetch_all_fund_holdings(as_of)

    # --- Pass 1: per-ticker analysis -------------------------------------- #
    analyses: list[_Analysis] = []
    for ticker in scored:
        try:
            td = fetch_ticker(ticker, as_of)
            analyses.append(_analyze(td, funds, smart_money_enabled))
        except Exception as exc:  # noqa: BLE001 - isolate per-ticker failures
            logger.error(f"{ticker}: analysis failed: {exc}")

    # --- Pass 2: portfolio optimization (watchlist only) + scoring -------- #
    rf = analyses[0].metrics["rf"] if analyses else 0.0
    watchlist_analyses = [a for a in analyses if a.ticker in watchlist_set]
    portfolio = _solve_portfolio(watchlist_analyses, rf)
    weights = portfolio.get("weights", pd.Series(dtype=float))
    n_assets = len(watchlist_analyses)

    results: list[ScoreResult] = []
    for a in analyses:
        signals = dict(a.signals)
        weight = float(weights.get(a.ticker, float("nan"))) if len(weights) else float("nan")
        if np.isfinite(weight):
            signals["markowitz_weight"] = normalize_markowitz(weight, n_assets)
            a.metrics["markowitz_weight"] = weight

        result = score_ticker(a.ticker, signals)
        result.price = a.price
        result.metrics = a.metrics
        result.in_watchlist = a.ticker in watchlist_set
        results.append(result)
        logger.success(
            f"{a.ticker}: {result.verdict.value} ({result.composite:+.1f}, "
            f"{result.confidence.value})"
        )

    # --- Verdict-change tracking (drives --changed-only) ------------------ #
    history = _load_history()
    for r in results:
        prev = history.get(r.ticker)
        if prev is not None:
            r.previous_verdict = prev.get("verdict")
            r.changed = r.previous_verdict != r.verdict.value
        else:
            r.changed = True  # no prior record -> treat as newly surfaced
    _save_history(results, as_of, history)

    # --- Optional Supabase sink (full result set, before --changed-only) --- #
    # No-ops when SUPABASE_SERVICE_KEY is absent; never raises into the run.
    supabase_writer.write_run_results(results, portfolio, as_of)

    if changed_only:
        before = len(results)
        results = [r for r in results if r.changed]
        logger.info(f"--changed-only: {len(results)}/{before} tickers changed verdict")

    results.sort(key=lambda r: r.composite, reverse=True)
    return RunResult(as_of=as_of, results=results, portfolio=portfolio)


__all__ = ["RunResult", "run"]
