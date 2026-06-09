"""Tier 3 report: full quant breakdown per ticker.

CLI: ``--detail full``. Written to ``reports/YYYY-MM-DD_detailed.md``.
Sections: Stochastic Models, Valuation & Risk, Risk Metrics, Position Sizing.
Appends a portfolio-level summary with an efficient-frontier PNG, max-Sharpe
weights table, and total expected return/risk.
"""

from __future__ import annotations

import math

from quant_agent.config import GBM_HORIZON_DAYS, REPORTS_DIR
from quant_agent.models import black_scholes as bs
from quant_agent.orchestrator import RunResult
from quant_agent.scoring.engine import ScoreResult

_ATM_OPTION_DAYS = 30


def _pct(x: float | None, signed: bool = True, dp: int = 1) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x:+.{dp}%}" if signed else f"{x:.{dp}%}"


def _usd(x: float | None) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"${x:,.2f}"


def _atm_greeks(r: ScoreResult) -> dict[str, float] | None:
    """ATM 30d call Greeks from the forward vol, for display."""
    m = r.metrics
    sigma = m.get("forward_sigma")
    if not r.price or sigma is None or not math.isfinite(sigma) or sigma <= 0:
        return None
    T = _ATM_OPTION_DAYS / 365.0
    return bs.greeks(r.price, r.price, T, m.get("rf", 0.0), sigma, "call")


def _disagreeing_signals(r: ScoreResult) -> str:
    sign = 1 if r.composite >= 0 else -1
    names = [s.name for s in r.signals if s.direction != 0 and s.direction != sign]
    return ", ".join(names) if names else "none"


def _ticker_block(r: ScoreResult) -> str:
    m = r.metrics
    g = _atm_greeks(r)
    dcf = m.get("dcf_intrinsic")
    dcf_upside = (dcf / r.price - 1.0) if dcf is not None and r.price else None
    z = m.get("altman_z")
    z_str = (
        "skipped (financial sector)" if z is None and m.get("is_financial")
        else "n/a" if z is None else f"{z:.2f}"
    )
    atm_iv = m.get("atm_iv")
    iv_str = _pct(atm_iv, signed=False) if atm_iv is not None and math.isfinite(atm_iv) else "n/a"
    greeks_line = (
        f"  Δ = {g['delta']:.3f}   Γ = {g['gamma']:.4f}   "
        f"Θ = {g['theta'] / 252:.3f}/day   Vega = {g['vega'] / 100:.3f}/1pp"
        if g else "  (no option Greeks — vol unavailable)"
    )

    n_total = r.n_agree + r.n_disagree
    lines = [
        "```",
        "═══════════════════════════════════════════════",
        f"  {r.ticker} — {m.get('horizon_days', GBM_HORIZON_DAYS)}d horizon analysis",
        "═══════════════════════════════════════════════",
        "",
        f"PRICE         {_usd(r.price)} (prev close)",
        f"VERDICT       {r.verdict.value}    (confidence: {r.confidence.value.upper()})",
        f"COMPOSITE     {r.composite:+.1f}",
        "",
        "──── Stochastic Models ─────────────────────────",
        f"GBM (Monte Carlo, {m.get('horizon_days')}d horizon)",
        f"  P5  {_usd(m.get('gbm_p5'))} │ P50 {_usd(m.get('gbm_p50'))} │ "
        f"P95 {_usd(m.get('gbm_p95'))}",
        f"  P(S_T > S_0)    = {_pct(m.get('gbm_prob_above_s0'), signed=False)}",
        f"  Expected return = {_pct(m.get('gbm_expected_return'))} over {m.get('horizon_days')}d",
        "",
        "Black-Scholes (ATM ~30d)",
        f"  Realized σ: {_pct(m.get('sigma'), signed=False)} │ "
        f"GARCH σ: {_pct(m.get('garch_sigma'), signed=False)} │ Implied σ: {iv_str}",
        f"  Vol signal: {m.get('vol_label', 'n/a')} ({_pct(m.get('vol_gap'))} vs realized)",
        greeks_line,
        "",
        "──── Valuation & Risk ──────────────────────────",
        f"CAPM: β = {m.get('beta', float('nan')):.2f}, R_f = {_pct(m.get('rf'), signed=False)}, "
        f"Required ER = {_pct(m.get('required_er'))}",
        f"      Actual ER = {_pct(m.get('actual_er'))}, Alpha = {_pct(m.get('alpha'))}",
        f"DCF Intrinsic = {_usd(dcf)} ({_pct(dcf_upside)} upside)  "
        f"[FCF growth {_pct(m.get('dcf_fcf_growth'), signed=False)}]",
        f"Piotroski F-Score: {m.get('piotroski')}/9 │ Altman Z-Score: {z_str}",
        "",
        "──── Risk Metrics (trailing) ───────────────────",
        f"Sharpe {m.get('sharpe', float('nan')):.2f} │ Sortino {m.get('sortino', float('nan')):.2f}"
        f" │ Max DD {_pct(m.get('max_dd'), signed=False, dp=1)}",
        f"VaR(95) {_pct(m.get('var_95'), dp=2)}/day │ VaR(99) {_pct(m.get('var_99'), dp=2)}/day │ "
        f"CVaR(95) {_pct(m.get('cvar_95'), dp=2)}/day",
        "",
        "──── Position Sizing ───────────────────────────",
        f"Kelly fraction (continuous): {_pct(m.get('kelly_continuous'), signed=False)}",
        f"Fractional Kelly (¼, capped): {_pct(m.get('kelly_sized'), signed=False)}",
        f"Markowitz max-Sharpe weight: {_pct(m.get('markowitz_weight'), signed=False)}",
    ]
    lines.extend(_smart_money_lines(r))
    lines += [
        "",
        f"SIGNALS AGREE: {r.n_agree}/{n_total}  │  DISAGREE: {_disagreeing_signals(r)}",
        "```",
        "",
    ]
    return "\n".join(lines)


def _smart_money_lines(r: ScoreResult) -> list[str]:
    """Smart Money section for the detailed block (insider Form 4 + 13F)."""
    sm = r.metrics.get("smart_money")
    if not sm or not sm.get("enabled"):
        return []
    out = ["", "──── Smart Money (SEC EDGAR) ────────────────────"]

    ins = sm.get("insider", {})
    if ins.get("bull_cluster"):
        status = f"BUY CLUSTER — {ins['n_cluster_buyers']} distinct insiders purchasing"
    elif ins.get("bear_cluster"):
        status = (
            f"C-SUITE SELL CLUSTER — {ins['n_cluster_csuite_sellers']} "
            "selling off-plan (non-10b5-1)"
        )
    else:
        status = "neutral — no qualifying cluster (routine activity)"
    out.append(f"Insider (Form 4): {status}  │ score {ins.get('normalized', 0.0):+.2f}")
    out.append(
        f"  window buys {ins.get('n_buys', 0)} ({ins.get('n_buyers', 0)} insiders), "
        f"sells {ins.get('n_sells', 0)}  "
        f"[$ {_usd(ins.get('buy_value', 0))} bought / {_usd(ins.get('sell_value', 0))} sold]"
    )

    inst = sm.get("institutional", {})
    if inst.get("has_signal"):
        out.append(
            f"Institutional (13F): {inst['n_holders']} of tracked funds hold  "
            f"│ score {inst['normalized']:+.2f}"
        )
        out.append(
            f"  new {inst['n_new']}, added {inst['n_added']}, "
            f"trimmed {inst['n_trimmed']}, exited {inst['n_exited']}"
        )
        if inst.get("holders"):
            out.append(f"  holders: {', '.join(inst['holders'])}")
    else:
        out.append("Institutional (13F): no tracked fund holds this name")

    weight_note = "" if _is_in_composite() else "  (informational — weight 0 in composite)"
    out.append(f"Combined smart-money score: {sm.get('score', 0.0):+.2f}{weight_note}")
    return out


def _is_in_composite() -> bool:
    from quant_agent.config import SCORING_WEIGHTS

    return SCORING_WEIGHTS.insider > 0 or SCORING_WEIGHTS.institutional > 0


def _portfolio_section(run_result: RunResult, chart_path: str | None) -> str:
    p = run_result.portfolio
    if not p:
        return "## Portfolio\n\n_Portfolio optimization unavailable (need ≥2 tickers)._\n"

    weights = p["weights"].sort_values(ascending=False)
    lines = [
        "## Portfolio — Max-Sharpe (Tangency)",
        "",
        f"- **Expected return:** {p['expected_return']:.1%}",
        f"- **Volatility:** {p['volatility']:.1%}",
        f"- **Sharpe:** {p['sharpe']:.2f}",
        "",
        "| Ticker | Weight |",
        "|---|---:|",
    ]
    for tic, w in weights.items():
        lines.append(f"| {tic} | {w:.1%} |")
    if chart_path:
        lines += ["", f"![Efficient frontier]({chart_path})"]
    return "\n".join(lines) + "\n"


def _plot_frontier(run_result: RunResult) -> str | None:
    """Render the efficient frontier + tangency point to a PNG; return its name."""
    p = run_result.portfolio
    if not p or "frontier" not in p or p["frontier"].empty:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frontier = p["frontier"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(frontier["volatility"], frontier["target_return"], "-", color="#2563eb",
                label="Efficient frontier")
        ax.scatter([p["volatility"]], [p["expected_return"]], color="#16a34a", zorder=5,
                   s=70, label="Max-Sharpe")
        ax.set_xlabel("Volatility (annualized)")
        ax.set_ylabel("Expected return (annualized)")
        ax.set_title(f"Efficient Frontier — {run_result.as_of.isoformat()}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        name = f"{run_result.as_of.isoformat()}_frontier.png"
        fig.savefig(REPORTS_DIR / name, dpi=110)
        plt.close(fig)
        return name
    except Exception:  # noqa: BLE001 - chart is best-effort
        return None


def write_markdown(run_result: RunResult) -> str:
    """Write reports/YYYY-MM-DD_detailed.md and return its path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{run_result.as_of.isoformat()}_detailed.md"

    chart = _plot_frontier(run_result)
    parts = [f"# FinQuant — {run_result.as_of.isoformat()} Detailed Analysis", ""]
    for r in run_result.results:
        parts.append(_ticker_block(r))
    parts.append(_portfolio_section(run_result, chart))

    path.write_text("\n".join(parts) + "\n")
    return str(path)


__all__ = ["write_markdown"]
