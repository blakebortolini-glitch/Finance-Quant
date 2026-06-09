"""Tier 2 report: 4-line plain-English summary per ticker (no jargon).

CLI: ``--detail brief``. Written to ``reports/YYYY-MM-DD_brief.md``.
Lines: Cheap on / Quality / Risk note / Size.
"""

from __future__ import annotations

import math

from quant_agent.config import ALTMAN_Z_SAFE, PIOTROSKI_STRONG, REPORTS_DIR
from quant_agent.orchestrator import RunResult
from quant_agent.scoring.engine import ScoreResult


def _fmt_pct(x: float, signed: bool = True) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def _cheap_on(r: ScoreResult) -> str:
    m = r.metrics
    parts: list[str] = []
    if m.get("dcf_intrinsic") is not None and r.price:
        upside = m["dcf_intrinsic"] / r.price - 1.0
        parts.append(f"DCF fair value ${m['dcf_intrinsic']:.0f} ({_fmt_pct(upside)})")
    else:
        parts.append("DCF n/a")
    parts.append(f"CAPM alpha {_fmt_pct(m.get('alpha', float('nan')))}")
    return ", ".join(parts)


def _quality(r: ScoreResult) -> str:
    m = r.metrics
    f_score = m.get("piotroski")
    f_label = "strong" if f_score >= PIOTROSKI_STRONG else "ok" if f_score >= 4 else "weak"
    quality = f"F-Score {f_score}/9 ({f_label})"

    z = m.get("altman_z")
    if z is None:
        z_str = "Z-Score skipped (financial)" if m.get("is_financial") else "Z-Score n/a"
    else:
        z_label = "safe" if z > ALTMAN_Z_SAFE else "grey" if z > 1.81 else "distress"
        z_str = f"Z-Score {z:.1f} ({z_label})"
    return f"{quality}, {z_str}"


def _risk_note(r: ScoreResult) -> str:
    m = r.metrics
    label = m.get("vol_label", "n/a")
    gap = m.get("vol_gap", float("nan"))
    if label == "vol premium":
        note = f"Options pricing high vol ({_fmt_pct(gap)} premium vs realized)"
    elif label == "vol discount":
        note = f"Options cheap ({_fmt_pct(gap)} discount vs realized)"
    elif label == "vol neutral":
        note = "Options roughly fairly priced on vol"
    else:
        note = "No options signal"
    max_dd = m.get("max_dd")
    if max_dd is not None and math.isfinite(max_dd):
        note += f"; max drawdown {max_dd:.0%}"
    return note


def _size(r: ScoreResult) -> str:
    m = r.metrics
    sized = m.get("kelly_sized")
    if sized is None or not math.isfinite(sized):
        return "Sizing n/a"
    mk = m.get("markowitz_weight")
    extra = f", Markowitz {mk:.1%}" if mk is not None and math.isfinite(mk) else ""
    return f"Suggested {sized:.1%} position (¼ Kelly){extra}"


def _smart_money(r: ScoreResult) -> str | None:
    sm = r.metrics.get("smart_money")
    if not sm or not sm.get("enabled"):
        return None
    parts: list[str] = []
    ins = sm.get("insider", {})
    if ins.get("bull_cluster"):
        parts.append(
            f"BUY CLUSTER: {ins['n_cluster_buyers']} insiders (score {ins['normalized']:+.2f})"
        )
    elif ins.get("bear_cluster"):
        parts.append(
            f"C-suite SELL cluster: {ins['n_cluster_csuite_sellers']} off-plan "
            f"(score {ins['normalized']:+.2f})"
        )
    else:
        parts.append(
            f"insiders neutral ({ins.get('n_buys', 0)}B/{ins.get('n_sells', 0)}S, no cluster)"
        )
    inst = sm.get("institutional", {})
    if inst.get("has_signal"):
        parts.append(
            f"13F: {inst['n_holders']} hold "
            f"(+{inst['n_new'] + inst['n_added']}/-{inst['n_trimmed'] + inst['n_exited']})"
        )
    return "; ".join(parts)


def _block(r: ScoreResult) -> list[str]:
    partial = "  *(partial signal set)*" if r.partial else ""
    header = (
        f"**{r.ticker}**  {r.composite:+.0f}  {r.verdict.value}"
        f"  —  {r.confidence.value} confidence{partial}"
    )
    lines = [
        header,
        f"  - **Cheap on:**  {_cheap_on(r)}",
        f"  - **Quality:**   {_quality(r)}",
        f"  - **Risk note:** {_risk_note(r)}",
        f"  - **Size:**      {_size(r)}",
    ]
    sm = _smart_money(r)
    if sm:
        lines.append(f"  - **Smart $:**    {sm}")
    lines.append("")
    return lines


def write_markdown(run_result: RunResult) -> str:
    """Write reports/YYYY-MM-DD_brief.md and return its path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{run_result.as_of.isoformat()}_brief.md"

    lines = [f"# FinQuant — {run_result.as_of.isoformat()} Brief", ""]
    for r in run_result.results:
        lines.extend(_block(r))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


__all__ = ["write_markdown"]
