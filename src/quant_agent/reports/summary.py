"""Tier 1 report: one-line-per-ticker summary table.

Default terminal output, rendered with ``rich`` (green = undervalued/growth,
gray = fair, red = overvalued). Also written to
``reports/YYYY-MM-DD_summary.md``. Sorted by composite score descending.
Designed to be scannable in 15 seconds.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from quant_agent.config import REPORTS_DIR
from quant_agent.orchestrator import RunResult
from quant_agent.scoring.engine import ScoreResult, Verdict

# Verdict -> (rich color, glyph, short label).
_STYLE: dict[Verdict, tuple[str, str, str]] = {
    Verdict.UNDERVALUED: ("green", "▲", "UNDERVALUED"),
    Verdict.GROWTH: ("green", "▲", "GROWTH"),
    Verdict.FAIR: ("grey70", "─", "FAIR"),
    Verdict.SLIGHT_OVER: ("red", "▼", "SLIGHTLY OVER"),
    Verdict.OVERVALUED: ("red", "▼", "OVERVALUED"),
}

_BARS = 10  # strength bar width


def _strength_bar(composite: float) -> str:
    """|composite|/100 mapped to a 10-cell block bar."""
    filled = round(abs(composite) / 100.0 * _BARS)
    filled = max(0, min(_BARS, filled))
    return "█" * filled + "░" * (_BARS - filled)


def _counts(results: list[ScoreResult]) -> tuple[int, int, int]:
    bullish = sum(1 for r in results if r.verdict in (Verdict.UNDERVALUED, Verdict.GROWTH))
    bearish = sum(1 for r in results if r.verdict in (Verdict.SLIGHT_OVER, Verdict.OVERVALUED))
    neutral = len(results) - bullish - bearish
    return bullish, neutral, bearish


def _transitions(results: list[ScoreResult]) -> list[str]:
    """Human-readable 'PREV → NOW' strings for tickers that changed verdict."""
    out = []
    for r in results:
        if r.changed and r.previous_verdict:
            out.append(f"{r.ticker}: {r.previous_verdict} → {r.verdict.value}")
    return out


def render_terminal(run_result: RunResult) -> None:
    """Print the summary table to the terminal via rich."""
    console = Console()
    title = f"FINQUANT — {run_result.as_of.isoformat()} Pre-Market"

    table = Table(title=title, title_style="bold", header_style="bold")
    table.add_column("TICKER", no_wrap=True)
    table.add_column("SIGNAL", no_wrap=True)
    table.add_column("SCORE", justify="right")
    table.add_column("STRENGTH", no_wrap=True)
    table.add_column("CONFIDENCE", no_wrap=True)

    for r in run_result.results:
        color, glyph, label = _STYLE[r.verdict]
        signal = f"[{color}]{glyph} {label}[/{color}]"
        score = f"[{color}]{r.composite:+.0f}[/{color}]"
        partial = "*" if r.partial else ""
        table.add_row(
            f"[bold]{r.ticker}[/bold]",
            signal,
            score,
            f"[{color}]{_strength_bar(r.composite)}[/{color}]",
            f"{r.confidence.value}{partial}",
        )

    console.print(table)

    if run_result.results:
        bullish, neutral, bearish = _counts(run_result.results)
        console.print(
            f"Watchlist: {len(run_result.results)} tickers │ "
            f"[green]Bullish: {bullish}[/green] │ "
            f"Neutral: {neutral} │ "
            f"[red]Bearish: {bearish}[/red]"
        )
        if any(r.partial for r in run_result.results):
            console.print(
                "[grey50]* partial signal set — some inputs unavailable "
                "(e.g. DCF/options) for this ticker[/grey50]"
            )
        transitions = _transitions(run_result.results)
        if transitions:
            console.print("[yellow]↺ Changed since last run:[/yellow] " + "  │  ".join(transitions))
    else:
        console.print("[red]No results — all tickers failed. Check logs.[/red]")


def write_markdown(run_result: RunResult) -> str:
    """Write reports/YYYY-MM-DD_summary.md and return its path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{run_result.as_of.isoformat()}_summary.md"

    lines = [
        f"# FinQuant — {run_result.as_of.isoformat()} Pre-Market",
        "",
        "| Ticker | Signal | Score | Strength | Confidence |",
        "|---|---|---:|---|---|",
    ]
    for r in run_result.results:
        _, glyph, label = _STYLE[r.verdict]
        partial = " \\*" if r.partial else ""
        lines.append(
            f"| **{r.ticker}** | {glyph} {label} | {r.composite:+.0f} | "
            f"`{_strength_bar(r.composite)}` | {r.confidence.value}{partial} |"
        )

    if run_result.results:
        bullish, neutral, bearish = _counts(run_result.results)
        lines += [
            "",
            f"**Watchlist:** {len(run_result.results)} tickers │ "
            f"Bullish: {bullish} │ Neutral: {neutral} │ Bearish: {bearish}",
        ]
        if any(r.partial for r in run_result.results):
            lines += [
                "",
                "\\* partial signal set — some inputs unavailable "
                "(e.g. DCF/options) for this ticker",
            ]
        transitions = _transitions(run_result.results)
        if transitions:
            lines += ["", "**↺ Changed since last run:** " + "; ".join(transitions)]

    path.write_text("\n".join(lines) + "\n")
    return str(path)


__all__ = ["render_terminal", "write_markdown"]
