"""Command-line interface (typer).

    quant-agent run                       # summary only, terminal output
    quant-agent run --detail brief        # also write brief.md
    quant-agent run --detail full         # also write detailed.md
    quant-agent run --tickers NVDA AAPL   # subset
    quant-agent run --watchlist sp500.txt # load tickers from a file
    quant-agent run --changed-only        # only tickers whose verdict changed
"""

from __future__ import annotations

import re
import sys
from enum import StrEnum
from pathlib import Path

import typer
from loguru import logger

from quant_agent.config import DEFAULT_WATCHLIST, LOG_LEVEL

app = typer.Typer(
    add_completion=False,
    help="Pre-market quantitative stock analysis agent.",
)


class Detail(StrEnum):
    summary = "summary"
    brief = "brief"
    full = "full"


def _configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level=LOG_LEVEL, format="{time:HH:mm:ss} | {level: <8} | {message}")


def _load_watchlist(path: Path) -> list[str]:
    """Read one ticker per line, ignoring blanks and comments."""
    lines = path.read_text().splitlines()
    return [t.strip().upper() for t in lines if t.strip() and not t.startswith("#")]


@app.callback()
def _main() -> None:
    """Pre-market quantitative stock analysis agent."""
    # Presence of a callback keeps ``run`` as an explicit subcommand even though
    # it is currently the only command.


@app.command()
def run(
    tickers: list[str] | None = typer.Option(
        None, "--tickers", "-t", help="Explicit ticker subset (space-separated)."
    ),
    watchlist: Path | None = typer.Option(
        None, "--watchlist", "-w", help="Path to a watchlist file (one ticker per line)."
    ),
    detail: Detail = typer.Option(
        Detail.summary, "--detail", "-d", help="Report tier to also write to disk."
    ),
    changed_only: bool = typer.Option(
        False, "--changed-only", help="Only show tickers whose verdict changed since yesterday."
    ),
    smart_money: bool = typer.Option(
        True, "--smart-money/--no-smart-money",
        help="Compute the SEC EDGAR Smart Money layer (insider Form 4 + 13F).",
    ),
    trading_days_only: bool = typer.Option(
        False, "--trading-days-only",
        help="Exit without running if today is not an NYSE trading day (for schedulers).",
    ),
    include_holdings: bool | None = typer.Option(
        None, "--include-holdings/--no-include-holdings",
        help="Also score every user-held ticker (default: on for full watchlist runs).",
    ),
) -> None:
    """Run the analysis pipeline and render the report(s)."""
    _configure_logging()

    if trading_days_only:
        from quant_agent.market_calendar import is_trading_day

        if not is_trading_day():
            logger.info("Not an NYSE trading day — skipping run.")
            raise typer.Exit(0)

    explicit_subset = bool(tickers)
    if tickers:
        # Accept repeated flags (--tickers A --tickers B) and split any value
        # on commas/whitespace so --tickers "NVDA AAPL" / "NVDA,AAPL" also work.
        symbols = [t.upper() for raw in tickers for t in re.split(r"[,\s]+", raw) if t]
    elif watchlist:
        symbols = _load_watchlist(watchlist)
    else:
        symbols = DEFAULT_WATCHLIST

    # The curated watchlist tags verdicts + scopes the portfolio; for an explicit
    # --tickers subset we still treat DEFAULT_WATCHLIST as the canonical watchlist.
    watchlist_canonical = DEFAULT_WATCHLIST if explicit_subset else symbols
    # Default: union in user holdings for full watchlist runs, not for subsets.
    do_holdings = (not explicit_subset) if include_holdings is None else include_holdings

    logger.info(f"Watchlist: {symbols} | detail={detail.value} | holdings={do_holdings}")

    # Local import keeps CLI startup fast and avoids importing heavy deps on --help.
    from quant_agent.orchestrator import run as run_pipeline
    from quant_agent.reports import summary as summary_report

    run_result = run_pipeline(
        tickers=symbols, detail=detail.value, changed_only=changed_only,
        smart_money=smart_money, include_holdings=do_holdings,
        watchlist=watchlist_canonical,
    )

    summary_report.render_terminal(run_result)
    summary_report.write_markdown(run_result)

    if detail in (Detail.brief, Detail.full):
        from quant_agent.reports import brief as brief_report

        brief_report.write_markdown(run_result)
    if detail == Detail.full:
        from quant_agent.reports import detailed as detailed_report

        detailed_report.write_markdown(run_result)


@app.command()
def backtest(
    tickers: list[str] | None = typer.Option(
        None, "--tickers", "-t", help="Tickers to backtest (default: watchlist)."
    ),
    horizon: int = typer.Option(90, "--horizon", help="Forward-return horizon in trading days."),
) -> None:
    """Walk-forward backtest of the GBM upside signal vs the base rate."""
    _configure_logging()
    from rich.console import Console
    from rich.table import Table

    from quant_agent.backtest.walk_forward import walk_forward_watchlist

    symbols = [t.upper() for t in tickers] if tickers else DEFAULT_WATCHLIST
    df = walk_forward_watchlist(symbols, horizon_days=horizon)
    if df.empty:
        typer.echo("No backtest results.")
        raise typer.Exit(1)

    table = Table(title=f"Walk-Forward Backtest — {horizon}d horizon", header_style="bold")
    for col in ("Ticker", "Signals", "Bullish", "Hit Rate", "Base Rate", "Edge", "Avg Fwd"):
        table.add_column(col, justify="right" if col != "Ticker" else "left")
    for _, r in df.iterrows():
        edge = r["edge"]
        color = "green" if edge > 0 else "red"
        table.add_row(
            r["ticker"], str(int(r["n_signals"])), str(int(r["n_bullish"])),
            f"{r['hit_rate']:.0%}", f"{r['base_rate']:.0%}",
            f"[{color}]{edge:+.0%}[/{color}]", f"{r['avg_forward_return']:+.1%}",
        )
    Console().print(table)


if __name__ == "__main__":
    app()
