"""FastAPI worker for on-demand single-ticker scoring (deployed on Render).

Flow: dashboard search box -> Vercel /api/score-ticker -> POST /score here.
This kicks off a background scoring run that writes verdict/signals/securities
to Supabase; the frontend polls Supabase for the result. Authenticated with a
shared ``WORKER_SECRET`` header so the endpoint isn't publicly invocable.

On-demand runs default to ``smart_money=False`` for speed (the slow 13F fetch is
skipped); a searched ticker gets full smart-money enrichment if it's later added
to a watchlist/portfolio and picked up by the nightly run. Toggle with the
``ONDEMAND_SMART_MONEY`` env var.

POST /run-daily: full pipeline for all watchlist tickers + user holdings.
Called by the Vercel cron at 13:30 UTC every weekday. Skips market holidays
via market_calendar.is_trading_day(). Runs async; returns 202 immediately.

Start command (Render):
    uv run uvicorn quant_agent.server:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os
import re
import threading
from datetime import date

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from loguru import logger
from pydantic import BaseModel

WORKER_SECRET = os.getenv("WORKER_SECRET", "")
ONDEMAND_SMART_MONEY = os.getenv("ONDEMAND_SMART_MONEY", "false").lower() == "true"
# US tickers: 1-10 chars, letters plus optional . or - (e.g. BRK.B, RDS-A).
_TICKER_RE = re.compile(r"^[A-Z][A-Z.\-]{0,9}$")

app = FastAPI(title="FinQuant Worker")

# ── on-demand /score state ────────────────────────────────────────────────────
_inflight: set[str] = set()
_lock = threading.Lock()

# ── daily /run-daily state ────────────────────────────────────────────────────
_daily_running: bool = False
_daily_lock = threading.Lock()


class ScoreRequest(BaseModel):
    ticker: str


# ── health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/")
def root() -> dict:
    return {"service": "quant-agent-worker", "ok": True}


# ── on-demand single-ticker scoring ──────────────────────────────────────────

def _score(ticker: str) -> None:
    """Run a single-ticker scoring pass; writes results to Supabase."""
    try:
        # Imported lazily so the server boots fast and import errors surface here.
        from quant_agent.config import DEFAULT_WATCHLIST
        from quant_agent.orchestrator import run as run_pipeline

        logger.info(f"on-demand scoring: {ticker}")
        run_pipeline(
            tickers=[ticker],
            watchlist=DEFAULT_WATCHLIST,
            include_holdings=False,
            smart_money=ONDEMAND_SMART_MONEY,
        )
        logger.success(f"on-demand scoring complete: {ticker}")
    except Exception as exc:  # noqa: BLE001 - log; the poller will time out on failure
        logger.error(f"on-demand scoring failed for {ticker}: {exc}")
    finally:
        with _lock:
            _inflight.discard(ticker)


@app.post("/score")
def score(
    req: ScoreRequest,
    background: BackgroundTasks,
    x_worker_secret: str | None = Header(default=None),
) -> dict:
    """Authenticate, validate the ticker, and kick off a background score."""
    if not WORKER_SECRET or x_worker_secret != WORKER_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")

    ticker = (req.ticker or "").strip().upper()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(status_code=400, detail="invalid ticker")

    with _lock:
        already_running = ticker in _inflight
        if not already_running:
            _inflight.add(ticker)

    if already_running:
        return {"status": "in_progress", "ticker": ticker}

    background.add_task(_score, ticker)
    return {"status": "accepted", "ticker": ticker}


# ── full daily pipeline ───────────────────────────────────────────────────────

def _run_daily() -> None:
    """Background: full watchlist + holdings scoring pass.

    Called by /run-daily on weekdays at 13:30 UTC via the Vercel cron.
    Writes all verdicts, signals, and securities rows to Supabase.
    Skips silently on market holidays (checked via market_calendar).
    """
    global _daily_running
    try:
        from quant_agent.config import DEFAULT_WATCHLIST
        from quant_agent.market_calendar import is_trading_day
        from quant_agent.orchestrator import run as run_pipeline

        if not is_trading_day():
            logger.info("run-daily: not a trading day — skipping")
            return

        logger.info(
            f"run-daily: starting full pipeline — "
            f"{len(DEFAULT_WATCHLIST)} watchlist tickers + all user holdings"
        )
        run_pipeline(
            tickers=DEFAULT_WATCHLIST,
            include_holdings=True,
            smart_money=True,
        )
        logger.success("run-daily: pipeline complete")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"run-daily: pipeline failed: {exc}")
    finally:
        with _daily_lock:
            _daily_running = False


@app.post("/run-daily")
def run_daily(
    background: BackgroundTasks,
    x_worker_secret: str | None = Header(default=None),
) -> dict:
    """Trigger the full daily scoring pipeline asynchronously.

    Authenticated with the same WORKER_SECRET as /score.
    Called by the Vercel cron (via /api/run-agent) every weekday at 13:30 UTC.
    Returns immediately with 202; the pipeline runs in a background thread.
    A second call while a run is in progress returns {"status": "in_progress"}.
    Market holiday detection happens inside _run_daily so the cron fires safely
    on holidays without needing to know the calendar on the Vercel side.
    """
    if not WORKER_SECRET or x_worker_secret != WORKER_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")

    global _daily_running
    with _daily_lock:
        if _daily_running:
            return {
                "status": "in_progress",
                "note": "daily run already underway",
                "as_of": date.today().isoformat(),
            }
        _daily_running = True

    background.add_task(_run_daily)
    return {"status": "accepted", "as_of": date.today().isoformat()}
