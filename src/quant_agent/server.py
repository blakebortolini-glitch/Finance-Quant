"""FastAPI worker for on-demand single-ticker scoring (deployed on Render).

Flow: dashboard search box -> Vercel /api/score-ticker -> POST /score here.
This kicks off a background scoring run that writes verdict/signals/securities
to Supabase; the frontend polls Supabase for the result. Authenticated with a
shared ``WORKER_SECRET`` header so the endpoint isn't publicly invocable.

On-demand runs default to ``smart_money=False`` for speed (the slow 13F fetch is
skipped); a searched ticker gets full smart-money enrichment if it's later added
to a watchlist/portfolio and picked up by the nightly run. Toggle with the
``ONDEMAND_SMART_MONEY`` env var.

Start command (Render):
    uv run uvicorn quant_agent.server:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os
import re
import threading

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from loguru import logger
from pydantic import BaseModel

WORKER_SECRET = os.getenv("WORKER_SECRET", "")
ONDEMAND_SMART_MONEY = os.getenv("ONDEMAND_SMART_MONEY", "false").lower() == "true"
# US tickers: 1-10 chars, letters plus optional . or - (e.g. BRK.B, RDS-A).
_TICKER_RE = re.compile(r"^[A-Z][A-Z.\-]{0,9}$")

app = FastAPI(title="FinQuant Worker")

_inflight: set[str] = set()
_lock = threading.Lock()


class ScoreRequest(BaseModel):
    ticker: str


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/")
def root() -> dict:
    return {"service": "quant-agent-worker", "ok": True}


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
