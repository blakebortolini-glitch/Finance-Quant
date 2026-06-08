"""Optional authenticated webhook that runs the quant agent on POST.

Only needed if you want Vercel's cron to *remotely* trigger a morning run. The
default setup does NOT need this — the launchd scheduler runs the agent locally
at 8:30 CT and writes to Supabase directly.

To use the cloud trigger instead:
  1. Set a shared secret:   export AGENT_WEBHOOK_SECRET=<same value set in Vercel>
  2. Run this listener:      uv run python scripts/agent_webhook.py
  3. Expose it publicly:     cloudflared tunnel --url http://localhost:8787
  4. In Vercel, set AGENT_WEBHOOK_URL to <tunnel-url>/  and AGENT_WEBHOOK_SECRET.
  5. Disable the launchd job (or the Vercel cron) so the agent doesn't run twice.

The listener must be running for the cloud trigger to work — hence "the local
machine must be on."
"""

from __future__ import annotations

import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

SECRET = os.getenv("AGENT_WEBHOOK_SECRET", "")
PORT = int(os.getenv("AGENT_WEBHOOK_PORT", "8787"))
PROJECT_DIR = Path(__file__).resolve().parents[1]

_lock = threading.Lock()


def _run_agent() -> None:
    """Run one full agent pass (skips market holidays via --trading-days-only)."""
    if not _lock.acquire(blocking=False):
        print("agent already running; ignoring trigger")
        return
    try:
        print("running quant-agent…")
        subprocess.run(
            ["uv", "run", "quant-agent", "run", "--detail", "full", "--trading-days-only"],
            cwd=PROJECT_DIR,
            check=False,
        )
        print("agent run complete")
    finally:
        _lock.release()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if not SECRET or self.headers.get("x-agent-secret") != SECRET:
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b"unauthorized")
            return
        threading.Thread(target=_run_agent, daemon=True).start()
        self.send_response(202)
        self.end_headers()
        self.wfile.write(b"accepted")

    def log_message(self, *args) -> None:  # quieter logs
        return


if __name__ == "__main__":
    if not SECRET:
        raise SystemExit("Set AGENT_WEBHOOK_SECRET before starting the webhook.")
    print(f"agent webhook listening on http://localhost:{PORT} (POST, x-agent-secret)")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
