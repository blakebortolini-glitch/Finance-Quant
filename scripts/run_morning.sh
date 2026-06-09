#!/usr/bin/env bash
#
# Morning run wrapper for the quant-agent scheduler.
# Invoked by launchd at 8:30 AM CT on weekdays; the agent itself skips
# market holidays via --trading-days-only.
#
# Behavior:
#   - Per-attempt timeout: 20 min (bash watchdog, since macOS has no `timeout`)
#   - On failure or timeout: waits 5 min and retries once
#   - Exits 0 on any success, 1 if both attempts fail
#
set -uo pipefail

# Expand PATH so uv is always found regardless of the invoking environment.
# uv installs to ~/.local/bin by default; ~/.cargo/bin is the fallback location.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:$PATH"

# Abort immediately with a clear message if uv is still not found.
if ! command -v uv &>/dev/null; then
  echo "ERROR: uv not found in PATH=$PATH" >&2
  exit 1
fi

PROJECT_DIR="$HOME/quant-agent"
cd "$PROJECT_DIR"
mkdir -p logs

LOG="logs/morning_$(date +%Y-%m-%d).log"
TIMEOUT_SECS=1200  # 20 minutes per attempt

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------------------
# run_agent: runs the agent with a 20-minute watchdog timeout.
#   stdout+stderr -> LOG (also echoed to launchd's StandardOutPath via tee)
#   Returns the agent's exit code, or 124 if killed by the watchdog.
# ---------------------------------------------------------------------------
run_agent() {
  uv run quant-agent run --detail full --trading-days-only 2>&1 | tee -a "$LOG" &
  local pipe_pid=$!

  # Watchdog: kill the pipeline if it outlives the timeout.
  (
    sleep "$TIMEOUT_SECS"
    if kill -0 "$pipe_pid" 2>/dev/null; then
      log "[watchdog] 20-min timeout reached — killing agent (PID $pipe_pid)"
      kill -TERM "$pipe_pid" 2>/dev/null || true
      sleep 5
      kill -KILL "$pipe_pid" 2>/dev/null || true
    fi
  ) &
  local watchdog_pid=$!

  wait "$pipe_pid" 2>/dev/null
  local ec=$?

  # Cancel the watchdog now that the pipeline has exited.
  kill "$watchdog_pid" 2>/dev/null || true
  wait "$watchdog_pid" 2>/dev/null || true

  return $ec
}

# ---------------------------------------------------------------------------
# Attempt 1
# ---------------------------------------------------------------------------
log "===== quant-agent morning run ====="

if run_agent; then
  log "Run 1 succeeded."
  exit 0
fi

log "Run 1 failed (exit $?) — retrying in 5 min…"
sleep 300

# ---------------------------------------------------------------------------
# Attempt 2 (retry)
# ---------------------------------------------------------------------------
log "===== quant-agent retry run ====="

if run_agent; then
  log "Retry succeeded."
  exit 0
fi

log "Retry also failed — giving up."
exit 1
