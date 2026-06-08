#!/usr/bin/env bash
#
# Morning run wrapper for the quant-agent scheduler.
# Invoked by launchd/cron at 8:30 AM CT on weekdays; the agent itself skips
# market holidays via --trading-days-only. Prints the summary table and writes
# all three report tiers (summary + brief + detailed) plus the frontier PNG.
#
set -euo pipefail

# launchd/cron run with a minimal PATH — make uv discoverable.
export PATH="$HOME/.local/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"

PROJECT_DIR="$HOME/quant-agent"
cd "$PROJECT_DIR"

mkdir -p logs
LOG="logs/morning_$(date +%Y-%m-%d).log"

echo "===== quant-agent morning run: $(date) =====" >>"$LOG"
uv run quant-agent run --detail full --trading-days-only 2>&1 | tee -a "$LOG"
