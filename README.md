# FinQuant

A production-quality Python agent that runs before market open and produces a
scannable quantitative analysis report. For each ticker on a configurable
watchlist it emits a verdict — **Undervalued / Fairly Priced / Overvalued /
Growth Candidate** — backed by the foundational equations of quantitative
finance.

> **Design principle:** the default report is scannable in 15 seconds
> (one line per ticker). Deeper detail is opt-in via CLI flags.

## Status

✅ **Operational.** All nine model modules, the composite scoring engine, three
report tiers, the Smart Money layer (SEC EDGAR), the walk-forward backtest, and a
morning scheduler are implemented and tested. Agent installed at `~/quant-agent`.

## Quickstart

```bash
# 1. Install uv (https://docs.astral.sh/uv/) if you don't have it.
# 2. Sync dependencies into a virtual environment.
uv sync

# 3. Configure secrets.
cp .env.example .env
#   then edit .env and add your free FRED API key (fred.stlouisfed.org)

# 4. Run.
uv run quant-agent run --tickers AAPL
```

## How to use this every morning

This is the practical, plain-English guide. The agent is installed at
`~/quant-agent`. Everything runs through `uv run` (no need to "activate" anything).

### The automatic run (already set up)

A macOS scheduler (launchd job `com.quantagent.morning`) runs the agent
**automatically every weekday at 8:30 AM** — one hour before the 9:30 ET open.
Market holidays are skipped automatically. Each morning it:

1. Prints the summary table to a log, and
2. Writes all three report files to `reports/` (plus the portfolio chart).

You don't have to do anything for this to happen. To read the results, open the
files in `~/quant-agent/reports/` (newest date wins), or check the run log at
`~/quant-agent/logs/morning_<date>.log`.

> **Time zone note:** the schedule fires at **8:30 your Mac's local time**. It's
> set assuming your Mac is on Central Time. If it isn't, edit the `<Hour>` values
> in `scripts/com.quantagent.morning.plist` and reinstall (see *Managing the
> scheduler* below).

### Running it manually (any time)

Open Terminal and run:

```bash
cd ~/quant-agent
uv run quant-agent run                  # the everyday command
```

That prints the **summary table** to your terminal and writes
`reports/<date>_summary.md`. That's all most mornings need.

### Every command, explained

```bash
# THE EVERYDAY RUN — summary table for the full watchlist, printed + saved
uv run quant-agent run

# ADD PLAIN-ENGLISH DETAIL — also writes the 4-line-per-ticker brief
uv run quant-agent run --detail brief

# FULL BREAKDOWN — also writes the deep per-ticker report + portfolio chart
uv run quant-agent run --detail full

# JUST A FEW NAMES — quote multiple tickers (or repeat the flag)
uv run quant-agent run --tickers "NVDA AAPL"

# USE A FILE — one ticker per line (edit watchlist.txt to change your list)
uv run quant-agent run --watchlist watchlist.txt

# ONLY WHAT CHANGED — show only tickers whose verdict flipped since the last run
uv run quant-agent run --changed-only

# FASTER — skip the SEC insider/13F layer (saves ~1 min of EDGAR fetching)
uv run quant-agent run --no-smart-money

# SCHEDULER MODE — exit immediately if the market is closed today
uv run quant-agent run --trading-days-only

# BACKTEST — how often a bullish GBM signal preceded a positive 90-day return
uv run quant-agent backtest
```

Flags combine, e.g. `uv run quant-agent run --detail full --changed-only`.

### What the output files are

All land in `~/quant-agent/reports/`, dated `YYYY-MM-DD`:

| File | What it is | When to read it |
|---|---|---|
| `<date>_summary.md` | One color-coded line per ticker, sorted by score. Always written. | 15-second morning scan |
| `<date>_brief.md` | 4 plain-English lines per ticker (why it's cheap, quality, risk, suggested size, smart-money). Written with `--detail brief` or `full`. | When a verdict catches your eye |
| `<date>_detailed.md` | Full quant breakdown per ticker + portfolio max-Sharpe weights. Written with `--detail full`. | Deep dive before acting |
| `<date>_frontier.png` | Efficient-frontier chart for the watchlist. Written with `--detail full`. | Portfolio context |

The agent also keeps `reports/verdict_history.json` (used by `--changed-only`)
and per-day run logs in `logs/`.

### Managing the scheduler

```bash
# See it's loaded:
launchctl print gui/$(id -u)/com.quantagent.morning | grep state

# Run it right now (don't wait for 8:30):
launchctl kickstart gui/$(id -u)/com.quantagent.morning

# Turn it off:
launchctl bootout gui/$(id -u)/com.quantagent.morning

# Turn it back on / reinstall after editing the plist:
cp scripts/com.quantagent.morning.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.quantagent.morning.plist
```

## CLI reference

```bash
quant-agent run                        # summary table, terminal output
quant-agent run --detail brief         # also write reports/<date>_brief.md
quant-agent run --detail full          # also write reports/<date>_detailed.md
quant-agent run --tickers "NVDA AAPL"  # subset (quote or repeat the flag)
quant-agent run --watchlist watchlist.txt
quant-agent run --changed-only         # only tickers whose verdict changed
quant-agent run --no-smart-money       # skip the SEC EDGAR layer (faster)
quant-agent run --trading-days-only    # no-op on market holidays (for schedulers)
quant-agent backtest                   # walk-forward GBM-signal backtest
```

## Report tiers

| Tier | File | Content |
|---|---|---|
| 1 — Summary | `reports/<date>_summary.md` | One color-coded line per ticker (default terminal output) |
| 2 — Brief | `reports/<date>_brief.md` | 4-line plain-English summary per ticker |
| 3 — Detailed | `reports/<date>_detailed.md` | Full quant breakdown + portfolio frontier |

## Model library (`src/quant_agent/models/`)

| Module | Equation set |
|---|---|
| `gbm.py` | Geometric Brownian Motion + vectorized Monte Carlo |
| `black_scholes.py` | BS pricing, Greeks, implied vol, IV surface |
| `ito.py` | Itô-derived log-normal moments (GBM validator) |
| `capm.py` | CAPM beta, required return, alpha signal |
| `kelly.py` | Discrete & continuous Kelly position sizing |
| `markowitz.py` | Mean-variance optimization, efficient frontier |
| `garch.py` | GARCH(1,1) one-day-ahead vol forecast |
| `valuation.py` | DCF, Piotroski F-Score, Altman Z-Score |
| `risk.py` | Sharpe, Sortino, VaR, CVaR, max drawdown |
| `insider.py` | Form 4 insider net-buy signal (Smart Money layer) |
| `institutional.py` | 13F institutional accumulation signal (Smart Money layer) |

## Smart Money layer (SEC EDGAR — free, no key)

A composite-ready signal layer built on [EdgarTools](https://github.com/dgunning/edgartools):

- **Insider (Form 4):** *buy-cluster focused.* Bullish only when 3+ distinct insiders make open-market purchases in a 30-day window (weighted by role — C-suite > officer > director — and decayed by recency). Routine selling scores neutral; only a cluster of 3+ C-suite sellers acting **outside** a 10b5-1 plan scores bearish.
- **Institutional (13F):** quarter-over-quarter accumulation across a configurable set of notable managers (`TRACKED_FUNDS` in `config.py`) — new positions and adds are bullish, trims and exits bearish.

Both are **computed and shown in the brief/detailed reports but kept out of the
7-signal composite by default** (`insider`/`institutional` weights = 0.0 in
`config.py`). Raise either weight to fold it into the verdict; the engine
renormalizes automatically. SEC requires a contact identity — set
`EDGAR_IDENTITY` in `.env` (falls back to the project owner's contact).

## Composite scoring

Seven weighted signals (configurable in `config.py`) fuse into a composite
score in `[-100, +100]`:

| Signal | Weight |
|---|---|
| CAPM alpha | 25% |
| DCF intrinsic vs price | 25% |
| GBM upside probability | 15% |
| BS implied vs realized vol | 10% |
| Piotroski F-Score | 10% |
| Markowitz max-Sharpe weight | 10% |
| Altman Z-Score | 5% |

Confidence reflects agreement across the seven signals.

## Configuration

Every constant, weight, and threshold lives in
[`src/quant_agent/config.py`](src/quant_agent/config.py) — no magic numbers
elsewhere. Secrets come from `.env` (see `.env.example`).

## Development

```bash
uv run pytest            # tests (≥80% coverage target on models/)
uv run ruff check .      # lint
uv run mypy src          # type-check
```

## Milestone plan

**Milestone 1 (ship first):** scaffold → data fetcher + cache → `gbm.py`
(validated against Itô moments) → `black_scholes.py` → `capm.py` → partial
scoring → summary table → `uv run quant-agent run --tickers AAPL`.

**After M1:** Kelly, Markowitz, Itô validation wiring, GARCH, DCF, brief &
detailed tiers, `--changed-only`, and the walk-forward backtest.
