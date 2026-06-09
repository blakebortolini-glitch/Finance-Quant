"""Central configuration: every constant, weight, and threshold lives here.

No magic numbers anywhere else in the codebase. Import from this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CACHE_DIR: Path = Path(os.getenv("QUANT_AGENT_CACHE_DIR", PROJECT_ROOT / "data" / "cache"))
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# --------------------------------------------------------------------------- #
# Secrets / environment
# --------------------------------------------------------------------------- #
FRED_API_KEY: str | None = os.getenv("FRED_API_KEY")
LOG_LEVEL: str = os.getenv("QUANT_AGENT_LOG_LEVEL", "INFO")

# SEC EDGAR requires a contact identity ("Name email") on every request.
# Override in .env via EDGAR_IDENTITY; falls back to the project owner's contact.
EDGAR_IDENTITY: str = os.getenv("EDGAR_IDENTITY", "FinQuant blakebortolini@gmail.com")

# Supabase sink (optional). The project URL is public (also baked into the
# frontend), so it has a sensible default; the SERVICE ROLE key is a secret and
# must come from .env. Writes are skipped gracefully when the key is absent.
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "https://fsvnarocfingywjmoevs.supabase.co")
SUPABASE_SERVICE_KEY: str | None = os.getenv("SUPABASE_SERVICE_KEY")

# --------------------------------------------------------------------------- #
# Data fetching
# --------------------------------------------------------------------------- #
PRICE_HISTORY_YEARS: int = 3
OPTIONS_N_EXPIRATIONS: int = 3          # nearest N monthly expirations
MARKET_PROXY: str = "SPY"               # CAPM market regression proxy
CACHE_TTL_TRADING_DAYS: int = 1         # do not refetch within a trading day
FETCH_MAX_RETRIES: int = 3
FETCH_RETRY_BACKOFF_SEC: float = 2.0

# FRED series IDs
FRED_RISK_FREE_SERIES: str = "DGS3MO"   # 3-month T-bill yield
FRED_CPI_SERIES: str = "CPIAUCSL"       # CPI, all urban consumers
FRED_SOFR_SERIES: str = "SOFR"

# Sectors (yfinance .info["sector"]) where the Altman Z-Score is unreliable —
# it was calibrated for manufacturers. Banks/insurance live in "Financial
# Services"; REITs in "Real Estate". For these the Altman signal is skipped.
FINANCIAL_SECTORS: frozenset[str] = frozenset({"Financial Services", "Real Estate"})

# Fallback macro values if FRED is unavailable (annualized, decimal).
FALLBACK_RISK_FREE_RATE: float = 0.0521

# --------------------------------------------------------------------------- #
# Default watchlist (override at runtime with --tickers or a watchlist file)
# --------------------------------------------------------------------------- #
DEFAULT_WATCHLIST: list[str] = [
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "AMD",
]

# --------------------------------------------------------------------------- #
# Smart Money layer (SEC EDGAR via edgartools — free, no key)
# --------------------------------------------------------------------------- #
SMART_MONEY_ENABLED: bool = True            # toggle the whole layer (CLI can override)
INSIDER_LOOKBACK_DAYS: int = 180            # Form 4 trailing window
EDGAR_CACHE_TTL_DAYS: int = 7               # 13F is quarterly; cache longer than prices

# Form 4 transaction codes (SEC). Only open-market trades carry a clear signal;
# grants/gifts/option-exercises/tax-withholding are non-discretionary noise.
INSIDER_BUY_CODES: frozenset[str] = frozenset({"P"})   # open-market purchase
INSIDER_SELL_CODES: frozenset[str] = frozenset({"S"})  # open-market sale

# Buy-cluster-focused insider signal (see models/insider.py):
#   bullish  only when >= MIN_CLUSTER distinct insiders BUY within the window
#   bearish  only when >= MIN_CLUSTER distinct C-suite insiders SELL in a
#            cluster *outside* a 10b5-1 plan; routine selling scores neutral.
INSIDER_CLUSTER_WINDOW_DAYS: int = 30
INSIDER_MIN_CLUSTER: int = 3                 # distinct insiders to qualify
INSIDER_DECAY_HALFLIFE_DAYS: int = 30        # recency decay on each trade
INSIDER_SATURATION: float = 3.0              # weighted sum mapping to |score|=1

# Role weights: C-suite conviction counts more than directors / large holders.
INSIDER_ROLE_WEIGHTS: dict[str, float] = {
    "csuite": 1.0,      # CEO / CFO / COO / President / "Chief ..."
    "officer": 0.7,     # other named officers (VP, Treasurer, Counsel, ...)
    "director": 0.5,
    "owner": 0.4,       # 10% beneficial owner
    "other": 0.5,
}
INSIDER_CSUITE_KEYWORDS: tuple[str, ...] = ("chief", "ceo", "cfo", "coo", "president")
INSIDER_OFFICER_KEYWORDS: tuple[str, ...] = (
    "officer", "vp", "vice president", "treasurer", "secretary", "counsel",
)
INSIDER_OWNER_KEYWORDS: tuple[str, ...] = ("10%", "ten percent", "beneficial owner")

# Notable institutional managers tracked for the 13F signal (name -> SEC CIK).
# Edit freely; these are well-known long-horizon / high-conviction filers.
TRACKED_FUNDS: dict[str, int] = {
    "Berkshire Hathaway": 1067983,
    "Bridgewater Associates": 1350694,
    "Renaissance Technologies": 1037389,
    "Tiger Global Management": 1167483,
    "Appaloosa (Tepper)": 1656456,
    "Baupost Group (Klarman)": 1061768,
    "Pershing Square (Ackman)": 1336528,
    "Scion Asset Mgmt (Burry)": 1649339,
}


# --------------------------------------------------------------------------- #
# Trading-calendar constants
# --------------------------------------------------------------------------- #
TRADING_DAYS_PER_YEAR: int = 252
WEEKS_PER_YEAR: int = 52

# --------------------------------------------------------------------------- #
# GBM / Monte Carlo
# --------------------------------------------------------------------------- #
GBM_DRIFT_WINDOW: int = 252             # lookback for drift/vol estimation
GBM_N_SIMS: int = 10_000
GBM_HORIZON_DAYS: int = 90              # default forward horizon
GBM_RANDOM_SEED: int = 42               # deterministic tests
GBM_UPSIDE_THRESHOLD: float = 1.05      # P(S_T > S_0 * 1.05) for scoring

# --------------------------------------------------------------------------- #
# Black-Scholes / implied vol
# --------------------------------------------------------------------------- #
IV_MAX_ITER: int = 100
IV_TOLERANCE: float = 1e-6
IV_INITIAL_GUESS: float = 0.30
VOL_PREMIUM_THRESHOLD_PP: float = 0.05  # |IV - realized sigma| > 5pp -> flag

# --------------------------------------------------------------------------- #
# CAPM
# --------------------------------------------------------------------------- #
CAPM_BETA_LOOKBACK_YEARS: int = 2
CAPM_RETURN_FREQUENCY: str = "W"        # weekly returns for beta regression
MARKET_RISK_PREMIUM: float = 0.055      # E(R_m) - R_f, long-run equity premium

# "Actual expected return" blend weights (must sum to 1.0).
ER_BLEND_ANALYST_TARGET: float = 0.40   # analyst consensus 1y price target
ER_BLEND_GBM_DRIFT: float = 0.35        # GBM-estimated drift
ER_BLEND_EARNINGS_YIELD: float = 0.25   # earnings-based forward yield

# --------------------------------------------------------------------------- #
# Kelly position sizing
# --------------------------------------------------------------------------- #
KELLY_FRACTION: float = 0.25            # quarter-Kelly default
KELLY_MAX_POSITION: float = 0.25        # hard cap per position (sanity guardrail)

# --------------------------------------------------------------------------- #
# Markowitz mean-variance optimization
# --------------------------------------------------------------------------- #
MARKOWITZ_N_FRONTIER_POINTS: int = 50
MARKOWITZ_ALLOW_SHORT: bool = False     # short-selling toggle

# --------------------------------------------------------------------------- #
# GARCH(1,1)
# --------------------------------------------------------------------------- #
GARCH_P: int = 1
GARCH_Q: int = 1
GARCH_FORECAST_HORIZON: int = 1         # one-day-ahead

# --------------------------------------------------------------------------- #
# DCF valuation
# --------------------------------------------------------------------------- #
DCF_FORECAST_YEARS: int = 5
DCF_TERMINAL_GROWTH: float = 0.03
DCF_SIZE_PREMIUM: float = 0.0           # WACC = rf + beta*MRP + size_premium

# FCF growth is per-ticker, derived from the trailing 3y FCF CAGR and clamped.
DCF_FCF_GROWTH_FLOOR: float = 0.02
DCF_FCF_GROWTH_CAP: float = 0.20
DCF_DEFAULT_FCF_GROWTH: float = 0.08    # fallback when CAGR cannot be computed

# --------------------------------------------------------------------------- #
# Risk metrics
# --------------------------------------------------------------------------- #
VAR_CONFIDENCE_LEVELS: tuple[float, ...] = (0.95, 0.99)

# --------------------------------------------------------------------------- #
# Composite scoring engine
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScoringWeights:
    """Signal weights for the composite verdict — all nine must sum to 1.0.

    The Smart Money signals are now part of the composite at small weights
    (insider 0.06, institutional 0.03). The core seven were rebalanced down to
    0.91 in aggregate (preserving their relative ordering) to make room:
        0.28/0.17/0.11/0.15/0.11/0.06/0.12  (sum 1.00, old)
     -> 0.25/0.15/0.10/0.14/0.10/0.05/0.12  (sum 0.91, new)
    The insider signal is buy-cluster-gated, so for most tickers it is absent
    and its weight renormalizes away — it only bites on genuine clusters.
    """

    # --- Core 7 (sum to 0.91) -------------------------------------------- #
    capm_alpha: float = 0.25
    gbm_upside: float = 0.15
    vol_premium: float = 0.10
    dcf_intrinsic: float = 0.14
    piotroski: float = 0.10
    altman_z: float = 0.05
    markowitz_weight: float = 0.12

    # --- Smart Money (in the composite at small weights) ----------------- #
    insider: float = 0.06
    institutional: float = 0.03

    def validate(self) -> None:
        total = (
            self.capm_alpha + self.gbm_upside + self.vol_premium + self.dcf_intrinsic
            + self.piotroski + self.altman_z + self.markowitz_weight
            + self.insider + self.institutional
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total}")


SCORING_WEIGHTS = ScoringWeights()

# Composite score (-100..+100) -> verdict thresholds.
VERDICT_UNDERVALUED: float = 30.0
VERDICT_SLIGHT_UNDER: float = 10.0
VERDICT_SLIGHT_OVER: float = -10.0
VERDICT_OVERVALUED: float = -30.0

# Fundamental signal thresholds.
PIOTROSKI_STRONG: int = 7
ALTMAN_Z_SAFE: float = 2.99

# Confidence: agreement count across the 7 signals.
CONFIDENCE_HIGH_MIN_AGREE: int = 6
CONFIDENCE_MEDIUM_MIN_AGREE: int = 4


@dataclass(frozen=True)
class Settings:
    """Aggregated settings object passed through the pipeline."""

    weights: ScoringWeights = field(default_factory=lambda: SCORING_WEIGHTS)
    risk_free_fallback: float = FALLBACK_RISK_FREE_RATE
    kelly_fraction: float = KELLY_FRACTION

    def __post_init__(self) -> None:
        self.weights.validate()


SETTINGS = Settings()
