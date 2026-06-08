"""Insider transaction signal from Form 4 data — buy-cluster focused.

Routine insider selling is uninformative (executives constantly sell vested
compensation), so this signal is deliberately asymmetric:

* **Bullish** only when ``INSIDER_MIN_CLUSTER``+ distinct insiders make
  open-market *purchases* (code P) within a rolling ``INSIDER_CLUSTER_WINDOW_DAYS``
  window. Strength is weighted by role (C-suite > officer > director > owner)
  and decayed by days since filing.
* **Bearish** only when ``INSIDER_MIN_CLUSTER``+ distinct *C-suite* insiders
  *sell* in a cluster that is NOT under a Rule 10b5-1 plan. Ordinary or
  plan-based selling scores neutral (0.0), never bearish.
* **Neutral (0.0)** otherwise — including isolated buys, routine sells, and
  10b5-1 program sales.

Final score = bull_score (>=0) + bear_score (<=0), in [-1, +1].
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from quant_agent.config import (
    INSIDER_BUY_CODES,
    INSIDER_CLUSTER_WINDOW_DAYS,
    INSIDER_CSUITE_KEYWORDS,
    INSIDER_DECAY_HALFLIFE_DAYS,
    INSIDER_MIN_CLUSTER,
    INSIDER_OFFICER_KEYWORDS,
    INSIDER_OWNER_KEYWORDS,
    INSIDER_ROLE_WEIGHTS,
    INSIDER_SATURATION,
    INSIDER_SELL_CODES,
)


def _classify_role(position: object) -> str:
    p = str(position).lower()
    if any(k in p for k in INSIDER_CSUITE_KEYWORDS):
        return "csuite"
    if any(k in p for k in INSIDER_OWNER_KEYWORDS):
        return "owner"
    if any(k in p for k in INSIDER_OFFICER_KEYWORDS):
        return "officer"
    if "director" in p:
        return "director"
    return "other"


def _role_weight(position: object) -> float:
    return INSIDER_ROLE_WEIGHTS[_classify_role(position)]


def _is_csuite(position: object) -> bool:
    return _classify_role(position) == "csuite"


def _decay(filing_date: object, as_of: date) -> float:
    """Exponential recency decay: 0.5 per half-life of days since filing."""
    try:
        fdate = pd.Timestamp(filing_date).date()
    except (TypeError, ValueError):
        return 1.0
    days = max(0, (as_of - fdate).days)
    return 0.5 ** (days / INSIDER_DECAY_HALFLIFE_DAYS)


def _best_cluster(df: pd.DataFrame, as_of: date) -> tuple[int, float]:
    """Best rolling window: (max distinct insiders, role/recency-weighted strength).

    The window with the most distinct insiders wins (ties broken by strength).
    Strength sums, over distinct insiders, each insider's best role_weight*decay.
    """
    if df.empty or "Insider" not in df.columns:
        return 0, 0.0
    work = df.assign(_d=pd.to_datetime(df["Date"])).dropna(subset=["_d"]).sort_values("_d")
    if work.empty:
        return 0, 0.0

    window = pd.Timedelta(days=INSIDER_CLUSTER_WINDOW_DAYS)
    best_n, best_strength = 0, 0.0
    for anchor in work["_d"].unique():
        win = work[(work["_d"] > anchor - window) & (work["_d"] <= anchor)]
        groups = win.groupby("Insider")
        n = groups.ngroups
        if n < best_n:
            continue
        strength = 0.0
        for _, g in groups:
            strength += max(
                _role_weight(row["Position"]) * _decay(row.get("FilingDate", row["Date"]), as_of)
                for _, row in g.iterrows()
            )
        if n > best_n or (n == best_n and strength > best_strength):
            best_n, best_strength = n, strength
    return best_n, best_strength


def insider_signal(transactions: pd.DataFrame, as_of: date | None = None) -> dict:
    """Compute the buy-cluster-focused insider signal.

    Args:
        transactions: Form 4 rows with Code, Value, Insider, Position, Date and
            (ideally) FilingDate + Is10b5_1 columns.
        as_of: reference date for recency decay (defaults to today).

    Returns:
        dict with score in [-1,+1] and supporting diagnostics.
    """
    as_of = as_of or date.today()
    empty = {
        "score": 0.0, "bull_cluster": False, "bear_cluster": False,
        "n_cluster_buyers": 0, "n_cluster_csuite_sellers": 0,
        "buy_value": 0.0, "sell_value": 0.0, "n_buys": 0, "n_sells": 0,
        "n_buyers": 0, "has_signal": False,
    }
    if transactions is None or transactions.empty or "Code" not in transactions.columns:
        return empty

    df = transactions.copy()
    df["Value"] = pd.to_numeric(df.get("Value"), errors="coerce").fillna(0.0).abs()
    if "Position" not in df.columns:
        df["Position"] = ""
    if "FilingDate" not in df.columns:
        df["FilingDate"] = df["Date"]
    is_plan = df["Is10b5_1"] if "Is10b5_1" in df.columns else pd.Series(False, index=df.index)
    code = df["Code"].astype(str).str.strip().str.upper()

    buys = df[code.isin(INSIDER_BUY_CODES)]
    sells = df[code.isin(INSIDER_SELL_CODES)]

    # --- Bullish: any 3+ distinct insiders buying in a 30d window ---------- #
    n_buyers_cluster, buy_strength = _best_cluster(buys, as_of)
    bull = n_buyers_cluster >= INSIDER_MIN_CLUSTER
    bull_score = min(1.0, buy_strength / INSIDER_SATURATION) if bull else 0.0

    # --- Bearish: 3+ distinct C-suite selling, NOT under a 10b5-1 plan ----- #
    csuite_open_sells = sells[
        ~is_plan.loc[sells.index].fillna(False)
        & sells["Position"].map(_is_csuite)
    ]
    n_csuite_sellers, sell_strength = _best_cluster(csuite_open_sells, as_of)
    bear = n_csuite_sellers >= INSIDER_MIN_CLUSTER
    bear_score = -min(1.0, sell_strength / INSIDER_SATURATION) if bear else 0.0

    return {
        "score": float(bull_score + bear_score),
        "bull_cluster": bool(bull),
        "bear_cluster": bool(bear),
        "n_cluster_buyers": int(n_buyers_cluster),
        "n_cluster_csuite_sellers": int(n_csuite_sellers),
        "buy_value": float(buys["Value"].sum()),
        "sell_value": float(sells["Value"].sum()),
        "n_buys": int(len(buys)),
        "n_sells": int(len(sells)),
        "n_buyers": int(buys["Insider"].nunique()) if "Insider" in buys.columns else int(len(buys)),
        "has_signal": bool(bull or bear),
    }


__all__ = ["insider_signal"]


if __name__ == "__main__":  # pragma: no cover
    from datetime import timedelta

    today = date.today()
    recent = pd.Timestamp(today - timedelta(days=5))
    demo = pd.DataFrame({
        "Code": ["P", "P", "P", "S"],
        "Value": [500_000, 300_000, 250_000, 100_000],
        "Insider": ["Alice", "Bob", "Carol", "Dave"],
        "Position": ["CEO", "CFO", "Director", "Director"],
        "Date": [recent, recent, recent, recent],
        "FilingDate": [recent, recent, recent, recent],
        "Is10b5_1": [False, False, False, False],
    })
    print(insider_signal(demo, today))  # 3 distinct buyers -> bullish cluster
