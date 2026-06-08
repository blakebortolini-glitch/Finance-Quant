"""Institutional accumulation signal from 13F holdings.

Across a tracked set of notable managers, compare each fund's latest 13F to its
prior quarter for a given ticker and classify the move:

    new position  -> +1   (fund initiated)
    added (>5%)   -> +1   (fund increased)
    trimmed (>5%) -> -1   (fund reduced)
    exited        -> -1   (fund sold out)
    held flat     ->  0

The score is net accumulation normalized by the number of funds active in the
name (held in either quarter):
    score = (new + added - trimmed - exited) / n_active   in [-1, +1]
0.0 means no tracked fund holds (or moved on) the ticker — an absent signal.
"""

from __future__ import annotations

import pandas as pd

_CHANGE_THRESHOLD = 0.05  # ignore <5% share moves as noise


def _shares(holdings: pd.DataFrame | None, ticker: str) -> float | None:
    """Total shares held of ``ticker`` in a holdings frame, or None if absent."""
    if holdings is None or holdings.empty or "Ticker" not in holdings.columns:
        return None
    row = holdings[holdings["Ticker"].astype(str).str.upper() == ticker.upper()]
    if row.empty:
        return None
    return float(row["shares"].sum())


def institutional_signal(ticker: str, funds: list[dict]) -> dict:
    """Net 13F accumulation signal for ``ticker`` across tracked funds."""
    empty = {
        "score": 0.0, "n_holders": 0, "n_new": 0, "n_added": 0,
        "n_trimmed": 0, "n_exited": 0, "n_active": 0,
        "holders": [], "has_signal": False,
    }
    if not funds:
        return empty

    n_new = n_added = n_trimmed = n_exited = n_holders = n_active = 0
    holders: list[str] = []

    for fund in funds:
        now = _shares(fund.get("latest"), ticker)
        prior = _shares(fund.get("prior"), ticker)

        if now is not None:
            n_holders += 1
            holders.append(fund["name"])

        held_now = now is not None and now > 0
        held_prior = prior is not None and prior > 0
        if not held_now and not held_prior:
            continue
        n_active += 1

        if held_now and not held_prior:
            n_new += 1
        elif held_prior and not held_now:
            n_exited += 1
        elif held_now and held_prior:
            if now > prior * (1 + _CHANGE_THRESHOLD):
                n_added += 1
            elif now < prior * (1 - _CHANGE_THRESHOLD):
                n_trimmed += 1

    if n_active == 0:
        return empty

    score = (n_new + n_added - n_trimmed - n_exited) / n_active
    return {
        "score": float(max(-1.0, min(1.0, score))),
        "n_holders": n_holders,
        "n_new": n_new, "n_added": n_added,
        "n_trimmed": n_trimmed, "n_exited": n_exited,
        "n_active": n_active,
        "holders": holders,
        "has_signal": True,
    }


__all__ = ["institutional_signal"]


if __name__ == "__main__":  # pragma: no cover
    funds = [
        {"name": "Fund A",
         "latest": pd.DataFrame({"Ticker": ["AAPL"], "shares": [100], "value": [1]}),
         "prior": pd.DataFrame({"Ticker": ["AAPL"], "shares": [50], "value": [1]})},
        {"name": "Fund B",
         "latest": pd.DataFrame({"Ticker": ["MSFT"], "shares": [10], "value": [1]}),
         "prior": pd.DataFrame({"Ticker": ["AAPL"], "shares": [20], "value": [1]})},
    ]
    print(institutional_signal("AAPL", funds))  # A added, B exited -> 0/2 = 0.0
