"""score(trades, goal) -> float in [-1, +1], TopStep-Combine aware.

Optimizes for what actually passes a Combine — consistency and staying inside
the limits — NOT raw profit or win rate alone. Composite of:
  - progress toward the profit target
  - dollar drawdown vs the trailing Max Loss Limit  (heavy weight)
  - win rate vs target
  - consistency: low variance + best-day under the consistency cap  (heavy weight)
A hard rule breach (day <= -daily_loss_limit, or drawdown >= trailing_mll)
returns -1 outright.

Works on dollar P&L (`pnl_usd`); falls back to `return_pct` for legacy trades.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterable


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _pnls(trades: list[dict]) -> list[float]:
    return [float(t.get("pnl_usd", 0.0)) for t in trades]


def _date_of(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).astimezone(timezone.utc).date().isoformat()
    except Exception:  # noqa: BLE001
        return "unknown"


# --- legacy %-return helpers (kept for back-compat) ------------------------
def _returns(trades: list[dict]) -> list[float]:
    return [float(t.get("return_pct", 0.0)) for t in trades if "return_pct" in t]


def realised_return(trades: list[dict]) -> float:
    eq = 1.0
    for r in _returns(trades):
        eq *= (1.0 + r)
    return eq - 1.0


def max_drawdown(trades: list[dict]) -> float:
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in _returns(trades):
        eq *= (1.0 + r)
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak if peak > 0 else 0.0)
    return mdd


# --- dollar helpers --------------------------------------------------------
def total_pnl(trades: list[dict]) -> float:
    return sum(_pnls(trades))


def pnl_drawdown(trades: list[dict]) -> float:
    """Peak-to-trough drawdown of the cumulative $ equity curve (positive)."""
    eq = peak = 0.0
    mdd = 0.0
    for p in _pnls(trades):
        eq += p
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return mdd


def win_rate(trades: list[dict]) -> float | None:
    closed = [t for t in trades if "win" in t]
    return (sum(1 for t in closed if t["win"]) / len(closed)) if closed else None


def daily_pnls(trades: list[dict]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for t in trades:
        out[_date_of(t.get("exit_ts", ""))] += float(t.get("pnl_usd", 0.0))
    return dict(out)


def score(trades: Iterable[dict], goal: dict) -> float:
    trades = list(trades)
    if not trades:
        return 0.0

    ts = goal.get("topstep", {})
    target = float(ts.get("profit_target", 6_000))
    mll = float(ts.get("trailing_mll", 3_000))
    daily_limit = float(ts.get("daily_loss_limit", 2_000))
    consistency_cap = float(ts.get("consistency_pct", 0.50)) * target
    wr_target = float(goal.get("objective", {}).get("win_rate_target", 0.65))

    pnls = _pnls(trades)
    days = daily_pnls(trades)

    # Hard rule breaches -> outright fail.
    if pnl_drawdown(trades) >= mll:
        return -1.0
    if days and min(days.values()) <= -daily_limit:
        return -1.0

    # 1) progress to target
    progress_s = _clamp(total_pnl(trades) / target if target > 0 else 0.0)

    # 2) drawdown vs trailing MLL (heavy)
    dd_s = _clamp(1.0 - pnl_drawdown(trades) / mll if mll > 0 else -1.0)

    # 3) win rate vs target
    wr = win_rate(trades)
    if wr is None:
        wr_s = 0.0
    else:
        span = max(1e-6, wr_target - 0.50)
        wr_s = _clamp((wr - 0.50) / span)

    # 4) consistency: low variance + best day under the cap (heavy)
    if len(pnls) >= 2:
        mean = sum(pnls) / len(pnls)
        sd = math.sqrt(sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1))
        scale = sum(abs(p) for p in pnls) / len(pnls) + 1e-9
        smooth_s = _clamp(1.0 - sd / scale)
    else:
        smooth_s = 0.0
    best_day = max([0.0] + list(days.values()))
    bestday_s = _clamp(1.0 - best_day / consistency_cap) if consistency_cap > 0 else 0.0
    consistency_s = (smooth_s + bestday_s) / 2.0

    return _clamp(0.20 * progress_s + 0.30 * dd_s +
                  0.20 * wr_s + 0.30 * consistency_s)
