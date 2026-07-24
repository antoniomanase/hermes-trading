"""TopStep Combine rule-enforcement engine.

Authoritative gatekeeper that sits ABOVE the signal. The signal proposes a
direction; this engine decides whether a trade is allowed at all, how many
contracts, and when to force-flatten. It can never be overridden by the signal.

Design follows the "consistency over profitability" method: size off the
drawdown BUFFER (not the account), win-rate-scaled; allow several small trades
per day (loosens the EOD trail); cap a single day's profit (consistency rule);
hard-stop the day well inside the daily loss limit; and keep a margin above the
end-of-day trailing Maximum Loss Limit — the one hard rule that kills accounts.

State is rebuilt from the trade log every cycle, so a restart never loses the
account's position relative to its limits.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone


def _date_of(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).astimezone(timezone.utc).date().isoformat()
    except Exception:  # noqa: BLE001
        return "unknown"


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class TopStepEngine:
    def __init__(self, topstep: dict, assets: list[dict]):
        self.start = float(topstep.get("account_size", 100_000))
        self.profit_target = float(topstep.get("profit_target", 6_000))
        self.trailing_mll = float(topstep.get("trailing_mll", 3_000))
        self.daily_loss_limit = float(topstep.get("daily_loss_limit", 2_000))
        self.consistency_pct = float(topstep.get("consistency_pct", 0.50))
        self.max_contracts = int(topstep.get("max_contracts", 10))
        self.soft_stop_pct = float(topstep.get("daily_soft_stop_pct", 0.60))
        self.hard_stop_pct = float(topstep.get("daily_hard_stop_pct", 0.75))
        self.profit_cap_pct = float(topstep.get("daily_profit_cap_pct", 0.40))
        self.floor_buffer = float(topstep.get("trailing_floor_buffer", 300))
        self.point_value = {a["symbol"]: float(a.get("point_value", 1.0))
                            for a in assets}

        # rebuilt each cycle
        self.realized_balance = self.start
        self.day_realized = 0.0
        self.day_trades = 0
        self.eod_peak = self.start
        self.best_day = 0.0
        self.total_profit = 0.0

    # -- state ------------------------------------------------------------
    def sync(self, trades: list[dict]) -> None:
        """Rebuild account state from closed trades. EOD trailing uses only
        *completed* days (today's P&L doesn't move the floor until it closes)."""
        today = _today()
        by_day: dict[str, float] = defaultdict(float)
        total = 0.0
        for t in trades:
            pnl = float(t.get("pnl_usd", 0.0))
            total += pnl
            by_day[_date_of(t.get("exit_ts", ""))] += pnl

        self.realized_balance = self.start + total
        self.total_profit = total
        self.day_realized = by_day.get(today, 0.0)
        self.day_trades = sum(1 for t in trades
                              if _date_of(t.get("exit_ts", "")) == today)

        # EOD peak from completed days only
        eod_bal = self.start
        peak = self.start
        for day in sorted(d for d in by_day if d not in ("unknown", today)):
            eod_bal += by_day[day]
            peak = max(peak, eod_bal)
        self.eod_peak = peak
        self.best_day = max([0.0] + [v for d, v in by_day.items()
                                     if d != "unknown"])

    # -- limits -----------------------------------------------------------
    def trailing_floor(self) -> float:
        """EOD trailing MLL floor. Rises with EOD peak, locks at breakeven."""
        return min(self.start, self.eod_peak - self.trailing_mll)

    def equity(self, unrealized: float = 0.0) -> float:
        return self.realized_balance + unrealized

    def day_pnl(self, unrealized: float = 0.0) -> float:
        return self.day_realized + unrealized

    def passed(self) -> bool:
        return self.realized_balance >= self.start + self.profit_target

    def failed(self) -> bool:
        # EOD floor breach is the true fail; this flags live proximity.
        return self.equity() <= self.trailing_floor()

    # -- gating -----------------------------------------------------------
    def can_enter(self, risk_cfg: dict, unrealized: float = 0.0) -> tuple[bool, str]:
        if self.passed():
            return False, "target_reached"
        max_trades = int(risk_cfg.get("max_trades_per_day", 5))
        if self.day_trades >= max_trades:
            return False, "max_trades_per_day"
        if self.day_pnl(unrealized) <= -self.soft_stop_pct * self.daily_loss_limit:
            return False, "daily_soft_stop"
        if self.day_realized >= self.profit_cap_pct * self.profit_target:
            return False, "consistency_profit_cap"
        if self.equity(unrealized) - self.trailing_floor() <= self.floor_buffer:
            return False, "near_trailing_floor"
        return True, "ok"

    def must_flatten(self, unrealized: float = 0.0) -> tuple[bool, str]:
        if self.day_pnl(unrealized) <= -self.hard_stop_pct * self.daily_loss_limit:
            return True, "daily_hard_stop"
        if self.equity(unrealized) - self.trailing_floor() <= self.floor_buffer / 2:
            return True, "trailing_floor_guard"
        return False, ""

    # -- sizing -----------------------------------------------------------
    def win_rate(self, trades: list[dict], window: int) -> float | None:
        closed = [t for t in trades if "win" in t][-window:]
        if len(closed) < max(5, window // 2):
            return None  # not enough sample — treat as cold start
        return sum(1 for t in closed if t["win"]) / len(closed)

    def risk_dollars(self, risk_cfg: dict, trades: list[dict]) -> float:
        """Risk-per-trade in $, as a win-rate-scaled fraction of the buffer."""
        lo = float(risk_cfg.get("risk_pct_of_buffer_min", 0.08))
        hi = float(risk_cfg.get("risk_pct_of_buffer_max", 0.20))
        target = float(risk_cfg.get("win_rate_target", 0.65))
        window = int(risk_cfg.get("win_rate_window", 20))
        wr = self.win_rate(trades, window)
        if wr is None:
            pct = lo  # cold start: most conservative
        else:
            # linear ramp from lo (wr<=0.50) to hi (wr>=target)
            span = max(1e-6, target - 0.50)
            frac = max(0.0, min(1.0, (wr - 0.50) / span))
            pct = lo + frac * (hi - lo)
        return pct * self.trailing_mll

    def contracts_for(self, symbol: str, entry_px: float, stop_px: float,
                      risk_cfg: dict, trades: list[dict],
                      unrealized: float = 0.0) -> int:
        pv = self.point_value.get(symbol, 1.0)
        risk_per_contract = abs(entry_px - stop_px) * pv
        if risk_per_contract <= 0:
            return 0
        budget = self.risk_dollars(risk_cfg, trades)

        # never let a stop-out breach the daily hard stop
        room_today = (self.hard_stop_pct * self.daily_loss_limit
                      + self.day_pnl(unrealized))
        budget = min(budget, max(0.0, room_today))
        # never let a stop-out breach the trailing floor
        room_floor = self.equity(unrealized) - self.trailing_floor() - self.floor_buffer
        budget = min(budget, max(0.0, room_floor))

        contracts = int(budget // risk_per_contract)
        contracts = min(contracts, self.max_contracts, self._scaling_cap())
        return max(0, contracts)

    def _scaling_cap(self) -> int:
        """Simple scaling plan: trade a fraction of max size until a buffer is
        built, ramping to full size as profit accrues toward target."""
        progress = max(0.0, min(1.0, self.total_profit / self.profit_target))
        cap = round(self.max_contracts * (0.4 + 0.6 * progress))
        return max(1, min(self.max_contracts, cap))

    # -- reporting --------------------------------------------------------
    def status(self, unrealized: float = 0.0) -> dict:
        return {
            "balance": round(self.realized_balance, 2),
            "equity": round(self.equity(unrealized), 2),
            "trailing_floor": round(self.trailing_floor(), 2),
            "day_pnl": round(self.day_pnl(unrealized), 2),
            "day_trades": self.day_trades,
            "eod_peak": round(self.eod_peak, 2),
            "best_day": round(self.best_day, 2),
            "profit_progress": round(self.total_profit / self.profit_target, 3),
            "passed": self.passed(),
        }
