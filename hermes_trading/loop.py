"""24/7 reliability loop.

Every cycle, per configured asset:
  1. fetch OHLCV via adapters (per-adapter retry: 3 attempts, exponential backoff)
  2. evaluate strategy.yaml (Lorentzian signal) against the price series
  3. the TopStep risk engine decides whether/how large to trade, and whether to
     force-flatten; paper positions are opened/closed and dollar outcomes logged
  4. write heartbeat (incl. live account status vs. the Combine limits)

Circuit-breaker: 5 consecutive whole-cycle failures -> halt loudly.
A SchemaError from any adapter is fatal (never trade on unknown data).
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import yaml

from . import reflect, state_dir
from .adapters import SchemaError, macro, news, onchain, price
from .risk.topstep import TopStepEngine
from .signals import lorentzian

CYCLE_SECONDS = 60
MAX_CONSEC_FAILURES = 5
ADAPTER_RETRIES = 3
STATUS_INTERVAL_SEC = 12 * 3600  # Telegram status ping cadence


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _with_retries(coro_factory, label: str):
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(1, ADAPTER_RETRIES + 1):
        try:
            return await coro_factory()
        except SchemaError:
            raise  # fatal — do not retry bad-schema data
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < ADAPTER_RETRIES:
                await asyncio.sleep(delay)
                delay *= 2
    raise RuntimeError(f"{label}: all {ADAPTER_RETRIES} attempts failed: {last_exc}")


class Worker:
    def __init__(self, assets: list[dict], goal: dict):
        self.assets = assets
        self.goal = goal
        self.sd: Path = state_dir()
        self.trades_path = self.sd / "trades.jsonl"
        self.heartbeat_path = self.sd / "heartbeat.json"
        self.point_value = {a["symbol"]: float(a.get("point_value", 1.0))
                            for a in assets}
        self.costs = {a["symbol"]: {"tick": float(a.get("tick_size", 0.0)),
                                    "comm": float(a.get("commission_per_side", 0.0))}
                      for a in assets}
        self.slippage_ticks = float(goal.get("costs", {}).get("slippage_ticks", 0))
        self.engine = TopStepEngine(goal.get("topstep", {}), assets)
        # open paper position per symbol
        self.positions: dict[str, dict] = {}
        # self-improvement + notifications (all run 24/7 on Railway, no laptop)
        self.reflection_every = int(goal.get("reflection_every", 30))
        self.tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.tg_chat = os.getenv("TELEGRAM_CHAT_ID")
        self._last_status_mono = 0.0

    # -- io ---------------------------------------------------------------
    def _strategy(self) -> dict:
        return yaml.safe_load((self.sd / "strategy.yaml").read_text())

    def _read_trades(self) -> list[dict]:
        if not self.trades_path.exists():
            return []
        return [json.loads(ln) for ln in self.trades_path.read_text().splitlines()
                if ln.strip()]

    def _log_trade(self, record: dict) -> None:
        with self.trades_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def _heartbeat(self, status: str, extra: dict | None = None) -> None:
        hb = {"status": status, "ts": _now_iso(), "positions": list(self.positions)}
        if extra:
            hb.update(extra)
        self.heartbeat_path.write_text(json.dumps(hb))

    # -- notifications ----------------------------------------------------
    def _notify(self, text: str) -> None:
        """Best-effort Telegram push (no-op if creds absent). Never raises."""
        if not (self.tg_token and self.tg_chat):
            return
        try:
            httpx.post(
                f"https://api.telegram.org/bot{self.tg_token}/sendMessage",
                json={"chat_id": self.tg_chat, "text": text}, timeout=10.0)
        except Exception as exc:  # noqa: BLE001
            print(f"[notify] telegram send failed: {exc}", flush=True)

    # -- self-improvement -------------------------------------------------
    def _reflection_checkpoint(self) -> int:
        """Trade count recorded at the last reflection (0 if none)."""
        path = self.sd / "hypotheses.jsonl"
        if not path.exists():
            return 0
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        if not lines:
            return 0
        try:
            return int(json.loads(lines[-1]).get("n_trades", 0) or 0)
        except Exception:  # noqa: BLE001
            return 0

    def _maybe_reflect(self, trades: list[dict]) -> None:
        """Every `reflection_every` closed trades, run the deterministic rule
        in-process (edits strategy.yaml on the volume) and text the change."""
        if len(trades) - self._reflection_checkpoint() < self.reflection_every:
            return
        try:
            h = reflect.run_fallback(self.sd)
            self._notify(
                f"🔧 Reflection v{h['from_version']}→v{h['to_version']}\n"
                f"{h['variable']}: {h['from']} → {h['to']}\n{h['rationale']}")
            print(f"[reflect] v{h['from_version']}->v{h['to_version']}: "
                  f"{h['variable']} -> {h['to']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[reflect] failed: {exc}", flush=True)

    def _maybe_status(self, status: dict) -> None:
        now = time.monotonic()
        if now - self._last_status_mono < STATUS_INTERVAL_SEC:
            return
        self._last_status_mono = now
        self._notify(
            "📊 Hermes TopStep $100K (paper)\n"
            f"strategy v{status.get('strategy_version')} · {status.get('passed') and '🎉 TARGET' or 'running'}\n"
            f"Equity ${status.get('equity', 0):,.0f} · Day P&L ${status.get('day_pnl', 0):,.0f}\n"
            f"Trailing floor ${status.get('trailing_floor', 0):,.0f}\n"
            f"Progress {status.get('profit_progress', 0) * 100:.1f}% of target · "
            f"{status.get('day_trades', 0)} trades today")

    # -- trading ----------------------------------------------------------
    def _unrealized(self, data: dict) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            last = data.get(sym, {}).get("last", pos["entry_px"])
            total += (last - pos["entry_px"]) * pos["direction"] * \
                pos["point_value"] * pos["contracts"]
        return total

    def _close(self, symbol: str, last: float, reason: str, strat_v) -> dict:
        pos = self.positions.pop(symbol)
        c = self.costs.get(symbol, {})
        slip = self.slippage_ticks * c.get("tick", 0.0)
        # exit fill is adverse by `slip`; round-turn commission on both sides
        exit_fill = last - pos["direction"] * slip
        gross = (exit_fill - pos["entry_px"]) * pos["direction"] * \
            pos["point_value"] * pos["contracts"]
        commission = 2.0 * c.get("comm", 0.0) * pos["contracts"]
        pnl = gross - commission
        ret = (exit_fill - pos["entry_px"]) / pos["entry_px"] * pos["direction"]
        rec = {
            "symbol": symbol,
            "direction": pos["direction"],
            "contracts": pos["contracts"],
            "point_value": pos["point_value"],
            "entry_ts": pos["entry_ts"], "exit_ts": _now_iso(),
            "entry_px": pos["entry_px"], "exit_px": round(exit_fill, 4),
            "gross_usd": round(gross, 2),
            "commission_usd": round(commission, 2),
            "pnl_usd": round(pnl, 2),
            "return_pct": ret,
            "win": pnl > 0,
            "reason": reason,
            "strategy_version": strat_v,
            "mode": "paper",
        }
        self._log_trade(rec)
        return rec

    async def _fetch_asset(self, asset: dict) -> dict:
        symbol, feed = asset["symbol"], asset["feed"]
        px = await _with_retries(lambda: price.fetch(feed=feed), f"price[{symbol}]")
        # Context adapters — fetched for schema validation + future signal use.
        await _with_retries(lambda: onchain.fetch(symbol), f"onchain[{symbol}]")
        await _with_retries(lambda: news.fetch(symbol), f"news[{symbol}]")
        await _with_retries(lambda: macro.fetch(symbol), f"macro[{symbol}]")
        return px

    async def _cycle(self, strat: dict) -> dict:
        sig_cfg = strat.get("signal", {})
        risk_cfg = strat.get("risk", {})
        strat_v = strat.get("version")

        trades = self._read_trades()
        self.engine.sync(trades)

        # 0) self-improvement: reflect if enough new trades have closed
        self._maybe_reflect(trades)

        # 1) fetch + classify every asset
        data: dict[str, dict] = {}
        for asset in self.assets:
            symbol = asset["symbol"]
            px = await self._fetch_asset(asset)
            last = px["last"]
            if last is None:
                continue
            sig = lorentzian.classify(px, sig_cfg)
            a = lorentzian.atr(np.asarray(px["highs"], float),
                               np.asarray(px["lows"], float),
                               np.asarray(px["closes"], float),
                               int(risk_cfg.get("atr_period", 14)))
            data[symbol] = {"last": last, "signal": sig["signal"],
                            "conf": sig["confidence"], "atr": float(a[-1])}

        unrl = self._unrealized(data)
        flatten, flat_reason = self.engine.must_flatten(unrl)

        # 2) manage / open per asset
        for asset in self.assets:
            symbol = asset["symbol"]
            d = data.get(symbol)
            if not d:
                continue
            last, sig = d["last"], d["signal"]
            pos = self.positions.get(symbol)

            if pos is not None:
                exit_reason = None
                if flatten:
                    exit_reason = flat_reason
                elif pos["direction"] == 1 and last <= pos["stop_px"]:
                    exit_reason = "stop"
                elif pos["direction"] == 1 and last >= pos["tp_px"]:
                    exit_reason = "take_profit"
                elif pos["direction"] == -1 and last >= pos["stop_px"]:
                    exit_reason = "stop"
                elif pos["direction"] == -1 and last <= pos["tp_px"]:
                    exit_reason = "take_profit"
                elif sig != 0 and sig != pos["direction"]:
                    exit_reason = "signal_flip"
                if exit_reason:
                    self._close(symbol, last, exit_reason, strat_v)
                continue

            # flat -> maybe enter
            if flatten or sig == 0:
                continue
            ok, _why = self.engine.can_enter(risk_cfg, unrl)
            if not ok:
                continue
            stop_dist = d["atr"] * float(risk_cfg.get("stop_atr_mult", 1.5))
            if stop_dist <= 0:
                continue
            rr = float(risk_cfg.get("reward_risk_ratio", 1.0))
            slip = self.slippage_ticks * self.costs.get(symbol, {}).get("tick", 0.0)
            entry_fill = last + sig * slip  # adverse entry fill
            stop_px = entry_fill - sig * stop_dist
            tp_px = entry_fill + sig * stop_dist * rr
            contracts = self.engine.contracts_for(
                symbol, entry_fill, stop_px, risk_cfg, trades, unrl)
            if contracts <= 0:
                continue
            self.positions[symbol] = {
                "symbol": symbol, "direction": sig, "contracts": contracts,
                "point_value": self.point_value.get(symbol, 1.0),
                "entry_px": entry_fill, "entry_ts": _now_iso(),
                "stop_px": stop_px, "tp_px": tp_px,
                "strategy_version": strat_v,
            }

        return {"strategy_version": strat_v, "signals":
                {s: data[s]["signal"] for s in data}, **self.engine.status(unrl)}

    async def run_forever(self) -> None:
        print("Booting hermes-trading worker "
              f"(assets={[a['symbol'] for a in self.assets]}, "
              f"engine=topstep, signal=lorentzian, brain=deterministic, mode=paper)",
              flush=True)
        self._notify("🚀 Hermes brain online on Railway (24/7). Paper mode, "
                     f"reflecting every {self.reflection_every} closed trades.")
        consec_failures = 0
        while True:
            cycle_start = time.monotonic()
            try:
                strat = self._strategy()
                status = await self._cycle(strat)
                consec_failures = 0
                self._heartbeat("ok", status)
                self._maybe_status(status)
            except SchemaError as exc:
                self._heartbeat("halt_schema", {"error": str(exc)})
                print(f"FATAL schema error, halting: {exc}", flush=True)
                raise
            except Exception as exc:  # noqa: BLE001
                consec_failures += 1
                self._heartbeat("error", {"error": str(exc),
                                          "consecutive": consec_failures})
                print(f"cycle error ({consec_failures}/{MAX_CONSEC_FAILURES}): {exc}",
                      flush=True)
                if consec_failures >= MAX_CONSEC_FAILURES:
                    print("Circuit breaker tripped — halting.", flush=True)
                    raise
            elapsed = time.monotonic() - cycle_start
            await asyncio.sleep(max(1.0, CYCLE_SECONDS - elapsed))
