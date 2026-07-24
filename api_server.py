import json
import os
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _read_trades(trades_path: Path) -> list[dict]:
    if not trades_path.exists():
        return []
    try:
        rows = [json.loads(ln) for ln in trades_path.read_text().splitlines() if ln.strip()]
        return rows
    except Exception:
        return []

def _read_heartbeat(heartbeat_path: Path) -> dict:
    if not heartbeat_path.exists():
        return {}
    try:
        return json.loads(heartbeat_path.read_text())
    except Exception:
        return {}

def _win_rate(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("win", False))
    return wins / len(trades) if trades else 0.0

def _total_pnl(trades: list[dict]) -> float:
    return sum(t.get("pnl_usd", 0) for t in trades)

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})

@app.get("/api/data")
async def get_data():
    try:
        from . import state_dir
        sd = state_dir()
        trades = _read_trades(sd / "trades.jsonl")
        hb = _read_heartbeat(sd / "heartbeat.json")
        
        cum_pnl = 0.0
        trades_with_cum = []
        for i, t in enumerate(trades):
            cum_pnl += t.get("pnl_usd", 0)
            trades_with_cum.append({
                "trade": i + 1,
                "symbol": t.get("symbol"),
                "direction": t.get("direction"),
                "entry_ts": t.get("entry_ts"),
                "exit_ts": t.get("exit_ts"),
                "entry_px": round(t.get("entry_px", 0), 4),
                "exit_px": round(t.get("exit_px", 0), 4),
                "pnl": round(t.get("pnl_usd", 0), 2),
                "cumPnL": round(cum_pnl, 2),
                "win": t.get("win", False),
                "reason": t.get("reason"),
            })
        
        wr = _win_rate(trades)
        pnl = _total_pnl(trades)
        
        return JSONResponse({
            "timestamp": _now_iso(),
            "balance": hb.get("balance", 100000),
            "equity": hb.get("equity", 100000),
            "pnl": round(pnl, 2),
            "winRate": round(wr, 4),
            "trailingFloor": hb.get("trailing_floor", 100000),
            "dayPnL": hb.get("day_pnl", 0),
            "dayTrades": hb.get("day_trades", 0),
            "strategyVersion": hb.get("strategy_version", "?"),
            "status": hb.get("status", "unknown"),
            "tradeCount": len(trades),
            "trades": trades_with_cum,
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
