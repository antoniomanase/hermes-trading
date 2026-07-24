"""Reflection cycle. Two modes:

  --fallback  deterministic rule (used before Hermes takes over). Changes
              exactly ONE variable, bumps version, archives prior, logs hypothesis.
  --hermes    production: builds a prompt from the last 25 trades + current
              strategy, calls `hermes` as a subprocess, parses/applies the
              proposed one-variable change.

Both paths enforce two guardrails:
  1. one_variable_only  — exactly one leaf may change.
  2. whitelist          — the changed leaf must live under signal.* or risk.*
                          (never a TopStep hard limit), and risk.reward_risk_ratio
                          may never exceed goal.objective.reward_risk_max.
Every prior version is preserved in state/history/.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import state_dir
from .score import pnl_drawdown, total_pnl, win_rate, daily_pnls, score


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(sd: Path, name: str) -> dict:
    return yaml.safe_load((sd / name).read_text())


def _read_trades(sd: Path, limit: int | None = None) -> list[dict]:
    path = sd / "trades.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    return rows[-limit:] if limit else rows


def _leaves(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if k == "version":
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_leaves(v, key + "."))
        else:
            out[key] = v
    return out


def _changed_paths(old: dict, new: dict) -> list[str]:
    lo, ln = _leaves(old), _leaves(new)
    keys = set(lo) | set(ln)
    return [k for k in keys if lo.get(k) != ln.get(k)]


def _assert_one_change(old: dict, new: dict) -> str:
    changed = _changed_paths(old, new)
    if len(changed) != 1:
        raise ValueError(f"one_variable_only violated: changed {changed}")
    return changed[0]


def _assert_whitelisted(path: str, new: dict, goal: dict) -> None:
    if not (path.startswith("signal.") or path.startswith("risk.")):
        raise ValueError(f"whitelist violated: {path} is not tunable "
                         "(TopStep hard limits live in goal.yaml)")
    if path == "risk.reward_risk_ratio":
        cap = float(goal.get("objective", {}).get("reward_risk_max", 1.5))
        val = float(new["risk"]["reward_risk_ratio"])
        if val > cap:
            raise ValueError(f"reward_risk_ratio {val} exceeds cap {cap}")


def _archive_and_bump(sd: Path, old: dict, new: dict, hypothesis: dict) -> str:
    old_v = str(old.get("version", "01"))
    (sd / "history").mkdir(exist_ok=True)
    (sd / "history" / f"v{old_v.zfill(4)}.yaml").write_text(
        yaml.safe_dump(old, sort_keys=False))
    new_v = f"{int(old_v) + 1:02d}"
    new["version"] = new_v
    (sd / "strategy.yaml").write_text(yaml.safe_dump(new, sort_keys=False))
    hypothesis = {**hypothesis, "ts": _now_iso(),
                  "from_version": old_v, "to_version": new_v}
    with (sd / "hypotheses.jsonl").open("a") as fh:
        fh.write(json.dumps(hypothesis) + "\n")
    return new_v


def run_fallback(sd: Path) -> dict:
    goal = _load(sd, "goal.yaml")
    strat = _load(sd, "strategy.yaml")
    trades = _read_trades(sd)
    new = copy.deepcopy(strat)
    risk = new["risk"]

    ts = goal.get("topstep", {})
    mll = float(ts.get("trailing_mll", 3_000))
    cap = float(ts.get("consistency_pct", 0.5)) * float(ts.get("profit_target", 6_000))
    wr_target = float(goal.get("objective", {}).get("win_rate_target", 0.65))
    rr_max = float(goal.get("objective", {}).get("reward_risk_max", 1.5))

    wr = win_rate(trades)
    dd = pnl_drawdown(trades)
    best_day = max([0.0] + list(daily_pnls(trades).values()))

    # Deterministic, consistency-first lever selection (one variable only).
    if wr is not None and wr < wr_target:
        old = float(risk["reward_risk_ratio"])
        risk["reward_risk_ratio"] = round(max(0.5, old - 0.1), 4)
        variable, before, after = "risk.reward_risk_ratio", old, risk["reward_risk_ratio"]
        why = f"win_rate {wr:.2f} < target {wr_target:.2f} -> lower R:R to raise win rate"
    elif dd > 0.5 * mll:
        old = float(risk["risk_pct_of_buffer_max"])
        floor = float(risk["risk_pct_of_buffer_min"])
        risk["risk_pct_of_buffer_max"] = round(max(floor, old - 0.02), 4)
        variable, before, after = "risk.risk_pct_of_buffer_max", old, risk["risk_pct_of_buffer_max"]
        why = f"drawdown ${dd:.0f} > 50% of MLL -> cut max risk size (lower variance)"
    elif best_day > cap:
        old = float(risk["reward_risk_ratio"])
        risk["reward_risk_ratio"] = round(max(0.5, old - 0.1), 4)
        variable, before, after = "risk.reward_risk_ratio", old, risk["reward_risk_ratio"]
        why = f"best day ${best_day:.0f} > consistency cap ${cap:.0f} -> smaller, more spread wins"
    else:
        old = float(risk["risk_pct_of_buffer_max"])
        risk["risk_pct_of_buffer_max"] = round(min(0.30, old + 0.01), 4)
        variable, before, after = "risk.risk_pct_of_buffer_max", old, risk["risk_pct_of_buffer_max"]
        why = "within limits -> probe slightly larger size to pass faster"

    path = _assert_one_change(strat, new)
    _assert_whitelisted(path, new, goal)
    hypothesis = {
        "mode": "fallback", "variable": variable, "from": before, "to": after,
        "rationale": why, "score_before": score(trades, goal),
        "n_trades": len(trades), "win_rate": wr, "drawdown_usd": round(dd, 2),
    }
    new_v = _archive_and_bump(sd, strat, new, hypothesis)
    hypothesis = {**hypothesis, "from_version": str(strat.get("version")),
                  "to_version": new_v}
    print(f"[fallback] v{hypothesis['from_version']} -> v{new_v}: {variable} "
          f"{before} -> {after} ({why})")
    return hypothesis


def run_hermes(sd: Path) -> dict:
    goal = _load(sd, "goal.yaml")
    strat = _load(sd, "strategy.yaml")
    trades = _read_trades(sd, limit=25)

    prompt = (
        "You refine a paper-trading strategy whose ONLY goal is to pass a TopStep "
        "Combine: consistency and staying inside the limits, not win rate or raw "
        "profit. Change EXACTLY ONE leaf, and ONLY under signal.* or risk.*.\n"
        f"GOAL (immutable):\n{yaml.safe_dump(goal, sort_keys=False)}\n"
        f"CURRENT STRATEGY:\n{yaml.safe_dump(strat, sort_keys=False)}\n"
        f"LAST {len(trades)} TRADES:\n{json.dumps(trades, indent=2)}\n"
        "Respond with ONLY a JSON object: "
        '{\"variable\": \"<dotted.path under signal/risk>\", \"to\": <value>, '
        '\"rationale\": \"...\"}'
    )
    proc = subprocess.run(["hermes", "-z", prompt],
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"hermes call failed: {proc.stderr.strip()}")

    out = proc.stdout.strip()
    start, end = out.find("{"), out.rfind("}")
    if start < 0 or end < 0:
        raise RuntimeError(f"no JSON in hermes output: {out[:200]}")
    proposal = json.loads(out[start:end + 1])

    new = copy.deepcopy(strat)
    keys = proposal["variable"].split(".")
    node = new
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = proposal["to"]

    path = _assert_one_change(strat, new)
    _assert_whitelisted(path, new, goal)
    hypothesis = {
        "mode": "hermes", "variable": proposal["variable"], "to": proposal["to"],
        "rationale": proposal.get("rationale", ""),
        "score_before": score(trades, goal), "n_trades": len(trades),
    }
    new_v = _archive_and_bump(sd, strat, new, hypothesis)
    hypothesis = {**hypothesis, "from_version": str(strat.get("version")),
                  "to_version": new_v}
    print(f"[hermes] v{hypothesis['from_version']} -> v{new_v}: "
          f"{proposal['variable']} -> {proposal['to']}")
    return hypothesis


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fallback", action="store_true", help="deterministic rule")
    group.add_argument("--hermes", action="store_true", help="call hermes for the change")
    args = parser.parse_args()

    sd = state_dir()
    if args.fallback:
        run_fallback(sd)
    else:
        run_hermes(sd)


if __name__ == "__main__":
    main()
