"""Entrypoint. Reads goal.yaml, starts the loop and API server.

Usage:
  python -m hermes_trading.run                 # all assets from goal.yaml
  python -m hermes_trading.run --asset MNQ     # restrict to one configured symbol
"""
from __future__ import annotations

import argparse
import asyncio
import os

import uvicorn
import yaml

from . import api_server, state_dir
from .loop import Worker


def load_goal() -> dict:
    return yaml.safe_load((state_dir() / "goal.yaml").read_text())


def resolve_assets(goal: dict, only: str | None) -> list[dict]:
    assets = goal.get("assets")
    if not assets:  # back-compat with a single-asset goal.yaml
        assets = [{"symbol": goal.get("asset", "MNQ"),
                   "feed": goal.get("feed", "NQ=F")}]
    if only:
        assets = [a for a in assets if a["symbol"].upper() == only.upper()]
        if not assets:
            raise SystemExit(f"--asset {only} not found in goal.yaml")
    return assets


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=None,
                        help="restrict to one configured symbol (e.g. MNQ)")
    args = parser.parse_args()

    goal = load_goal()
    assets = resolve_assets(goal, args.asset)
    worker = Worker(assets=assets, goal=goal)

    port = int(os.getenv("PORT", 8000))
    config = uvicorn.Config(api_server.app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)

    await asyncio.gather(
        worker.run_forever(),
        server.serve()
    )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
