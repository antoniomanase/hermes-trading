"""On-chain adapter.

On-chain data is a crypto concept and does not apply to CME futures (MNQ/MGC).
For a futures worker this adapter is a no-op that returns a valid, empty payload
so the loop's adapter contract still holds. If GLASSNODE_API_KEY is set AND a
crypto asset is configured, this is where a real on-chain fetch would go.
"""
from __future__ import annotations

import os

from . import require_schema

SCHEMA_VERSION = 1


async def fetch(symbol: str | None = None) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "applicable": bool(os.getenv("GLASSNODE_API_KEY")) and _is_crypto(symbol),
        "metrics": {},  # empty for futures
    }
    return require_schema(payload, SCHEMA_VERSION, "onchain")


def _is_crypto(symbol: str | None) -> bool:
    return bool(symbol) and "/" in symbol
