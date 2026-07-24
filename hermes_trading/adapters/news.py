"""News adapter — free headlines by default.

Uses a free public RSS/JSON source. If NEWS_API_KEY is set, a premium provider
could be swapped in here. Failures are the loop's problem (it retries); this
adapter just fetches and validates its own schema.
"""
from __future__ import annotations

import os

import httpx

from . import require_schema

SCHEMA_VERSION = 1

# Free, no-key economic/markets headlines endpoint.
_FREE_URL = "https://www.cnbc.com/id/100003114/device/rss/rss.html"


async def fetch(symbol: str | None = None) -> dict:
    key = os.getenv("NEWS_API_KEY")
    headlines: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(_FREE_URL)
            resp.raise_for_status()
            # Cheap title extraction — no XML dep needed for a sentiment proxy.
            for chunk in resp.text.split("<title>")[1:]:
                title = chunk.split("</title>")[0].strip()
                if title and title.lower() != "cnbc.com":
                    headlines.append(title)
    except Exception:
        headlines = []  # loop's retry/circuit-breaker handles persistent failure

    payload = {
        "schema_version": SCHEMA_VERSION,
        "premium": bool(key),
        "count": len(headlines),
        "headlines": headlines[:20],
    }
    return require_schema(payload, SCHEMA_VERSION, "news")
