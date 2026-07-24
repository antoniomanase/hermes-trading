"""Macro adapter — free market context via yfinance (VIX, DXY, 10y yield).

Relevant to futures: risk regime (VIX), dollar (DXY), rates (^TNX). Free data.
"""
from __future__ import annotations

import asyncio

import yfinance as yf

from . import require_schema

SCHEMA_VERSION = 1

_TICKERS = {"vix": "^VIX", "dxy": "DX-Y.NYB", "us10y": "^TNX"}


def _sync_fetch() -> dict:
    out: dict[str, float | None] = {}
    for name, tkr in _TICKERS.items():
        try:
            df = yf.download(tkr, period="5d", interval="1d",
                             progress=False, auto_adjust=False)
            closes = df["Close"].dropna()
            out[name] = float(closes.to_numpy().ravel()[-1]) if len(closes) else None
        except Exception:
            out[name] = None
    return out


async def fetch(symbol: str | None = None) -> dict:
    metrics = await asyncio.to_thread(_sync_fetch)
    payload = {"schema_version": SCHEMA_VERSION, "metrics": metrics}
    return require_schema(payload, SCHEMA_VERSION, "macro")
