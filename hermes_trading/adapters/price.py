"""Price adapter — futures OHLCV via yfinance (paper-mode price feed).

MNQ/MGC are CME futures and are NOT on ccxt/crypto exchanges, so we pull the
continuous front-month proxy from yfinance (NQ=F, GC=F). Paper mode only needs
prices to simulate fills; no exchange credentials are required.

Returns full OHLC series (the Lorentzian classifier needs high/low for CCI, ADX
and WaveTrend). `closes`/`last` are retained for back-compat with older callers.

If EXCHANGE_API_KEY is set in .env you could swap in a real futures feed here,
but the default is free public data.
"""
from __future__ import annotations

import asyncio
import os

import yfinance as yf

from . import require_schema

SCHEMA_VERSION = 2  # bumped: added OHLCV arrays (was closes-only at v1)

# Enough history for the classifier's max_bars_back (~2000) plus feature warmup.
MAX_BARS = 2500


def _col(df, name: str) -> list[float]:
    series = df[name].dropna()
    return [float(x) for x in series.to_numpy().ravel().tolist()]


def _sync_fetch(feed: str, lookback: str, interval: str) -> dict:
    df = yf.download(feed, period=lookback, interval=interval,
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"price: yfinance returned no data for {feed}")
    df = df.dropna().tail(MAX_BARS)
    opens = _col(df, "Open")
    highs = _col(df, "High")
    lows = _col(df, "Low")
    closes = _col(df, "Close")
    volumes = _col(df, "Volume") if "Volume" in df else [0.0] * len(closes)
    n = min(len(opens), len(highs), len(lows), len(closes))
    return {
        "schema_version": SCHEMA_VERSION,
        "feed": feed,
        "interval": interval,
        "opens": opens[-n:],
        "highs": highs[-n:],
        "lows": lows[-n:],
        "closes": closes[-n:],
        "volumes": volumes[-n:],
        "last": closes[-1] if closes else None,
        "n": n,
    }


async def fetch(feed: str = "NQ=F", lookback: str = "5d",
                interval: str = "1m") -> dict:
    """Return recent OHLCV for a futures feed. Premium key overrides are read
    from the environment but default to free yfinance data."""
    _ = os.getenv("EXCHANGE_API_KEY")  # reserved for a real futures feed swap
    payload = await asyncio.to_thread(_sync_fetch, feed, lookback, interval)
    return require_schema(payload, SCHEMA_VERSION, f"price[{feed}]")
