"""Lorentzian Classification signal (after jdehorty's "Machine Learning:
Lorentzian Classification" on TradingView).

A k-nearest-neighbours classifier that rates historical bars by *Lorentzian
distance* across a set of normalized feature series (RSI, WaveTrend, CCI, ADX),
votes with their realized future direction, and gates the vote through
volatility / regime / ADX filters and a Nadaraya-Watson kernel.

This is a faithful-but-pragmatic port: the neighbour vote and Lorentzian metric
match the original; the filters are documented approximations of jdehorty's and
are all tunable from strategy.yaml (so Hermes can refine them). Pure numpy —
runs in a few ms on ~2000 bars, deterministic, no lookahead in the prediction
for the current bar (labels only use bars whose future is already known).
"""
from __future__ import annotations

import numpy as np

# --- indicators (numpy, no external TA dep) --------------------------------


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == 0:
        return x
    alpha = 2.0 / (n + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _rma(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder's smoothing (used by RSI/ADX)."""
    alpha = 1.0 / n
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi(close: np.ndarray, n: int) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    avg_gain = _rma(gain, n)
    avg_loss = _rma(loss, n)
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.inf),
                   where=avg_loss > 0)
    return 100.0 - 100.0 / (1.0 + rs)


def _cci(high, low, close, n: int) -> np.ndarray:
    tp = (high + low + close) / 3.0
    sma = _rolling_mean(tp, n)
    mad = _rolling_mad(tp, n)
    return np.divide(tp - sma, 0.015 * mad,
                     out=np.zeros_like(tp), where=mad > 0)


def _wavetrend(high, low, close, n1: int, n2: int) -> np.ndarray:
    ap = (high + low + close) / 3.0
    esa = _ema(ap, n1)
    d = _ema(np.abs(ap - esa), n1)
    ci = np.divide(ap - esa, 0.015 * d, out=np.zeros_like(ap), where=d > 0)
    return _ema(ci, n2)


def _adx(high, low, close, n: int) -> np.ndarray:
    up = np.diff(high, prepend=high[0])
    dn = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = _true_range(high, low, close)
    atr = _rma(tr, n)
    plus_di = 100.0 * np.divide(_rma(plus_dm, n), atr,
                                out=np.zeros_like(atr), where=atr > 0)
    minus_di = 100.0 * np.divide(_rma(minus_dm, n), atr,
                                 out=np.zeros_like(atr), where=atr > 0)
    denom = plus_di + minus_di
    dx = 100.0 * np.divide(np.abs(plus_di - minus_di), denom,
                           out=np.zeros_like(denom), where=denom > 0)
    return _rma(dx, n)


def _true_range(high, low, close) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    return np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])


def atr(high, low, close, n: int = 14) -> np.ndarray:
    return _rma(_true_range(high, low, close), n)


def _rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) >= n:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    for i in range(min(n - 1, len(x))):
        out[i] = x[: i + 1].mean()
    return out


def _rolling_mad(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        lo = max(0, i - n + 1)
        w = x[lo: i + 1]
        out[i] = np.abs(w - w.mean()).mean()
    return out


def _normalize(x: np.ndarray) -> np.ndarray:
    """Min-max to [0,1] so Lorentzian distances weight features evenly."""
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


# --- feature assembly ------------------------------------------------------


def _feature(kind: str, a: int, b: int, o, h, l, c) -> np.ndarray:
    kind = kind.lower()
    if kind == "rsi":
        return _normalize(_rsi(c, a))
    if kind == "wt":
        return _normalize(_wavetrend(h, l, c, a, b))
    if kind == "cci":
        return _normalize(_cci(h, l, c, a))
    if kind == "adx":
        return _normalize(_adx(h, l, c, a))
    raise ValueError(f"unknown feature kind: {kind}")


# --- Nadaraya-Watson rational-quadratic kernel -----------------------------


def _rational_quadratic(src: np.ndarray, lookback: int, weight: float,
                        level: int) -> np.ndarray:
    """Kernel-regression estimate (jdehorty's kernel). Returns smoothed line."""
    n = len(src)
    out = np.copy(src).astype(float)
    for i in range(level, n):
        num = 0.0
        den = 0.0
        for j in range(0, min(i, 200)):
            w = (1.0 + (j * j) / (lookback * lookback * 2.0 * weight)) ** (-weight)
            num += src[i - j] * w
            den += w
        out[i] = num / den if den > 0 else src[i]
    return out


# --- classifier ------------------------------------------------------------


def classify(ohlc: dict, cfg: dict) -> dict:
    """Return {signal, prediction, confidence, filters} for the latest bar.

    signal is +1 (long), -1 (short) or 0 (stand aside).
    """
    o = np.asarray(ohlc["opens"], dtype=float)
    h = np.asarray(ohlc["highs"], dtype=float)
    l = np.asarray(ohlc["lows"], dtype=float)
    c = np.asarray(ohlc["closes"], dtype=float)

    neighbors = int(cfg.get("neighbors", 8))
    max_bars = int(cfg.get("max_bars_back", 2000))
    future = 4  # jdehorty labels each bar by the move 4 bars ahead

    if len(c) < max(120, future + neighbors + 5):
        return {"signal": 0, "prediction": 0.0, "confidence": 0.0,
                "filters": {"reason": "insufficient_bars", "n": int(len(c))}}

    feats = np.column_stack([
        _feature(f["kind"], int(f.get("a", 14)), int(f.get("b", 1)), o, h, l, c)
        for f in cfg["feature_series"]
    ])  # shape (n_bars, n_feat)

    n = len(c)
    lo = max(0, n - max_bars)
    # labels: +1 if price rose over the next `future` bars, else -1.
    fut = np.sign(np.roll(c, -future) - c)  # last `future` entries invalid
    cur = feats[-1]

    # training set: bars with a known future, chronologically sub-sampled (i%4)
    idx = np.arange(lo, n - future)
    idx = idx[idx % 4 == 0]
    if len(idx) <= neighbors:
        return {"signal": 0, "prediction": 0.0, "confidence": 0.0,
                "filters": {"reason": "insufficient_history"}}

    # Lorentzian distance: sum_f log(1 + |x_f - y_f|)
    d = np.log1p(np.abs(feats[idx] - cur)).sum(axis=1)
    k = min(neighbors, len(idx))
    nn = np.argpartition(d, k - 1)[:k]
    prediction = float(fut[idx[nn]].sum())  # in [-k, +k]

    raw = int(np.sign(prediction))
    filt = _apply_filters(o, h, l, c, feats, raw, cfg)
    signal = raw if (raw != 0 and filt["pass"]) else 0
    return {
        "signal": signal,
        "prediction": prediction,
        "confidence": abs(prediction) / max(1, k),
        "filters": filt,
    }


def _apply_filters(o, h, l, c, feats, raw, cfg) -> dict:
    f = cfg.get("filters", {})
    out = {"pass": True}

    tr = _true_range(h, l, c)
    atr_recent = tr[-10:].mean()
    atr_hist = tr[-100:].mean() if len(tr) >= 100 else tr.mean()
    vol_ok = True
    if f.get("volatility", True):
        vol_ok = atr_recent >= 0.75 * atr_hist  # avoid dead low-vol chop
    out["volatility_ok"] = bool(vol_ok)

    regime_ok = True
    if f.get("regime", True):
        w = 20
        y = c[-w:]
        x = np.arange(w)
        slope = np.polyfit(x, y, 1)[0]
        slope_norm = slope / (np.mean(y) + 1e-9) * w  # dimensionless trend
        thr = float(f.get("regime_threshold", -0.1))
        if raw > 0:
            regime_ok = slope_norm >= thr          # don't long a hard downtrend
        elif raw < 0:
            regime_ok = slope_norm <= -thr         # don't short a hard uptrend
        out["slope_norm"] = round(float(slope_norm), 4)
    out["regime_ok"] = bool(regime_ok)

    adx_ok = True
    if f.get("adx", False):
        adx_val = _adx(h, l, c, 14)[-1]
        adx_ok = adx_val >= float(f.get("adx_threshold", 20))
        out["adx"] = round(float(adx_val), 2)
    out["adx_ok"] = bool(adx_ok)

    kernel_ok = True
    k = cfg.get("kernel", {})
    if k.get("enabled", True):
        est = _rational_quadratic(
            c, int(k.get("lookback", 8)), float(k.get("relative_weight", 8.0)),
            int(k.get("regression_level", 25)))
        rising = est[-1] > est[-2]
        if raw > 0:
            kernel_ok = rising
        elif raw < 0:
            kernel_ok = not rising
    out["kernel_ok"] = bool(kernel_ok)

    out["pass"] = bool(vol_ok and regime_ok and adx_ok and kernel_ok)
    return out
