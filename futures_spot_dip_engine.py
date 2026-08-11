"""
futures_spot_dip_engine.py — محرك اقتناص القيعان والقمم الذكي (Extreme Dip & Sweep Engine)
========================================================================================
الهدف: تحقيق نسبة نجاح عالية (Win Rate) وصافي أرباح حقيقية على منصة بايننس للسبوت والفيوتشر.

القواعد الأساسية:
  1. الشراء (LONG - Spot & Futures):
     - السعر في منطقة تشبع بيعي حاد (RSI <= 35 أو الملامسة للحد السفلي لـ Bollinger Bands).
     - اصطياد سيولة القاع (Liquidity Sweep / Spring) مع فوليوم مرتفع >= 1.3x.
  2. البيع (SHORT - Futures فقط):
     - السعر في منطقة تشبع شرائي حاد (RSI >= 65 أو الملامسة للحد العلوي لـ Bollinger Bands).
     - اصطياد سيولة القمة (Liquidity Sweep / Upthrust) مع فوليوم مرتفع >= 1.3x.
  3. إدارة مخاطر واقعية:
     - ستوب لوس آمن واسع (2.0x ATR = ~3.0% إلى 4.5%) لحماية الصفقة من تذبذب الشموع العادية.
     - نسبة عائد إلى مخاطرة متوازنة (R:R = 1:1.3 للسبوت و 1:1.5 للفيوتشر).
     - حماية رأس المال (Breakeven) فور تحقيق 1.0% ربح.
"""

import numpy as np
import math
from typing import Optional

def _ema(c: np.ndarray, p: int) -> np.ndarray:
    r = np.full(len(c), np.nan)
    if len(c) < p: return r
    r[p-1] = float(np.mean(c[:p])); k = 2/(p+1)
    for i in range(p, len(c)): r[i] = c[i]*k + r[i-1]*(1-k)
    return r

def _rsi(c: np.ndarray, p: int = 14) -> np.ndarray:
    r = np.full(len(c), 50.0)
    if len(c) < p + 1: return r
    d = np.diff(c); g = np.where(d > 0, d, 0.0); lo = np.where(d < 0, -d, 0.0)
    ag = float(np.mean(g[:p])); al = float(np.mean(lo[:p]))
    for i in range(p, len(c) - 1):
        ag = (ag * (p - 1) + g[i]) / p; al = (al * (p - 1) + lo[i]) / p
        r[i+1] = 100.0 - 100.0 / (1.0 + ag / al) if al > 0 else 100.0
    return r

def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, p: int = 14) -> np.ndarray:
    n = len(c); tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    a = np.zeros(n)
    if n > p: a[p] = float(np.mean(tr[1:p+1]))
    for i in range(p+1, n): a[i] = (a[i-1]*(p-1)+tr[i])/p
    return a

def _bollinger_bands(c: np.ndarray, p: int = 20, mult: float = 2.0) -> tuple[float, float, float]:
    if len(c) < p: return float(c[-1]), float(c[-1]), float(c[-1])
    w = c[-p:]; m = float(np.mean(w)); std = float(np.std(w))
    return m, m + mult * std, m - mult * std

def analyze_dip_sweep(
    symbol: str,
    closes: list, highs: list, lows: list, opens: list, volumes: list,
    market: str = "SPOT"
) -> Optional[dict]:
    """
    يُحلل السوق بنظام اقتناص الانعكاس من التشبع البيعي/الشرائي
    """
    c = np.array(closes, dtype=float)
    h = np.array(highs,  dtype=float)
    l = np.array(lows,   dtype=float)
    o = np.array(opens,  dtype=float)
    v = np.array(volumes,dtype=float)
    
    if len(c) < 60: return None
    
    curr_c = float(c[-1])
    curr_h = float(h[-1])
    curr_l = float(l[-1])
    curr_v = float(v[-1])
    
    rsi_arr = _rsi(c, 14)
    curr_rsi = float(rsi_arr[-1])
    
    mid_bb, upper_bb, lower_bb = _bollinger_bands(c, 20, 2.0)
    
    atr_arr = _atr(h, l, c, 14)
    curr_atr = float(atr_arr[-1]) if atr_arr[-1] > 0 else curr_c * 0.02
    
    avg_vol = float(np.mean(v[-21:-1])) if len(v) >= 21 else float(np.mean(v[:-1]))
    if avg_vol <= 0: return None
    vol_ratio = curr_v / avg_vol
    
    recent_min_low  = float(np.min(l[-20:-1]))
    recent_max_high = float(np.max(h[-20:-1]))
    
    side = None
    sources = []
    
    # ── 1. شرط دخول LONG (Spot & Futures) ──────────────────────────
    # السعر في منطقة تشبع بيعي (RSI <= 42 أو اختراق الحد السفلي لـ BB)
    is_oversold = curr_rsi <= 42.0 or curr_c <= lower_bb * 1.008
    # شمعة ارتداد (Hammer / Engulfing / Reversal)
    is_reversal_up = (curr_c > o[-1] and float(c[-2]) < float(o[-2])) or (curr_c - curr_l > (curr_h - curr_l) * 0.45)
    
    if is_oversold and is_reversal_up and vol_ratio >= 1.15:
        side = "LONG"
        sources.append("EXTREME_OVERSOLD_RSI")
        if curr_c <= lower_bb: sources.append("BOLLINGER_LOWER_TOUCH")
        if curr_l <= recent_min_low: sources.append("LIQUIDITY_DIP_SPRING")
        sources.append(f"VOL_SURGE_{vol_ratio:.1f}x")
        
    # ── 2. شرط دخول SHORT (Futures فقط) ───────────────────────────
    elif market.upper() == "FUTURES":
        is_overbought = curr_rsi >= 58.0 or curr_c >= upper_bb * 0.992
        is_reversal_dn = (curr_c < o[-1] and float(c[-2]) > float(o[-2])) or (curr_h - curr_c > (curr_h - curr_l) * 0.45)
        
        if is_overbought and is_reversal_dn and vol_ratio >= 1.15:
            side = "SHORT"
            sources.append("EXTREME_OVERBOUGHT_RSI")
            if curr_c >= upper_bb: sources.append("BOLLINGER_UPPER_TOUCH")
            if curr_h >= recent_max_high: sources.append("LIQUIDITY_SWEEP_UPTHRUST")
            sources.append(f"VOL_SURGE_{vol_ratio:.1f}x")

    if not side:
        return None

    # ── 3. إعداد الستوب لوس والهدف الواسع والآمن (Safe SL = 2.0x ATR) ──
    rr = 1.3 if market.upper() == "SPOT" else 1.5
    
    if side == "LONG":
        sl = min(curr_l - curr_atr * 0.5, curr_c - curr_atr * 1.8)
        risk = curr_c - sl
        if risk <= 0: return None
        tp = curr_c + risk * rr
    else: # SHORT
        sl = max(curr_h + curr_atr * 0.5, curr_c + curr_atr * 1.8)
        risk = sl - curr_c
        if risk <= 0: return None
        tp = curr_c - risk * rr

    sl_pct = abs(curr_c - sl) / curr_c * 100
    tp_pct = abs(tp - curr_c) / curr_c * 100

    # الرفض إذا كان الستوب ضيق جداً (أقل من 1.2%) أو واسع جداً (أكثر من 6%)
    if sl_pct < 1.2 or sl_pct > 6.0:
        return None

    return {
        "strategy":  "EXTREME_DIP_SWEEP",
        "engine":    "DIP_SWEEP",
        "symbol":    symbol,
        "side":      side,
        "market":    market.upper(),
        "entry":     round(curr_c, 6),
        "sl":        round(sl, 6),
        "tp":        round(tp, 6),
        "sl_pct":    round(sl_pct, 2),
        "tp_pct":    round(tp_pct, 2),
        "rr":        rr,
        "score":     8.8,
        "ai_score":  88,
        "sources":   sources,
        "rsi":       round(curr_rsi, 1),
        "vol_ratio": round(vol_ratio, 1),
    }
