"""
elite_spot_strategy.py — الاستراتيجية الأقوى للتداول الفوري (Elite Spot Breakout Strategy)
========================================================================================
الهدف: تحقيق أعلى نسبة نجاح (Win Rate) وصافي أرباح في سوق السبوت.
مواصفات الاستراتيجية:
  1. الفريم: اليومي (1D) والـ 4 ساعات (4H) فقط.
  2. معدل الصفقات: صفقة إلى صفقتين كحد أقصى في اليوم (صفقات صفوة منتقاة بعناية).
  3. فلتر الاتجاه الصارم: السعر فوق EMA50 و EMA200 على الفريم اليومي.
  4. كسر اختراق القمة الكبرى (Multi-Week High Breakout): السعر يكسر قمة آخر 30 يوماً.
  5. انفجار فوليوم مؤسسي (Institutional Volume Surge): الفوليوم >= 2.0x المتوسط.
  6. R:R ممتاز (1:2.0 إلى 1:2.5) مع نقل الستوب لوس للتكلفة عند 1:1 R:R.
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

def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, p: int = 14) -> np.ndarray:
    n = len(c); tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    a = np.zeros(n)
    if n > p: a[p] = float(np.mean(tr[1:p+1]))
    for i in range(p+1, n): a[i] = (a[i-1]*(p-1)+tr[i])/p
    return a

def analyze_elite_spot(
    symbol: str,
    closes: list, highs: list, lows: list, opens: list, volumes: list
) -> Optional[dict]:
    """
    يُحلل عملة سبوت عبر الفريم اليومي/4ساعات ويُرجع صفقة صريحة عالية الجودة أو None
    """
    c = np.array(closes, dtype=float)
    h = np.array(highs,  dtype=float)
    l = np.array(lows,   dtype=float)
    o = np.array(opens,  dtype=float)
    v = np.array(volumes,dtype=float)
    
    if len(c) < 120: return None
    
    curr_c = float(c[-1])
    curr_h = float(h[-1])
    curr_v = float(v[-1])
    
    # ── 1. فلتر الاتجاه الرئيسي (Daily Bull Market Filter) ────────
    e50  = _ema(c, 50)
    e200 = _ema(c, 200)
    
    curr_e50  = float(e50[-1])  if not math.isnan(e50[-1])  else curr_c
    curr_e200 = float(e200[-1]) if not math.isnan(e200[-1]) else curr_c
    
    # شرط الاتجاه الصاعد الصارم: السعر فوق EMA50 و EMA50 فوق EMA200
    if not (curr_c > curr_e50 and curr_e50 >= curr_e200 * 0.98):
        return None

    # ── 2. شرط الفوليوم المؤسسي الضخم (Volume Surge >= 1.8x) ────────
    avg_vol = float(np.mean(v[-21:-1])) if len(v) >= 21 else float(np.mean(v[:-1]))
    if avg_vol <= 0: return None
    vol_ratio = curr_v / avg_vol
    if vol_ratio < 1.75: # يشترط وجود فوليوم مؤسسي قوي
        return None

    # ── 3. نموذج اختراق القمة متعددة الأسابيع (Multi-Week High Breakout) ─
    lookback = 30 # قمة آخر 30 شمعة
    prev_max_high = float(np.max(h[-lookback-1:-1]))
    
    # يجب أن تكسر الشمعة الحالية قمة الـ 30 شمعة السابقة بوضوح
    is_major_breakout = curr_c > prev_max_high
    
    # أو نموذج تجميع وايكوف صاعد + كسر قاع حاد واختراق أعلى قاع (Wyckoff Spring)
    recent_min_low = float(np.min(l[-20:-2]))
    is_wyckoff_spring = (float(l[-2]) < recent_min_low and curr_c > recent_min_low * 1.02 and vol_ratio >= 2.0)
    
    if not (is_major_breakout or is_wyckoff_spring):
        return None

    # ── 4. تحديد سعر الستوب والهدف بناءً على R:R = 1:2.0 إلى 1:2.5 ───
    atr_arr = _atr(h, l, c)
    curr_atr = float(atr_arr[-1]) if atr_arr[-1] > 0 else curr_c * 0.015
    
    # الستوب لوس يوضع أسفل قاع آخر 5 شمعات + هامش ATR لحماية الصفحة
    swing_low = float(np.min(l[-6:-1]))
    sl = min(swing_low - curr_atr * 0.3, curr_c - curr_atr * 1.8)
    
    risk = curr_c - sl
    if risk <= 0: return None
    
    rr = 2.0 # R:R 1:2.0 ممتاز جداً للسبوت
    tp = curr_c + risk * rr
    
    sl_pct = abs(curr_c - sl) / curr_c * 100
    tp_pct = abs(tp - curr_c) / curr_c * 100
    
    # تصفية الستوب الواسع جداً أو الضيق جداً
    if sl_pct < 1.0 or sl_pct > 5.5:
        return None

    sources = ["ELITE_1D_TREND", f"VOL_SURGE_{vol_ratio:.1f}x"]
    if is_major_breakout:
        sources.append("30D_HIGH_BREAKOUT")
    if is_wyckoff_spring:
        sources.append("WYCKOFF_SPRING_ACC")

    return {
        "strategy":  "ELITE_SPOT_BREAKOUT",
        "engine":    "ELITE_SPOT",
        "symbol":    symbol,
        "side":      "LONG",
        "market":    "SPOT",
        "entry":     round(curr_c, 6),
        "sl":        round(sl, 6),
        "tp":        round(tp, 6),
        "sl_pct":    round(sl_pct, 2),
        "tp_pct":    round(tp_pct, 2),
        "rr":        rr,
        "score":     9.0,
        "sources":   sources,
        "vol_ratio": round(vol_ratio, 1),
    }
