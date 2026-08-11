"""
mtf_futures_engine.py — محرك التداول متعدد الأطر الزمنية (Multi-Timeframe MTF Futures Engine)
========================================================================================
الهدف: قنص صفقات الفيوتشر بسعر دخول دقيق وستوب لوس ضيق جداً (0.3% - 0.8%)
من خلال التحليل التنازلي Top-Down Analysis:

  1. الفريم الأكبر (4H / 1H)  -> تحديد الاتجاه العام ومناطق السيولة الكبرى (Bias & Macro OB)
  2. الفريم الأصغر (15M / 5M) -> تحديد نقطة الدخول الدقيقة عند كسر الهيكل الصغير (Micro CHoCH) 
                                واختبار منطقة Order Block صغيرة (5M OB / FVG)

النتيجة: R:R مرتفع جداً (1:2.5 إلى 1:4) مع نسبة مخاطرة صغيرة جداً.
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

def _rsi(c: np.ndarray, p: int = 14) -> float:
    if len(c) < p + 1: return 50.0
    d = np.diff(c); g = np.where(d > 0, d, 0.0); lo = np.where(d < 0, -d, 0.0)
    ag = float(np.mean(g[:p])); al = float(np.mean(lo[:p]))
    for i in range(p, len(c) - 1):
        ag = (ag * (p - 1) + g[i]) / p; al = (al * (p - 1) + lo[i]) / p
    return 100.0 - 100.0 / (1.0 + ag / al) if al > 0 else 100.0


def analyze_mtf_futures(
    symbol: str,
    data_macro: dict, # بيانات 1H أو 4H (closes, highs, lows, opens, volumes)
    data_micro: dict, # بيانات 15M أو 5M (closes, highs, lows, opens, volumes)
) -> Optional[dict]:
    """
    يُحلل السوق بنظام MTF يُرجع إشارة فيوتشر احترافية أو None
    """
    # ── 1. تحليل الفريم الأكبر (Macro 1H/4H Bias) ─────────────
    mc_c = np.array(data_macro["closes"], dtype=float)
    mc_h = np.array(data_macro["highs"],  dtype=float)
    mc_l = np.array(data_macro["lows"],   dtype=float)
    mc_v = np.array(data_macro["volumes"],dtype=float)
    
    if len(mc_c) < 50: return None
    
    macro_e50 = _ema(mc_c, 50)
    macro_e200= _ema(mc_c, 200)
    
    curr_mc_c = float(mc_c[-1])
    curr_e50  = float(macro_e50[-1]) if not math.isnan(macro_e50[-1]) else curr_mc_c
    curr_e200 = float(macro_e200[-1]) if not math.isnan(macro_e200[-1]) else curr_mc_c
    
    # تحديد الاتجاه العام للفريم الأكبر
    macro_bias = "NEUTRAL"
    if curr_mc_c > curr_e50 and curr_e50 > curr_e200:
        macro_bias = "BULLISH"
    elif curr_mc_c < curr_e50 and curr_e50 < curr_e200:
        macro_bias = "BEARISH"
    elif curr_mc_c > curr_e50:
        macro_bias = "BULLISH"
    elif curr_mc_c < curr_e50:
        macro_bias = "BEARISH"
        
    if macro_bias == "NEUTRAL":
        return None # لا ندخل صفقات في سوق متذبذب بدون اتجاه واضح

    # ── 2. تحليل الفريم الدقيق (Micro 5M/15M Precision Entry) ─
    mi_c = np.array(data_micro["closes"], dtype=float)
    mi_h = np.array(data_micro["highs"],  dtype=float)
    mi_l = np.array(data_micro["lows"],   dtype=float)
    mi_o = np.array(data_micro["opens"],  dtype=float)
    mi_v = np.array(data_micro["volumes"],dtype=float)
    
    if len(mi_c) < 60: return None
    
    n_mi = len(mi_c)
    curr_entry = float(mi_c[-1])
    atr_mi = _atr(mi_h, mi_l, mi_c)
    curr_atr = float(atr_mi[-1]) if atr_mi[-1] > 0 else curr_entry * 0.003
    avg_vol  = float(np.mean(mi_v[-20:])) if len(mi_v) >= 20 else float(np.mean(mi_v))
    vol_spike= avg_vol > 0 and mi_v[-1] >= avg_vol * 1.35
    
    # أسباب الدخول الدقيقة
    micro_signal = None
    sl_level = None
    sources = [f"MACRO_{macro_bias}_BIAS"]
    
    if macro_bias == "BULLISH":
        # ابحث عن كسر هيكل صغير صاعد (Micro CHoCH) + Order Block صغير في آخر 15 شمعة
        for j in range(n_mi-15, n_mi-2):
            if j < 1: continue
            # Micro Order Block: شمعة هابطة تليها صعود قوي
            if mi_c[j] < mi_o[j] and mi_c[j+1] > mi_o[j+1] and mi_c[j+2] > mi_c[j+1]:
                ob_low  = float(mi_l[j])
                ob_high = float(mi_h[j])
                # السعر الحالي يختبر هذا الـ OB الصغير
                if ob_low * 0.998 <= curr_entry <= ob_high * 1.005:
                    micro_signal = "LONG"
                    sl_level = ob_low - curr_atr * 0.3
                    sources.append("MICRO_5M_OB_RETEST")
                    break
        
        # كشف Liquidity Sweep صغير على فريم 5m
        if not micro_signal and n_mi >= 20:
            recent_low = float(np.min(mi_l[-20:-3]))
            if float(mi_l[-2]) < recent_low and float(mi_c[-1]) > recent_low and vol_spike:
                micro_signal = "LONG"
                sl_level = float(mi_l[-2]) - curr_atr * 0.2
                sources.append("MICRO_5M_LIQUIDITY_SWEEP")

    elif macro_bias == "BEARISH":
        for j in range(n_mi-15, n_mi-2):
            if j < 1: continue
            if mi_c[j] > mi_o[j] and mi_c[j+1] < mi_o[j+1] and mi_c[j+2] < mi_c[j+1]:
                ob_high = float(mi_h[j])
                ob_low  = float(mi_l[j])
                if ob_low * 0.995 <= curr_entry <= ob_high * 1.002:
                    micro_signal = "SHORT"
                    sl_level = ob_high + curr_atr * 0.3
                    sources.append("MICRO_5M_OB_RETEST")
                    break
                    
        if not micro_signal and n_mi >= 20:
            recent_high = float(np.max(mi_h[-20:-3]))
            if float(mi_h[-2]) > recent_high and float(mi_c[-1]) < recent_high and vol_spike:
                micro_signal = "SHORT"
                sl_level = float(mi_h[-2]) + curr_atr * 0.2
                sources.append("MICRO_5M_LIQUIDITY_SWEEP")

    if not micro_signal or sl_level is None:
        return None

    # ── 3. حساب الأهداف ومخاطرة الفيوتشر الـ Tight (R:R = 1:2.5) ──
    rr = 2.5 # نسبة مخاطرة إلى عائد ممتازة للفيوتشر
    
    if micro_signal == "LONG":
        sl = min(sl_level, curr_entry - curr_atr * 0.5)
        risk = curr_entry - sl
        if risk <= 0: return None
        tp = curr_entry + risk * rr
    else: # SHORT
        sl = max(sl_level, curr_entry + curr_atr * 0.5)
        risk = sl - curr_entry
        if risk <= 0: return None
        tp = curr_entry - risk * rr

    sl_pct = abs(curr_entry - sl) / curr_entry * 100
    tp_pct = abs(tp - curr_entry) / curr_entry * 100

    # الرفض إذا كان الستوب واسعاً أكثر من اللزوم على فريم دقيق
    if sl_pct > 1.8 or sl_pct < 0.2:
        return None

    rsi_val = _rsi(mi_c)

    return {
        "strategy":  "MTF_FUTURES_PRECISION",
        "engine":    "MTF_FUTURES",
        "symbol":    symbol,
        "side":      micro_signal,
        "market":    "FUTURES",
        "entry":     round(curr_entry, 6),
        "sl":        round(sl, 6),
        "tp":        round(tp, 6),
        "sl_pct":    round(sl_pct, 2),
        "tp_pct":    round(tp_pct, 2),
        "rr":        rr,
        "score":     8.5,
        "ai_score":  85,
        "sources":   sources,
        "rsi":       round(rsi_val, 1),
        "macro_bias":macro_bias,
    }
