"""
devel_master_strategy.py — استراتيجية DEVEL_MASTER الشاملة
==============================================================
تجمع خمس مدارس تحليل فني في محرك واحد:

  [1] Classic   — ترند هابط طويل، Fibonacci، Flag، S/R Retest، قنوات
  [2] ICT       — Order Block، Fair Value Gap، Liquidity Sweep، BOS/CHoCH
  [3] Wyckoff   — Accumulation/Distribution، Spring، Sign of Strength
  [4] Candles   — Engulfing، Pin Bar، Hammer، Morning/Evening Star، Marubozu
  [5] Momentum  — RSI divergence، MACD cross، ATR expansion، Volume spike

الدخول: ≥ 5.5 نقطة من 10 متفقة على نفس الاتجاه
Futures RR = 1.5 | Spot RR = 1.3
SL = خلف Order Block أو Swing HL + ATR buffer

الواجهة الرئيسية:
    build_signal(symbol, closes, highs, lows, opens, volumes, market) → dict | None
"""

from __future__ import annotations
import os
import math
import numpy as np
from typing import Optional

# ── إعدادات ──────────────────────────────────────────────────
RR_FUTURES        = float(os.getenv("DM_RR_FUTURES",   "1.5"))
RR_SPOT           = float(os.getenv("DM_RR_SPOT",      "1.3"))
MIN_ENTRY_SCORE   = float(os.getenv("DM_MIN_SCORE",    "5.5"))
SL_ATR_BUFFER     = float(os.getenv("DM_SL_ATR_BUF",  "0.4"))  # ATR buffer فوق/تحت منطقة SL
MIN_SCORE         = int(os.getenv("DM_AI_MIN_SCORE", "60"))       # للتوافق مع trading_bot

# ═══════════════════════════════════════════════════════════════
# 1. مؤشرات فنية أساسية
# ═══════════════════════════════════════════════════════════════

def _ema(c: np.ndarray, p: int) -> np.ndarray:
    r = np.full(len(c), np.nan)
    if len(c) < p: return r
    r[p-1] = float(np.mean(c[:p])); k = 2/(p+1)
    for i in range(p, len(c)): r[i] = c[i]*k + r[i-1]*(1-k)
    return r

def _rsi(c: np.ndarray, p: int = 14) -> np.ndarray:
    r = np.full(len(c), 50.0)
    if len(c) < p+1: return r
    d = np.diff(c)
    g = np.where(d>0, d, 0.0); lo = np.where(d<0, -d, 0.0)
    ag = np.mean(g[:p]); al = np.mean(lo[:p])
    for i in range(p, len(c)-1):
        ag = (ag*(p-1)+g[i])/p; al = (al*(p-1)+lo[i])/p
        r[i+1] = 100-100/(1+ag/al) if al > 0 else 100.0
    return r

def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, p: int = 14) -> np.ndarray:
    n = len(c); tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    a = np.zeros(n)
    if n > p: a[p] = float(np.mean(tr[1:p+1]))
    for i in range(p+1, n): a[i] = (a[i-1]*(p-1)+tr[i])/p
    return a

def _macd(c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e12 = _ema(c, 12); e26 = _ema(c, 26)
    ml  = np.where(np.isnan(e12)|np.isnan(e26), np.nan, e12-e26)
    sig = _ema(np.where(np.isnan(ml), 0, ml), 9)
    return ml, sig, ml-sig

def _adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, p: int = 14) -> tuple[float, float, float]:
    n = len(c)
    if n < p*2+5: return 0.0, 50.0, 50.0
    pdm=np.zeros(n); mdm=np.zeros(n); tr=np.zeros(n)
    for i in range(1, n):
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        pdm[i]=up if (up>dn and up>0) else 0
        mdm[i]=dn if (dn>up and dn>0) else 0
        tr[i]=max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    sTR=sum(tr[1:p+1]); sP=sum(pdm[1:p+1]); sM=sum(mdm[1:p+1])
    dx=[]; pdi=mdi=0.0
    for i in range(p+1, n):
        sTR=sTR-sTR/p+tr[i]; sP=sP-sP/p+pdm[i]; sM=sM-sM/p+mdm[i]
        pdi=(sP/sTR)*100 if sTR else 0; mdi=(sM/sTR)*100 if sTR else 0
        den=pdi+mdi; dx.append(abs(pdi-mdi)/den*100 if den else 0)
    return float(np.mean(dx[-p:])) if len(dx)>=p else 0.0, pdi, mdi

def _obv(c: np.ndarray, v: np.ndarray) -> np.ndarray:
    o = np.zeros(len(c))
    for i in range(1, len(c)):
        o[i] = o[i-1] + (v[i] if c[i]>c[i-1] else -v[i] if c[i]<c[i-1] else 0)
    return o

def _bollinger(c: np.ndarray, p: int=20, mult: float=2.0) -> tuple[float,float,float]:
    if len(c) < p: return np.nan, np.nan, np.nan
    w=c[-p:]; m=float(np.mean(w)); s=float(np.std(w))
    return m, m+mult*s, m-mult*s

def _swing_highs(h: np.ndarray, lb: int=5) -> list[int]:
    return [i for i in range(lb, len(h)-lb) if h[i]==max(h[i-lb:i+lb+1])]

def _swing_lows(l: np.ndarray, lb: int=5) -> list[int]:
    return [i for i in range(lb, len(l)-lb) if l[i]==min(l[i-lb:i+lb+1])]


# ═══════════════════════════════════════════════════════════════
# 2. التحليل الكلاسيكي (Classic)
# ═══════════════════════════════════════════════════════════════

def _count_downtrend_candles(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                              lookback: int = 60) -> int:
    """كم شمعة في الترند الهابط قبل الشمعة الأخيرة (lower highs + lower lows)"""
    n = min(lookback, len(h)-2)
    count = 0
    for i in range(len(h)-2, len(h)-2-n, -1):
        if i < 1: break
        if h[i] < h[i-1] and l[i] < l[i-1]:
            count += 1
        else:
            break
    return count

def _count_uptrend_candles(h: np.ndarray, l: np.ndarray, lookback: int=60) -> int:
    """كم شمعة في الترند الصاعد"""
    n = min(lookback, len(h)-2)
    count = 0
    for i in range(len(h)-2, len(h)-2-n, -1):
        if i < 1: break
        if h[i] > h[i-1] and l[i] > l[i-1]:
            count += 1
        else:
            break
    return count

def _fibonacci_score(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                     v: np.ndarray, avg_vol: float) -> tuple[float, str]:
    """
    رسم Fibonacci من أعلى قمة إلى أدنى قاع في آخر 100 شمعة.
    يُرجع (score, side) عند اختراق مستوى ذهبي بزخم.
    """
    if len(c) < 50: return 0.0, "none"
    window = min(100, len(c)-1)
    hh = float(np.max(h[-window:]))
    ll = float(np.min(l[-window:]))
    rng = hh - ll
    if rng < 1e-10: return 0.0, "none"

    fib_levels = {
        "0.236": ll + rng * 0.236,
        "0.382": ll + rng * 0.382,
        "0.500": ll + rng * 0.500,
        "0.618": ll + rng * 0.618,   # الذهبي
        "0.786": ll + rng * 0.786,   # شبه الذهبي
    }
    curr = float(c[-1]); prev = float(c[-2])
    vol_ok = avg_vol > 0 and v[-1] >= avg_vol * 1.3

    for name, level in fib_levels.items():
        tol = rng * 0.008  # 0.8% هامش
        # اختراق صاعد لمستوى فيبوناتشي بزخم
        if prev < level - tol and curr > level + tol and vol_ok:
            weight = 2.0 if name in ("0.618", "0.786") else 1.0
            return weight, "LONG"
        # اختراق هابط
        if prev > level + tol and curr < level - tol and vol_ok:
            weight = 2.0 if name in ("0.618", "0.786") else 1.0
            return weight, "SHORT"
    return 0.0, "none"

def _find_sr_levels(h: np.ndarray, l: np.ndarray, min_touches: int=2,
                    tol: float=0.005) -> list[tuple[float,str,int]]:
    """
    إيجاد مستويات S/R بلمسات متعددة.
    يُرجع: [(سعر, نوع, عدد_لمسات), ...]
    """
    all_prices = [(float(x), "R") for x in h] + [(float(x), "S") for x in l]
    groups: list[dict] = []
    for price, kind in all_prices:
        merged = False
        for g in groups:
            if abs(price - g["price"]) / g["price"] < tol:
                g["count"] += 1
                g["kind"]   = kind
                merged = True; break
        if not merged:
            groups.append({"price": price, "count": 1, "kind": kind})
    return [(g["price"], g["kind"], g["count"])
            for g in groups if g["count"] >= min_touches]

def _sr_breakout_retest_score(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                               v: np.ndarray, avg_vol: float) -> tuple[float, str]:
    """
    يكتشف: اختراق مقاومة/دعم قوي + إعادة اختبار = أفضل مناطق دخول
    يُرجع (score, side)
    """
    if len(c) < 60: return 0.0, "none"
    levels = _find_sr_levels(h[:-5], l[:-5], min_touches=2)
    curr = float(c[-1]); curr_l = float(l[-1]); curr_h = float(h[-1])
    vol_ok = avg_vol > 0 and v[-1] >= avg_vol * 1.2

    for level, kind, touches in levels:
        tol = level * 0.004
        # اختراق صاعد سابق + الآن إعادة اختبار (retest)
        recent_break_up = any(float(c[i]) > level + tol and float(c[i-1]) < level
                              for i in range(len(c)-15, len(c)-3))
        if recent_break_up and level - tol <= curr_l <= level + tol*2 and curr > level:
            weight = min(2.5, 1.0 + touches * 0.25)
            if vol_ok: weight += 0.3
            return weight, "LONG"

        # كسر هابط سابق + إعادة اختبار (retest)
        recent_break_dn = any(float(c[i]) < level - tol and float(c[i-1]) > level
                              for i in range(len(c)-15, len(c)-3))
        if recent_break_dn and level - tol*2 <= curr_h <= level + tol and curr < level:
            weight = min(2.5, 1.0 + touches * 0.25)
            if vol_ok: weight += 0.3
            return weight, "SHORT"

    return 0.0, "none"

def _channel_breakout_score(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                             v: np.ndarray, avg_vol: float) -> tuple[float, str]:
    """كشف اختراق قناة صاعدة (بيع) أو هابطة (شراء)"""
    if len(c) < 30: return 0.0, "none"
    win = min(40, len(c)-2)
    # رسم خطي (Linear regression) للـ highs والـ lows
    x = np.arange(win)
    ch  = h[-win-1:-1]; cl = l[-win-1:-1]
    try:
        ph = np.polyfit(x, ch, 1); pl = np.polyfit(x, cl, 1)
        upper_end = np.polyval(ph, win); lower_end = np.polyval(pl, win)
    except Exception:
        return 0.0, "none"

    curr = float(c[-1]); vol_ok = avg_vol>0 and v[-1]>=avg_vol*1.4

    # اختراق صاعد لقناة هابطة
    if ph[0] < -0.00001 and pl[0] < -0.00001:  # قناة هابطة
        if curr > upper_end and vol_ok:
            return 1.5, "LONG"

    # اختراق هابط لقناة صاعدة
    if ph[0] > 0.00001 and pl[0] > 0.00001:  # قناة صاعدة
        if curr < lower_end and vol_ok:
            return 1.5, "SHORT"

    return 0.0, "none"

def _bull_flag_score(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                     v: np.ndarray) -> tuple[float, str]:
    """
    كشف Bull Flag: ارتفاع > 15% ثم تراجع منظم 38-55% بحجم أقل
    """
    if len(c) < 30: return 0.0, "none"
    # ابحث عن pole (الارتفاع الحاد)
    for pole_len in range(5, 20):
        pole_start = len(c) - pole_len - 15
        if pole_start < 0: break
        pole_gain = (c[pole_start+pole_len] - c[pole_start]) / c[pole_start] * 100
        if pole_gain < 12: continue  # الارتفاع يجب أن يكون > 12%

        # فحص التراجع (flag): يجب أن يكون هادئاً وبحجم أقل
        flag_zone = c[pole_start+pole_len:]
        if len(flag_zone) < 3: continue
        flag_low = float(np.min(flag_zone))
        flag_top = float(c[pole_start+pole_len])
        retrace  = (flag_top - flag_low) / (flag_top - c[pole_start]) * 100
        if not (30 <= retrace <= 60): continue  # تراجع 30-60% من الـ pole

        # الشمعة الأخيرة تخترق قمة الـ flag
        if c[-1] > flag_top * 0.998:
            return 1.5, "LONG"

    return 0.0, "none"


# ═══════════════════════════════════════════════════════════════
# 3. ICT (Inner Circle Trader)
# ═══════════════════════════════════════════════════════════════

def _find_order_blocks(o: np.ndarray, h: np.ndarray, l: np.ndarray,
                       c: np.ndarray, lookback: int=30) -> list[dict]:
    """
    Order Block = آخر شمعة هابطة قبل حركة صاعدة حادة / العكس
    """
    blocks = []
    n = min(lookback, len(c)-3)
    for i in range(len(c)-n, len(c)-3):
        # Bullish OB: شمعة هابطة تليها 3 شمعات صاعدة متتالية
        if c[i] < o[i]:  # شمعة هابطة
            up_move = all(c[i+j] > c[i+j-1] for j in range(1, min(4, len(c)-i)))
            if up_move:
                blocks.append({
                    "type": "bullish",
                    "high": float(h[i]),
                    "low":  float(l[i]),
                    "idx":  i,
                })
        # Bearish OB: شمعة صاعدة تليها 3 شمعات هابطة
        elif c[i] > o[i]:  # شمعة صاعدة
            dn_move = all(c[i+j] < c[i+j-1] for j in range(1, min(4, len(c)-i)))
            if dn_move:
                blocks.append({
                    "type": "bearish",
                    "high": float(h[i]),
                    "low":  float(l[i]),
                    "idx":  i,
                })
    return blocks

def _ict_order_block_score(o: np.ndarray, h: np.ndarray, l: np.ndarray,
                           c: np.ndarray) -> tuple[float, str, Optional[float]]:
    """
    هل السعر الحالي عند منطقة Order Block؟
    يُرجع (score, side, ob_low/ob_high للـ SL)
    """
    if len(c) < 30: return 0.0, "none", None
    obs = _find_order_blocks(o, h, l, c)
    curr = float(c[-1])
    best = 0.0; best_side = "none"; sl_level = None

    for ob in obs:
        # السعر يلمس Bullish OB (منطقة شراء)
        if ob["type"] == "bullish" and ob["low"] <= curr <= ob["high"] * 1.01:
            age = len(c) - ob["idx"]
            score = 2.0 if age < 10 else (1.5 if age < 20 else 1.0)
            if score > best:
                best = score; best_side = "LONG"; sl_level = ob["low"]
        # السعر يلمس Bearish OB (منطقة بيع)
        elif ob["type"] == "bearish" and ob["low"] * 0.99 <= curr <= ob["high"]:
            age = len(c) - ob["idx"]
            score = 2.0 if age < 10 else (1.5 if age < 20 else 1.0)
            if score > best:
                best = score; best_side = "SHORT"; sl_level = ob["high"]

    return best, best_side, sl_level

def _find_fvg(h: np.ndarray, l: np.ndarray, lookback: int=20) -> list[dict]:
    """Fair Value Gap = فجوة بين wick[i] وwick[i+2]"""
    gaps = []
    for i in range(len(h)-lookback, len(h)-2):
        if i < 0: continue
        # Bullish FVG: high[i] < low[i+2]
        if h[i] < l[i+2]:
            gaps.append({"type":"bullish","top":float(l[i+2]),"bottom":float(h[i]),"idx":i})
        # Bearish FVG: low[i] > high[i+2]
        elif l[i] > h[i+2]:
            gaps.append({"type":"bearish","top":float(l[i]),"bottom":float(h[i+2]),"idx":i})
    return gaps

def _ict_fvg_score(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> tuple[float, str]:
    """هل السعر في منطقة FVG غير مملوءة؟"""
    if len(c) < 10: return 0.0, "none"
    gaps = _find_fvg(h, l)
    curr = float(c[-1])
    for gap in gaps:
        if gap["type"] == "bullish" and gap["bottom"] <= curr <= gap["top"]:
            return 1.0, "LONG"
        if gap["type"] == "bearish" and gap["bottom"] <= curr <= gap["top"]:
            return 1.0, "SHORT"
    return 0.0, "none"

def _ict_liquidity_sweep_score(h: np.ndarray, l: np.ndarray,
                                c: np.ndarray) -> tuple[float, str]:
    """
    Liquidity Sweep: السعر يتجاوز قمة/قاع سابقة ثم يعود بسرعة
    = smart money يجمع السيولة قبل الحركة الحقيقية
    """
    if len(c) < 20: return 0.0, "none"
    window = 20
    prev_h = float(np.max(h[-window-3:-3]))
    prev_l = float(np.min(l[-window-3:-3]))

    # تجاوز القمة ثم إغلاق تحتها (Bearish sweep → LONG setup)
    if float(h[-2]) > prev_h and float(c[-1]) < prev_h:
        return 1.5, "LONG"
    # تجاوز القاع ثم إغلاق فوقه (Bullish sweep → SHORT setup)
    if float(l[-2]) < prev_l and float(c[-1]) > prev_l:
        return 1.5, "SHORT"
    return 0.0, "none"

def _ict_bos_score(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> tuple[float, str]:
    """
    BOS = Break of Structure: اختراق آخر Swing High/Low
    CHoCH = Change of Character: أول كسر معاكس بعد ترند طويل
    """
    if len(c) < 30: return 0.0, "none"
    sh = _swing_highs(h, lb=5)
    sl_ = _swing_lows(l, lb=5)
    curr = float(c[-1]); prev = float(c[-2])
    score = 0.0; side = "none"

    if sh:
        last_sh = float(h[sh[-1]])
        if prev < last_sh and curr > last_sh:
            score = 1.5; side = "LONG"  # BOS صاعد

    if sl_:
        last_sl = float(l[sl_[-1]])
        if prev > last_sl and curr < last_sl:
            score = 1.5; side = "SHORT"  # BOS هابط

    return score, side


# ═══════════════════════════════════════════════════════════════
# 4. Wyckoff
# ═══════════════════════════════════════════════════════════════

def _wyckoff_accumulation_score(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                                 v: np.ndarray) -> float:
    """
    يكشف مرحلة Wyckoff Accumulation:
    Phase A: Selling Climax (SC) — حجم ضخم + شمعة هابطة كبيرة
    Phase B: Secondary Test — حجم أقل + السعر يختبر قاع SC
    Phase C: Spring — يكسر قاع B ثم يرتد بقوة
    الدخول المثالي: أول Retest بعد Sign of Strength
    """
    if len(c) < 60 or np.mean(v) == 0: return 0.0
    score = 0.0
    n = len(c)

    # Phase A: Selling Climax — أكبر شمعة هابطة في آخر 60 شمعة مع حجم ضخم
    bodies   = [abs(c[i]-c[i-1]) for i in range(n-60, n)]
    avg_body = float(np.mean(bodies)) if bodies else 0
    avg_vol  = float(np.mean(v[n-60:n]))

    for i in range(n-55, n-30):
        if (c[i] < c[i-1] and                    # شمعة هابطة
                abs(c[i]-c[i-1]) > avg_body*2.0 and  # أكبر من الوسط بمرتين
                v[i] > avg_vol * 2.0):               # حجم ضخم
            sc_low = float(l[i])
            # Phase B: Secondary Test — السعر يعود لاختبار قاع SC
            for j in range(i+3, min(i+25, n-5)):
                if l[j] <= sc_low * 1.02 and v[j] < v[i] * 0.7:
                    # Phase C: Spring — كسر مؤقت ثم ارتداد
                    for k in range(j+1, min(j+10, n-3)):
                        if l[k] < sc_low and c[k] > sc_low:  # Spring!
                            # Phase D: Sign of Strength في الشمعة الأخيرة
                            if c[-1] > c[-2] and v[-1] > avg_vol * 1.3:
                                score = 2.5
                                break
                    if score > 0: break
            if score > 0: break

    return score

def _wyckoff_distribution_score(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                                  v: np.ndarray) -> float:
    """Wyckoff Distribution (عكس Accumulation) — إشارة بيع"""
    if len(c) < 60 or np.mean(v) == 0: return 0.0
    score = 0.0; n = len(c)
    avg_body = float(np.mean([abs(c[i]-c[i-1]) for i in range(n-60, n)]))
    avg_vol  = float(np.mean(v[n-60:n]))

    for i in range(n-55, n-30):
        if (c[i] > c[i-1] and abs(c[i]-c[i-1]) > avg_body*2.0 and v[i] > avg_vol*2.0):
            bc_high = float(h[i])
            for j in range(i+3, min(i+25, n-5)):
                if h[j] >= bc_high * 0.98 and v[j] < v[i]*0.7:
                    for k in range(j+1, min(j+10, n-3)):
                        if h[k] > bc_high and c[k] < bc_high:  # Upthrust!
                            if c[-1] < c[-2] and v[-1] > avg_vol*1.3:
                                score = 2.5; break
                    if score > 0: break
            if score > 0: break
    return score


# ═══════════════════════════════════════════════════════════════
# 5. الشموع اليابانية (محسّنة بشرط السياق)
# ═══════════════════════════════════════════════════════════════

def _candle_score(o: np.ndarray, h: np.ndarray, l: np.ndarray,
                  c: np.ndarray, v: np.ndarray,
                  near_support: bool, near_resistance: bool) -> tuple[float, str]:
    """
    يُحلل آخر 3 شموع مع مراعاة السياق (S/R + حجم)
    """
    if len(c) < 4: return 0.0, "none"
    i   = len(c)-1
    scr = 0.0; side = "none"

    def _body_pct(idx):
        rng = h[idx]-l[idx]
        return abs(c[idx]-o[idx])/rng if rng>0 else 0

    def _lower_wick(idx):
        return min(o[idx],c[idx])-l[idx]

    def _upper_wick(idx):
        return h[idx]-max(o[idx],c[idx])

    body = abs(c[i]-o[i]); rng = h[i]-l[i]
    bp   = _body_pct(i); lw = _lower_wick(i); uw = _upper_wick(i)
    avg_v = float(np.mean(v[max(0,i-20):i])) if i>=20 else float(v[i])
    vol_conf = v[i] >= avg_v * 1.3 if avg_v > 0 else False

    # ── Bullish patterns ──────────────────────────────
    # 1. Bullish Engulfing
    if (i > 0 and c[i-1]<o[i-1] and c[i]>o[i] and
            c[i]>o[i-1] and o[i]<c[i-1] and bp>0.6):
        scr = 2.0 if (near_support or vol_conf) else 1.2
        side = "LONG"

    # 2. Hammer / Pin Bar Bull (بعد ترند هابط)
    elif lw >= rng*0.55 and uw <= rng*0.15 and bp < 0.35:
        scr = 1.8 if near_support else 1.0
        side = "LONG"

    # 3. Morning Star (3 شموع)
    elif (i >= 2 and c[i-2]<o[i-2] and             # شمعة هابطة كبيرة
              _body_pct(i-1) < 0.25 and              # دوجي/صغيرة
              c[i]>o[i] and c[i]>((c[i-2]+o[i-2])/2)):
        scr = 2.0 if near_support else 1.3
        side = "LONG"

    # 4. Bullish Marubozu (زخم نظيف)
    elif c[i]>o[i] and bp>0.85 and vol_conf:
        scr = 1.5; side = "LONG"

    # ── Bearish patterns ──────────────────────────────
    # 5. Bearish Engulfing
    elif (i > 0 and c[i-1]>o[i-1] and c[i]<o[i] and
              c[i]<o[i-1] and o[i]>c[i-1] and bp>0.6):
        scr = 2.0 if (near_resistance or vol_conf) else 1.2
        side = "SHORT"

    # 6. Shooting Star / Pin Bar Bear
    elif uw >= rng*0.55 and lw <= rng*0.15 and bp < 0.35:
        scr = 1.8 if near_resistance else 1.0
        side = "SHORT"

    # 7. Evening Star
    elif (i >= 2 and c[i-2]>o[i-2] and
              _body_pct(i-1) < 0.25 and
              c[i]<o[i] and c[i]<((c[i-2]+o[i-2])/2)):
        scr = 2.0 if near_resistance else 1.3
        side = "SHORT"

    # 8. Bearish Marubozu
    elif c[i]<o[i] and bp>0.85 and vol_conf:
        scr = 1.5; side = "SHORT"

    return scr, side


# ═══════════════════════════════════════════════════════════════
# 6. مؤشر الزخم المُركّب
# ═══════════════════════════════════════════════════════════════

def _momentum_score(rsi: np.ndarray, macd_hist: np.ndarray,
                    atr: np.ndarray, v: np.ndarray,
                    adx_v: float, pdi: float, mdi: float) -> tuple[float, str]:
    """
    يقيس جودة الزخم من 4 مصادر:
    RSI direction change + MACD cross + ATR expansion + Volume
    """
    if len(rsi) < 5: return 0.0, "none"
    scr = 0.0; side = "none"
    n   = len(rsi)

    rsi_v   = float(rsi[n-1]);    rsi_p = float(rsi[n-3])
    hist_v  = float(macd_hist[n-1]) if not math.isnan(macd_hist[n-1]) else 0
    hist_p  = float(macd_hist[n-3]) if not math.isnan(macd_hist[n-3]) else 0
    avg_vol = float(np.mean(v[max(0,n-20):n-1])) if n >= 20 else float(v[n-1])
    atr_exp = (float(atr[n-1]) > float(np.mean(atr[max(0,n-5):n-1])) * 1.1
               if n >= 5 else False)
    vol_spike = avg_vol > 0 and v[n-1] >= avg_vol * 1.35

    # Bullish momentum
    if rsi_v > rsi_p and rsi_v > 48:
        scr += 0.5
        if rsi_v < 70: scr += 0.3  # بدون ذروة شراء
    if hist_v > 0 and hist_v > hist_p: scr += 0.5
    if adx_v >= 22 and pdi > mdi:     scr += 0.5
    if atr_exp:                        scr += 0.3
    if vol_spike:                      scr += 0.2
    if scr >= 1.5: side = "LONG"

    # Bearish momentum (إعادة الحساب)
    scr_s = 0.0
    if rsi_v < rsi_p and rsi_v < 52:
        scr_s += 0.5
        if rsi_v > 30: scr_s += 0.3
    if hist_v < 0 and hist_v < hist_p: scr_s += 0.5
    if adx_v >= 22 and mdi > pdi:     scr_s += 0.5
    if atr_exp:                        scr_s += 0.3
    if vol_spike:                      scr_s += 0.2
    if scr_s >= 1.5 and scr_s > scr:
        scr = scr_s; side = "SHORT"

    return min(scr, 2.5), side


# ═══════════════════════════════════════════════════════════════
# 7. محرك الدرجات الرئيسي
# ═══════════════════════════════════════════════════════════════

def _aggregate_scores(
    # Classic
    classic_long: float, classic_short: float, classic_detail: str,
    # ICT
    ob_score: float, ob_side: str, ob_sl: Optional[float],
    fvg_score: float, fvg_side: str,
    sweep_score: float, sweep_side: str,
    bos_score: float, bos_side: str,
    # Wyckoff
    wyc_acc: float, wyc_dist: float,
    # Candles
    cdl_score: float, cdl_side: str,
    # Momentum
    mom_score: float, mom_side: str,
) -> dict:
    """
    يجمع الدرجات من كل المدارس ويُرجع التوصية النهائية
    """
    long_pts  = 0.0; short_pts = 0.0
    long_src: list[str] = []; short_src: list[str] = []

    # Classic (max 2.5)
    long_pts  += classic_long
    short_pts += classic_short
    if classic_long  > 0: long_src.append(f"CLASSIC+{classic_long:.1f}")
    if classic_short > 0: short_src.append(f"CLASSIC+{classic_short:.1f}")

    # ICT Order Block (max 2.0)
    if ob_side == "LONG":
        long_pts += ob_score; long_src.append(f"OB+{ob_score:.1f}")
    elif ob_side == "SHORT":
        short_pts += ob_score; short_src.append(f"OB+{ob_score:.1f}")

    # ICT FVG (max 1.0)
    if fvg_side == "LONG":
        long_pts += fvg_score; long_src.append("FVG")
    elif fvg_side == "SHORT":
        short_pts += fvg_score; short_src.append("FVG")

    # ICT Liquidity Sweep (max 1.5)
    if sweep_side == "LONG":
        long_pts += sweep_score; long_src.append("SWEEP")
    elif sweep_side == "SHORT":
        short_pts += sweep_score; short_src.append("SWEEP")

    # ICT BOS (max 1.5)
    if bos_side == "LONG":
        long_pts += bos_score; long_src.append("BOS")
    elif bos_side == "SHORT":
        short_pts += bos_score; short_src.append("BOS")

    # Wyckoff (max 2.5)
    if wyc_acc > 0:
        long_pts += wyc_acc; long_src.append(f"WYCKOFF_ACC+{wyc_acc:.1f}")
    if wyc_dist > 0:
        short_pts += wyc_dist; short_src.append(f"WYCKOFF_DIST+{wyc_dist:.1f}")

    # Candles (max 2.0)
    if cdl_side == "LONG":
        long_pts += cdl_score; long_src.append(f"CDL+{cdl_score:.1f}")
    elif cdl_side == "SHORT":
        short_pts += cdl_score; short_src.append(f"CDL+{cdl_score:.1f}")

    # Momentum (max 2.5)
    if mom_side == "LONG":
        long_pts += mom_score; long_src.append(f"MOM+{mom_score:.1f}")
    elif mom_side == "SHORT":
        short_pts += mom_score; short_src.append(f"MOM+{mom_score:.1f}")

    return {
        "long_score":  round(long_pts,  2),
        "short_score": round(short_pts, 2),
        "long_src":    long_src,
        "short_src":   short_src,
        "ob_sl":       ob_sl,
    }


# ═══════════════════════════════════════════════════════════════
# 8. الواجهة الرئيسية
# ═══════════════════════════════════════════════════════════════

def build_signal(
    symbol:  str,
    closes:  list[float],
    highs:   list[float],
    lows:    list[float],
    opens:   list[float],
    volumes: list[float],
    market:  str = "FUTURES",
) -> Optional[dict]:
    """
    يُحلل آخر شمعة مكتملة ويُرجع إشارة DEVEL_MASTER أو None.

    المُخرجات (dict):
        side      : "LONG" | "SHORT"
        entry     : سعر الدخول
        sl        : وقف الخسارة (خلف OB أو Swing + ATR buffer)
        tp        : هدف الربح
        score     : مجموع نقاط التقاطع
        ai_score  : درجة 0-100 للعرض
        sources   : مصادر التأكيد
        strategy  : "DEVEL_MASTER"
    """
    c = np.array(closes,  dtype=float)
    h = np.array(highs,   dtype=float)
    l = np.array(lows,    dtype=float)
    o = np.array(opens,   dtype=float)
    v = np.array(volumes, dtype=float)

    if len(c) < 200:
        return None

    # ── حساب المؤشرات ──────────────────────────────────────
    rsi_arr  = _rsi(c)
    atr_arr  = _atr(h, l, c)
    _, _, mh = _macd(c)
    adx_v, pdi, mdi = _adx(h, l, c)
    e20 = _ema(c, 20); e50 = _ema(c, 50)
    avg_vol = float(np.mean(v[-20:])) if len(v) >= 20 else float(np.mean(v))

    curr_atr = float(atr_arr[-1]) if atr_arr[-1] > 0 else float(c[-1]) * 0.008

    # ── EMA 200 Trend Alignment ─────────────────────────────
    e200 = _ema(c, 200)
    curr_e200 = float(e200[-1]) if not math.isnan(e200[-1]) else float(c[-1])
    above_e200 = float(c[-1]) > curr_e200
    below_e200 = float(c[-1]) < curr_e200

    # ── Volume Check ────────────────────────────────────────
    v_ratio = v[-1] / avg_vol if avg_vol > 0 else 1.0
    vol_strong = v_ratio >= 1.25

    # ── موقع السعر ──────────────────────────────────────────
    rh = float(np.max(h[-15:])); rl = float(np.min(l[-15:]))
    pos = (float(c[-1]) - rl) / (rh - rl) if rh != rl else 0.5
    near_sup = pos <= 0.30; near_res = pos >= 0.70

    # ── Classic ─────────────────────────────────────────────
    dn_len   = _count_downtrend_candles(h, l, c)
    up_len   = _count_uptrend_candles(h, l)
    cl, cs   = 0.0, 0.0
    cdetail  = ""

    # اختراق ترند هابط طويل (فقط إذا كسر السعر قمة الشمعة السابقة بزخم)
    if dn_len >= 10 and float(c[-1]) > float(h[-2]) and vol_strong:
        cl += (2.0 if dn_len >= 20 else 1.5)
        cdetail = f"DOWNTREND_BRK({dn_len}c)"

    # اختراق ترند صاعد طويل (بيع)
    if up_len >= 10 and float(c[-1]) < float(l[-2]) and vol_strong:
        cs += (2.0 if up_len >= 20 else 1.5)
        cdetail = f"UPTREND_BRK({up_len}c)"

    # Fibonacci (النسبة الذهبية 0.618 فقط)
    fib_s, fib_side = _fibonacci_score(h, l, c, v, avg_vol)
    if fib_side == "LONG" and vol_strong:  cl += fib_s
    if fib_side == "SHORT" and vol_strong: cs += fib_s

    # S/R Breakout + Retest
    sr_s, sr_side = _sr_breakout_retest_score(h, l, c, v, avg_vol)
    if sr_side == "LONG":  cl += sr_s
    if sr_side == "SHORT": cs += sr_s

    # Channel Breakout
    ch_s, ch_side = _channel_breakout_score(h, l, c, v, avg_vol)
    if ch_side == "LONG":  cl += ch_s
    if ch_side == "SHORT": cs += ch_s

    # Bull Flag
    bf_s, bf_side = _bull_flag_score(h, l, c, v)
    if bf_side == "LONG": cl += bf_s

    # ── ICT ─────────────────────────────────────────────────
    ob_s, ob_side, ob_sl = _ict_order_block_score(o, h, l, c)
    fvg_s, fvg_side      = _ict_fvg_score(h, l, c)
    sweep_s, sweep_side  = _ict_liquidity_sweep_score(h, l, c)
    bos_s, bos_side      = _ict_bos_score(h, l, c)

    # ── Wyckoff ─────────────────────────────────────────────
    wyc_acc  = _wyckoff_accumulation_score(h, l, c, v)
    wyc_dist = _wyckoff_distribution_score(h, l, c, v)

    # ── Candles (تقييد الشموع المنفردة: تقبل فقط إذا كان هناك تأكيد من مدرسة أخرى) ──
    cdl_s, cdl_side = _candle_score(o, h, l, c, v, near_sup, near_res)
    has_structure_long  = (cl > 0 or ob_side == "LONG" or wyc_acc > 0 or sr_side == "LONG")
    has_structure_short = (cs > 0 or ob_side == "SHORT" or wyc_dist > 0 or sr_side == "SHORT")

    if cdl_side == "LONG" and not has_structure_long:
        cdl_s = cdl_s * 0.4  # تخفيض وزن الشمعة إذا كانت منفردة
    if cdl_side == "SHORT" and not has_structure_short:
        cdl_s = cdl_s * 0.4

    # ── Momentum ────────────────────────────────────────────
    mom_s, mom_side = _momentum_score(rsi_arr, mh, atr_arr, v, adx_v, pdi, mdi)

    # ── تجميع الدرجات ───────────────────────────────────────
    agg = _aggregate_scores(
        cl, cs, cdetail,
        ob_s, ob_side, ob_sl,
        fvg_s, fvg_side,
        sweep_s, sweep_side,
        bos_s, bos_side,
        wyc_acc, wyc_dist,
        cdl_s, cdl_side,
        mom_s, mom_side,
    )

    ls  = agg["long_score"]
    ss  = agg["short_score"]
    gap = 0.8  # يجب أن يتقدم الاتجاه الفائز بهذا الهامش

    # ── فلتر بيئة السوق وتوافق حركة البيتكوين (BTC Correlation Shield) ──────
    from market_regime_shield import check_btc_correlation_guard

    if ls >= MIN_ENTRY_SCORE and ls > ss + gap:
        if above_e200 or ob_side == "LONG" or wyc_acc > 0:
            btc_pass, btc_reason = check_btc_correlation_guard(symbol, "LONG")
            if btc_pass:
                side = "LONG"; raw = ls; src = agg["long_src"]
                src.append("BTC_SHIELD_CONFIRMED")
            else:
                return None
        else:
            return None
    elif ss >= MIN_ENTRY_SCORE and ss > ls + gap:
        if below_e200 or ob_side == "SHORT" or wyc_dist > 0:
            btc_pass, btc_reason = check_btc_correlation_guard(symbol, "SHORT")
            if btc_pass:
                side = "SHORT"; raw = ss; src = agg["short_src"]
                src.append("BTC_SHIELD_CONFIRMED")
            else:
                return None
        else:
            return None
    else:
        return None

    # ── حساب SL و TP ────────────────────────────────────────
    entry = float(c[-1])
    rr    = RR_FUTURES if market.upper() == "FUTURES" else RR_SPOT

    # SL الذكي: خلف Order Block إن وُجد، وإلا خلف Swing + ATR buffer
    sh_list = _swing_highs(h, lb=3)
    sl_list = _swing_lows(l,  lb=3)

    if side == "LONG":
        # خيار 1: خلف OB
        if ob_sl is not None and ob_side == "LONG":
            sl_base = ob_sl
        # خيار 2: آخر Swing Low
        elif sl_list:
            sl_base = float(l[sl_list[-1]])
        else:
            sl_base = entry - curr_atr * 2
        sl = sl_base - curr_atr * SL_ATR_BUFFER
        tp = entry + (entry - sl) * rr

    else:  # SHORT
        if ob_sl is not None and ob_side == "SHORT":
            sl_base = ob_sl
        elif sh_list:
            sl_base = float(h[sh_list[-1]])
        else:
            sl_base = entry + curr_atr * 2
        sl = sl_base + curr_atr * SL_ATR_BUFFER
        tp = entry - (sl - entry) * rr

    risk_pct = abs(entry - sl) / entry * 100
    tp_pct   = abs(tp - entry) / entry * 100

    # ── ai_score (0-100) للعرض ──────────────────────────────
    # النقاط الكلية تتراوح نظرياً من 0 إلى ~14
    ai_score = min(100, int(raw / 10.0 * 100))

    return {
        "strategy":  "DEVEL_MASTER",
        "engine":    "DEVEL_MASTER",
        "symbol":    symbol,
        "side":      side,
        "market":    market,
        "entry":     round(entry, 8),
        "sl":        round(sl,    8),
        "tp":        round(tp,    8),
        "sl_pct":    round(risk_pct, 2),
        "tp_pct":    round(tp_pct,   2),
        "rr":        rr,
        "score":     round(raw, 2),
        "ai_score":  ai_score,
        "sources":   src,
        "adx":       round(adx_v, 1),
        "rsi":       round(float(rsi_arr[-1]), 1),
        "downtrend_candles": dn_len,
        "uptrend_candles":   up_len,
    }
