"""
market_regime_shield.py — محرك فلاتر بيئة السوق وحماية البيتكوين
===================================================================
1. BTC Correlation Shield:
   يفحص حركة البيتكوين (BTCUSDT) لمنع دخول صفقات عكس اتجاه البيتكوين الحاد.
   - إذا كان BTC بيهبط حاد -> يمنع صفقات LONG على العملات البديلة.
   - إذا كان BTC بيصعد حاد -> يمنع صفقات SHORT على العملات البديلة.

2. Adaptive Market Regime Filter:
   - سوق اتجاهي (Trending - ADX > 22): قبول إشارات الاختراق والاستمرار (Breakouts & OB Continuation).
   - سوق عرضي (Ranging - ADX <= 22): قبول إشارات ارتداد النطاق واصطياد السيولة فقط (Liquidity Sweeps).
"""

import requests
import numpy as np
import logging

log = logging.getLogger("market_regime")

_btc_cache = {"timestamp": 0, "trend": "NEUTRAL", "pct_change_15m": 0.0}

def get_btc_trend_shield() -> dict:
    """
    يُرجع حالة اتجاه البيتكوين مع كاش (Cache) لمدة دقيقة واحدة لتوفير الطلبات
    """
    import time
    now = time.time()
    if now - _btc_cache["timestamp"] < 60: # كاش 60 ثانية
        return _btc_cache

    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=15m&limit=30", timeout=5)
        d = r.json()
        if isinstance(d, list) and len(d) >= 20:
            closes = [float(x[4]) for x in d]
            curr_c = closes[-1]
            prev_c = closes[-4] # التغير في آخر ساعة (4 شمعات 15م)
            chg_pct = (curr_c - prev_c) / prev_c * 100

            trend = "NEUTRAL"
            if chg_pct <= -0.7:
                trend = "HEAVY_BEARISH"  # هبوط حاد في البيتكوين
            elif chg_pct >= +0.7:
                trend = "HEAVY_BULLISH"  # صعود حاد في البيتكوين
            elif chg_pct > 0.15:
                trend = "BULLISH"
            elif chg_pct < -0.15:
                trend = "BEARISH"

            _btc_cache["timestamp"] = now
            _btc_cache["trend"] = trend
            _btc_cache["pct_change_15m"] = round(chg_pct, 2)
            return _btc_cache
    except Exception as e:
        log.warning(f"BTC shield fetch warning: {e}")

    return {"trend": "NEUTRAL", "pct_change_15m": 0.0}


def check_btc_correlation_guard(symbol: str, side: str) -> tuple[bool, str]:
    """
    يفحص تماشي الصفقة مع اتجاه البيتكوين.
    إذا كانت العملة بديلة (ليست BTC) والبيتكوين يهبط حاداً -> يرفض LONG.
    """
    if "BTC" in symbol.upper():
        return True, "BTC_DIRECT"

    btc_data = get_btc_trend_shield()
    btc_trend = btc_data["trend"]

    # حظر LONG للعملات البديلة إذا كان BTC يهبط بقوة
    if side == "LONG" and btc_trend == "HEAVY_BEARISH":
        return False, f"BLOCKED: BTC dumping ({btc_data['pct_change_15m']}%)"

    # حظر SHORT للعملات البديلة إذا كان BTC يصعد بقوة
    if side == "SHORT" and btc_trend == "HEAVY_BULLISH":
        return False, f"BLOCKED: BTC pumping (+{btc_data['pct_change_15m']}%)"

    return True, f"PASSED (BTC {btc_trend})"
