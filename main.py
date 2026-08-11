"""
devel_test_bot — بوت اختبار استراتيجية DEVEL_MASTER
=====================================================
مشروع Railway مستقل لاختبار الاستراتيجية في قناة تيليجرام منفصلة.

المميزات:
  - يمسح السوق كل 4 ساعات
  - يُرسل الإشارات لقناة اختبار منفصلة
  - يحسب Win Rate تلقائياً ويُقارن بالوقف
  - يعمل على Spot و Futures معاً
"""

import os, time, json, asyncio, logging, requests, numpy as np
from datetime import datetime, timezone
from telegram import Bot, InputMediaPhoto
from chart_generator import generate_chart, generate_outcome_chart, generate_near_alert_chart
from mtf_futures_engine import analyze_mtf_futures
from elite_spot_strategy import analyze_elite_spot
from smc_vwap_institutional_engine import analyze_smc_vwap_institutional
from futures_spot_dip_engine import analyze_dip_sweep






# ── إعدادات من .env ──────────────────────────────────────────
TELEGRAM_TOKEN    = os.getenv("TEST_BOT_TOKEN", "")       # توكن بوت الاختبار
TEST_CHANNEL_ID   = os.getenv("TEST_CHANNEL_ID", "")      # ID قناة الاختبار (مثال: -1001234567890)
SCAN_INTERVAL_H   = int(os.getenv("SCAN_INTERVAL_H", "4"))
MIN_SCORE         = float(os.getenv("DM_MIN_SCORE", "5.0"))
RR_FUTURES        = float(os.getenv("DM_RR_FUTURES", "1.5"))
RR_SPOT           = float(os.getenv("DM_RR_SPOT", "1.3"))
ENABLE_FUTURES    = os.getenv("ENABLE_FUTURES", "true").lower() == "true"
ENABLE_SPOT       = os.getenv("ENABLE_SPOT", "true").lower() == "true"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("devel_test")

FAPI = "https://fapi.binance.com"
SPOT_API = "https://api.binance.com"

# ── قائمة العملات للاختبار ──────────────────────────────────
SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","XLMUSDT","NEARUSDT",
    "LDOUSDT","DOTUSDT","LINKUSDT","AVAXUSDT","LTCUSDT","TRXUSDT","SUIUSDT",
    "INJUSDT","APTUSDT","ARBUSDT","SYNUSDT","HBARUSDT","STXUSDT","RUNEUSDT",
    "ICPUSDT","FTMUSDT","OPUSDT","TIAUSDT","MATICUSDT","ATOMUSDT","UNIUSDT",
    "RENDERUSDT","WIFUSDT","PEPEUSDT","FLOKIUSDT","ETCUSDT","DOGEUSDT","SHIBUSDT",
    "FILUSDT","BCHUSDT","FETUSDT","GALAUSDT","SANDUSDT","AXSUSDT","AAVEUSDT",
    "MKRUSDT","COMPUSDT","KASUSDT"
]

# ── المؤشرات ─────────────────────────────────────────────────
def _ema(c, p):
    r = np.full(len(c), np.nan)
    if len(c) < p: return r
    r[p-1] = np.mean(c[:p]); k = 2/(p+1)
    for i in range(p, len(c)): r[i] = c[i]*k + r[i-1]*(1-k)
    return r

def _rsi(c, p=14):
    r = np.full(len(c), 50.0)
    if len(c) < p+1: return r
    d=np.diff(c); g=np.where(d>0,d,0.); lo=np.where(d<0,-d,0.)
    ag=np.mean(g[:p]); al=np.mean(lo[:p])
    for i in range(p, len(c)-1):
        ag=(ag*(p-1)+g[i])/p; al=(al*(p-1)+lo[i])/p
        r[i+1]=100-100/(1+ag/al) if al>0 else 100.
    return r

def _atr(h, l, c, p=14):
    n=len(c); tr=np.zeros(n)
    for i in range(1,n): tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    a=np.zeros(n)
    if n>p: a[p]=np.mean(tr[1:p+1])
    for i in range(p+1,n): a[i]=(a[i-1]*(p-1)+tr[i])/p
    return a

def _macd_hist(c):
    e12=_ema(c,12); e26=_ema(c,26)
    ml=np.where(np.isnan(e12)|np.isnan(e26),np.nan,e12-e26)
    sig=_ema(np.where(np.isnan(ml),0,ml),9)
    return ml-sig

def _adx(h,l,c,p=14):
    n=len(c)
    if n<p*2+5: return 0.,50.,50.
    pdm=np.zeros(n); mdm=np.zeros(n); tr=np.zeros(n)
    for i in range(1,n):
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        pdm[i]=up if up>dn and up>0 else 0
        mdm[i]=dn if dn>up and dn>0 else 0
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    sT=sum(tr[1:p+1]); sP=sum(pdm[1:p+1]); sM=sum(mdm[1:p+1])
    dx=[]; pdi=mdi=0.
    for i in range(p+1,n):
        sT=sT-sT/p+tr[i]; sP=sP-sP/p+pdm[i]; sM=sM-sM/p+mdm[i]
        pdi=(sP/sT)*100 if sT else 0; mdi=(sM/sT)*100 if sT else 0
        den=pdi+mdi; dx.append(abs(pdi-mdi)/den*100 if den else 0)
    adx=float(np.mean(dx[-p:])) if len(dx)>=p else 0.
    return adx,pdi,mdi

# ── جلب بيانات Binance ──────────────────────────────────────
def fetch_klines(symbol, tf="4h", limit=300, market="futures"):
    for attempt in range(3):
        try:
            base = FAPI if market=="futures" else SPOT_API
            ep   = "/fapi/v1/klines" if market=="futures" else "/api/v3/klines"
            r = requests.get(f"{base}{ep}?symbol={symbol}&interval={tf}&limit={limit}", timeout=15)
            d = r.json()
            if not isinstance(d, list) or len(d) < 100: return None
            return {
                "closes":  [float(x[4]) for x in d],
                "highs":   [float(x[2]) for x in d],
                "lows":    [float(x[3]) for x in d],
                "opens":   [float(x[1]) for x in d],
                "volumes": [float(x[5]) for x in d],
            }
        except:
            if attempt < 2: time.sleep(3)
    return None

# ── فحص صلاحية الرمز في السوق ───────────────────────────────
def is_valid_symbol(symbol, market):
    try:
        base = FAPI if market=="futures" else SPOT_API
        ep   = "/fapi/v1/exchangeInfo" if market=="futures" else "/api/v3/exchangeInfo"
        r = requests.get(f"{base}{ep}", timeout=15)
        info = r.json()
        syms = [s["symbol"] for s in info.get("symbols",[]) if s.get("status") in ("TRADING","OPEN")]
        return symbol in syms
    except:
        return True  # في حالة الخطأ افترض صالح

# ── DEVEL_MASTER Engine ──────────────────────────────────────
def analyze(symbol, market="futures", tf="4h"):
    """يُحلل الرمز ويُرجع إشارة أو None"""
    from devel_master_strategy import build_signal
    data = fetch_klines(symbol, tf, 400, market)
    if data is None: return None
    return build_signal(
        symbol,
        data["closes"], data["highs"], data["lows"],
        data["opens"],  data["volumes"],
        market.upper()
    )

# ── تنسيق رسالة الإشارة ──────────────────────────────────────
SOURCES_AR = {
    "CLASSIC":      "اختراق ترند كلاسيكي",
    "DOWNTREND_BRK":"كسر ترند هابط طويل",
    "UPTREND_BRK":  "كسر ترند صاعد طويل",
    "FIB":          "اختراق مستوى فيبوناتشي",
    "SR_RETEST":    "إعادة اختبار دعم/مقاومة",
    "OB":           "منطقة Order Block (ICT)",
    "FVG":          "Fair Value Gap (ICT)",
    "SWEEP":        "اصطياد سيولة Liquidity Sweep",
    "BOS":          "كسر هيكل السوق BOS",
    "WYCKOFF_ACC":  "تراكم Wyckoff Accumulation",
    "WYCKOFF_DIST": "توزيع Wyckoff Distribution",
    "CDL":          "نموذج شمعة ياباني",
    "MOM":          "زخم مؤكد Momentum",
}

def sources_to_arabic(sources: list) -> str:
    lines = []
    for src in sources:
        key = src.split("+")[0].upper()
        ar = next((v for k,v in SOURCES_AR.items() if k in key), src)
        lines.append(f"  • {ar}")
    return "\n".join(lines) if lines else "  • DEVEL_MASTER"

def format_caption(sig: dict) -> str:
    side   = sig["side"]
    mkt    = sig["market"].lower()
    entry  = sig["entry"]
    sl     = sig["sl"]
    tp     = sig["tp"]
    rr     = sig["rr"]
    score  = sig["score"]
    sources= sig.get("sources", [])
    rsi    = sig.get("rsi", 0)
    adx    = sig.get("adx", 0)
    sl_pct = abs(entry-sl)/entry*100
    tp_pct = abs(tp-entry)/entry*100
    now    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    side_ar = "شراء" if side=="LONG" else "بيع"
    emo     = "🟢" if side=="LONG" else "🔴"
    mkt_emo = "📊" if mkt=="futures" else "💰"

    return (
        f"{emo} <b>DEVEL_MASTER — {side_ar} ({side})</b>\n"
        f"{mkt_emo} <b>{sig['symbol']}</b> | {mkt.upper()} | 4H\n"
        f"{'─'*30}\n"
        f"📍 <b>الدخول:</b> <code>{entry:.6f}</code>\n"
        f"🛡 <b>وقف الخسارة:</b> <code>{sl:.6f}</code> <i>(-{sl_pct:.2f}%)</i>\n"
        f"🎯 <b>الهدف:</b> <code>{tp:.6f}</code> <i>(+{tp_pct:.2f}%)</i>\n"
        f"⚖️ <b>نسبة R:R:</b> 1:{rr}\n"
        f"{'─'*30}\n"
        f"📊 <b>قوة الإشارة:</b> {score:.1f}/10\n"
        f"📈 <b>RSI:</b> {rsi:.0f}   |   <b>ADX:</b> {adx:.0f}\n"
        f"{'─'*30}\n"
        f"🔍 <b>أسباب الدخول:</b>\n"
        f"{sources_to_arabic(sources)}\n"
        f"{'─'*30}\n"
        f"⏰ {now}\n"
        f"<i>⚠️ إشارة اختبار — ليست توصية استثمارية</i>"
    )

# ── كاشف الانفجارات السعرية والفوليوم (Pump & Dump Alert) ───────
def detect_pump_dump(symbol: str, data: dict, market: str) -> Optional[dict]:
    closes  = data["closes"]
    highs   = data["highs"]
    lows    = data["lows"]
    volumes = data["volumes"]
    if len(closes) < 30: return None

    curr_c = closes[-1]
    prev_c = closes[-2]
    curr_v = volumes[-1]
    avg_v  = float(np.mean(volumes[-21:-1])) if len(volumes) >= 21 else float(np.mean(volumes[:-1]))

    v_ratio = curr_v / avg_v if avg_v > 0 else 1.0
    price_change_pct = (curr_c - prev_c) / prev_c * 100

    # انفجار صاعد (PUMP): ارتفاع > 2.0% مع حجم تداول >= 2.5x
    if price_change_pct >= 2.0 and v_ratio >= 2.5:
        return {
            "type": "PUMP",
            "symbol": symbol,
            "market": market,
            "price": curr_c,
            "change_pct": round(price_change_pct, 2),
            "vol_ratio": round(v_ratio, 1),
            "volume": curr_v,
        }

    # هبوط حاد (DUMP): هبوط <= -2.0% مع حجم تداول >= 2.5x
    if price_change_pct <= -2.0 and v_ratio >= 2.5:
        return {
            "type": "DUMP",
            "symbol": symbol,
            "market": market,
            "price": curr_c,
            "change_pct": round(price_change_pct, 2),
            "vol_ratio": round(v_ratio, 1),
            "volume": curr_v,
        }

    return None

def format_pump_caption(alert: dict) -> str:
    typ   = alert["type"]
    sym   = alert["symbol"]
    mkt   = alert["market"].upper()
    price = alert["price"]
    chg   = alert["change_pct"]
    v_rat = alert["vol_ratio"]
    emo   = "🚨🚀 <b>انفجار صاعد (PUMP ALERT)</b>" if typ == "PUMP" else "⚠️📉 <b>هبوط حاد (DUMP ALERT)</b>"
    chg_str = f"+{chg}%" if chg > 0 else f"{chg}%"
    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return (
        f"{emo}\n"
        f"📊 <b>{sym}</b> | {mkt} | 4H/15M\n"
        f"{'─'*30}\n"
        f"💰 <b>السعر الحالي:</b> <code>{price:.6f}</code>\n"
        f"📈 <b>تغير السعر:</b> <code>{chg_str}</code>\n"
        f"⚡️ <b>تضاعف الفوليوم:</b> <code>{v_rat}x</code> فوق المتوسط\n"
        f"{'─'*30}\n"
        f"🔍 <b>التحليل:</b> رصد دخول سيولة ضخمة وزخم حاد في السوق\n"
        f"⏰ {now}"
    )

# ── تتبع وحفظ الإشارات في ملف دائِم (Persistent Signal Tracking) ────
SIGNALS_FILE = "signals_history.json"

def load_signals_history() -> dict:
    if os.path.exists(SIGNALS_FILE):
        try:
            with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_signals_history(signals_dict: dict):
    try:
        with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
            json.dump(signals_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Error saving signals history: {e}")

active_signals: dict = load_signals_history()

def check_signal_outcomes() -> tuple[list, list]:
    """
    يتحقق من كافة الإشارات السابقة:
    1. الإشارات التي أغلقت (WIN / LOSS).
    2. الإشارات التي اقتربت جداً من الهدف أو الستوب (بفارق 0.30% أو أقل).
    """
    global active_signals
    closed_events = []
    near_alerts   = []

    for key, sig_data in list(active_signals.items()):
        if sig_data.get("status") in ("WIN", "LOSS", "BE"):
            continue

        if key.startswith("PUMP_"):
            continue

        sym    = sig_data.get("symbol")
        market = sig_data.get("market", "FUTURES").lower()
        side   = sig_data.get("side", "LONG")
        entry  = sig_data.get("entry", 0.0)
        sl     = sig_data.get("sl", 0.0)
        tp     = sig_data.get("tp", 0.0)

        if not sym or not entry or not sl or not tp:
            continue

        data = fetch_klines(sym, "5m", 15, market)
        if data is None:
            continue

        high_val = float(np.max(data["highs"]))
        low_val  = float(np.min(data["lows"]))
        curr_val = float(data["closes"][-1])

        status = None
        exit_price = curr_val

        # فحص الوصول التام للهدف أو الستوب
        if side == "LONG":
            if high_val >= tp:
                status = "WIN"; exit_price = tp
            elif low_val <= sl:
                status = "LOSS"; exit_price = sl
        else: # SHORT
            if low_val <= tp:
                status = "WIN"; exit_price = tp
            elif high_val >= sl:
                status = "LOSS"; exit_price = sl

        if status:
            pnl_pct = ((exit_price - entry) / entry * 100) if side == "LONG" else ((entry - exit_price) / entry * 100)
            sig_data["status"]     = status
            sig_data["exit_price"] = exit_price
            sig_data["pnl_pct"]    = round(pnl_pct, 2)
            sig_data["closed_at"]  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            closed_events.append(sig_data)
        else:
            # حساب المسافة المتبقية بالـ % للهدف والستوب
            if side == "LONG":
                rem_tp_pct = (tp - curr_val) / curr_val * 100
                rem_sl_pct = (curr_val - sl) / curr_val * 100
            else:
                rem_tp_pct = (curr_val - tp) / curr_val * 100
                rem_sl_pct = (sl - curr_val) / curr_val * 100

            # فحص إذا كان المتبقي للهدف 0.30% أو أقل ولم يُرسل تنبيه مسبقاً
            if 0 < rem_tp_pct <= 0.30 and not sig_data.get("near_tp_sent"):
                sig_data["near_tp_sent"] = True
                sig_data["rem_tp_pct"]   = round(rem_tp_pct, 2)
                sig_data["rem_sl_pct"]   = round(rem_sl_pct, 2)
                sig_data["alert_type"]   = "NEAR_TP"
                near_alerts.append(sig_data)

            # فحص إذا كان المتبقي للستوب 0.30% أو أقل ولم يُرسل تنبيه مسبقاً
            elif 0 < rem_sl_pct <= 0.30 and not sig_data.get("near_sl_sent"):
                sig_data["near_sl_sent"] = True
                sig_data["rem_tp_pct"]   = round(rem_tp_pct, 2)
                sig_data["rem_sl_pct"]   = round(rem_sl_pct, 2)
                sig_data["alert_type"]   = "NEAR_SL"
                near_alerts.append(sig_data)

    if closed_events or near_alerts:
        save_signals_history(active_signals)

    return closed_events, near_alerts

def format_near_alert_message(sig_data: dict) -> str:
    sym        = sig_data["symbol"]
    side       = sig_data["side"]
    mkt        = sig_data.get("market", "FUTURES").upper()
    alert_type = sig_data["alert_type"]
    rem_tp     = sig_data.get("rem_tp_pct", 0.0)
    rem_sl     = sig_data.get("rem_sl_pct", 0.0)
    entry      = sig_data["entry"]
    tp         = sig_data["tp"]
    sl         = sig_data["sl"]
    now        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    side_ar = "شراء (LONG)" if side == "LONG" else "بيع (SHORT)"

    if alert_type == "NEAR_TP":
        hdr = "⏳🎯 <b>تنبيه اقتراب شديد من الهدف! (NEAR TARGET)</b> 🟢"
        rem_str = f"باقي فقط <b>+{rem_tp:.2f}%</b> للوصول للهدف 🎯"
    else:
        hdr = "⚠️🛑 <b>تحذير اقتراب من الستوب! (NEAR STOP LOSS)</b> 🔴"
        rem_str = f"باقي فقط <b>-{rem_sl:.2f}%</b> للوصول للستوب 🛡️"

    return (
        f"{hdr}\n"
        f"📊 <b>{sym}</b> | {mkt} | {side_ar}\n"
        f"{'─'*30}\n"
        f"📍 <b>سعر الدخول:</b> <code>{entry:.6f}</code>\n"
        f"🎯 <b>سعر الهدف:</b> <code>{tp:.6f}</code>\n"
        f"🛡️ <b>سعر الستوب:</b> <code>{sl:.6f}</code>\n"
        f"{'─'*30}\n"
        f"⚡️ <b>المسافة المتبقية:</b> {rem_str}\n"
        f"📊 المتبقي للهدف: <code>+{rem_tp:.2f}%</code> | المتبقي للستوب: <code>-{rem_sl:.2f}%</code>\n"
        f"{'─'*30}\n"
        f"⏰ {now}"
    )


def format_outcome_message(sig_data: dict) -> str:
    sym     = sig_data["symbol"]
    side    = sig_data["side"]
    mkt     = sig_data.get("market", "FUTURES").upper()
    status  = sig_data["status"]
    entry   = sig_data["entry"]
    exit_p  = sig_data["exit_price"]
    pnl_pct = sig_data.get("pnl_pct", 0.0)
    now     = sig_data.get("closed_at", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    if status == "WIN":
        emo = "🎯🎯 <b>تحقق الهدف بالكامل! (TARGET HIT)</b> 🟢"
        pnl_str = f"+{pnl_pct:.2f}%"
    else:
        emo = "🛡️🛑 <b>ضرب وقف الخسارة (STOP LOSS HIT)</b> 🔴"
        pnl_str = f"{pnl_pct:.2f}%"

    side_ar = "شراء (LONG)" if side == "LONG" else "بيع (SHORT)"

    return (
        f"{emo}\n"
        f"📊 <b>{sym}</b> | {mkt} | {side_ar}\n"
        f"{'─'*30}\n"
        f"📍 <b>سعر الدخول:</b> <code>{entry:.6f}</code>\n"
        f"🏁 <b>سعر الخروج:</b> <code>{exit_p:.6f}</code>\n"
        f"📈 <b>الربح / الخسارة (PnL):</b> <b>{pnl_str}</b>\n"
        f"{'─'*30}\n"
        f"⏰ {now}"
    )


# ── الماسح الرئيسي ──────────────────────────────────────────
async def run_scanner(bot: Bot):
    log.info("Starting DEVEL_MASTER scan...")
    signals_sent = 0
    tf = "4h"

    markets = []
    if ENABLE_FUTURES: markets.append("futures")
    if ENABLE_SPOT:    markets.append("spot")

    for symbol in SYMBOLS:
        for market in markets:
            try:
                # 1. فحص إشارات الفيوتشر الاحترافية بـ Multi-Timeframe (MTF Precision Futures)
                raw_1h = fetch_klines(symbol, "1h", 100, market)
                raw_5m = fetch_klines(symbol, "5m", 100, market)
                if raw_1h and raw_5m:
                    mtf_sig = analyze_mtf_futures(symbol, raw_1h, raw_5m)
                    if mtf_sig:
                        mkey = f"MTF_{symbol}_{market}_{mtf_sig['side']}"
                        if mkey not in active_signals:
                            chart_buf = None
                            try:
                                chart_buf = generate_chart(
                                    symbol,
                                    raw_5m["closes"], raw_5m["highs"],
                                    raw_5m["lows"],   raw_5m["opens"],
                                    raw_5m["volumes"],
                                    mtf_sig, lookback=80
                                )
                            except Exception as chart_err:
                                log.warning(f"Chart error MTF {symbol}: {chart_err}")

                            caption = format_caption(mtf_sig)
                            if chart_buf:
                                await bot.send_photo(chat_id=TEST_CHANNEL_ID, photo=chart_buf, caption=caption, parse_mode="HTML")
                            else:
                                await bot.send_message(chat_id=TEST_CHANNEL_ID, text=caption, parse_mode="HTML")

                            active_signals[mkey] = mtf_sig
                            signals_sent += 1
                            log.info(f"MTF Futures Signal sent: {symbol} {market} {mtf_sig['side']} RR={mtf_sig['rr']}")

                # 2. فحص إشارات DEVEL_MASTER (فريم 4H)
                sig = analyze(symbol, market, tf)
                if sig is not None:
                    key = f"{symbol}_{market}"
                    if key not in active_signals:
                        chart_buf = None
                        try:
                            raw_data = fetch_klines(symbol, "4h", 100, market)
                            if raw_data:
                                chart_buf = generate_chart(
                                    symbol,
                                    raw_data["closes"], raw_data["highs"],
                                    raw_data["lows"],   raw_data["opens"],
                                    raw_data["volumes"],
                                    sig
                                )
                        except Exception as chart_err:
                            log.warning(f"Chart error {symbol}: {chart_err}")

                        caption = format_caption(sig)
                        if chart_buf:
                            await bot.send_photo(chat_id=TEST_CHANNEL_ID, photo=chart_buf, caption=caption, parse_mode="HTML")
                        else:
                            await bot.send_message(chat_id=TEST_CHANNEL_ID, text=caption, parse_mode="HTML")

                        active_signals[key] = sig
                        save_signals_history(active_signals)
                        signals_sent += 1
                        log.info(f"Signal sent: {symbol} {market} {sig['side']} score={sig['score']:.1f}")

                # 2. فحص إشارات الانفجار والزخم المفاجئ (Pump & Dump)
                raw_data = fetch_klines(symbol, "15m", 40, market)
                if raw_data:
                    pump_alert = detect_pump_dump(symbol, raw_data, market)
                    if pump_alert:
                        pkey = f"PUMP_{symbol}_{market}_{pump_alert['type']}"
                        if pkey not in active_signals:
                            pump_caption = format_pump_caption(pump_alert)
                            await bot.send_message(chat_id=TEST_CHANNEL_ID, text=pump_caption, parse_mode="HTML")
                            active_signals[pkey] = pump_alert
                            save_signals_history(active_signals)
                            log.info(f"Pump/Dump Alert sent: {symbol} {market} {pump_alert['type']} ({pump_alert['change_pct']}%, {pump_alert['vol_ratio']}x vol)")

                # 3. فحص استراتيجية النخبة للتداول الفوري (ELITE SPOT BREAKOUT - 1D/4H High Volume)
                if market == "spot":
                    raw_spot_4h = fetch_klines(symbol, "4h", 120, "spot")
                    if raw_spot_4h:
                        elite_sig = analyze_elite_spot(
                            symbol,
                            raw_spot_4h["closes"], raw_spot_4h["highs"],
                            raw_spot_4h["lows"],   raw_spot_4h["opens"],
                            raw_spot_4h["volumes"]
                        )
                        if elite_sig:
                            es_key = f"ELITE_SPOT_{symbol}_LONG"
                            if es_key not in active_signals:
                                chart_buf = None
                                try:
                                    chart_buf = generate_chart(
                                        symbol,
                                        raw_spot_4h["closes"], raw_spot_4h["highs"],
                                        raw_spot_4h["lows"],   raw_spot_4h["opens"],
                                        raw_spot_4h["volumes"],
                                        elite_sig, lookback=80
                                    )
                                except Exception as chart_err:
                                    log.warning(f"Chart error Elite Spot {symbol}: {chart_err}")

                                caption = format_caption(elite_sig)
                                if chart_buf:
                                    await bot.send_photo(chat_id=TEST_CHANNEL_ID, photo=chart_buf, caption=caption, parse_mode="HTML")
                                else:
                                    await bot.send_message(chat_id=TEST_CHANNEL_ID, text=caption, parse_mode="HTML")

                                active_signals[es_key] = elite_sig
                                save_signals_history(active_signals)
                                signals_sent += 1
                                log.info(f"ELITE Spot Signal sent: {symbol} LONG score={elite_sig['score']}")

                # 4. فحص استراتيجية السيولة والمؤسسات (SMC-VWAP Institutional Engine)
                raw_4h = fetch_klines(symbol, "4h", 120, market)
                if raw_4h:
                    smc_sig = analyze_smc_vwap_institutional(
                        symbol,
                        raw_4h["closes"], raw_4h["highs"],
                        raw_4h["lows"],   raw_4h["opens"],
                        raw_4h["volumes"], market
                    )
                    if smc_sig:
                        smc_key = f"SMC_VWAP_{symbol}_{market}_{smc_sig['side']}"
                        if smc_key not in active_signals:
                            chart_buf = None
                            try:
                                chart_buf = generate_chart(
                                    symbol,
                                    raw_4h["closes"], raw_4h["highs"],
                                    raw_4h["lows"],   raw_4h["opens"],
                                    raw_4h["volumes"],
                                    smc_sig, lookback=80
                                )
                            except Exception as chart_err:
                                log.warning(f"Chart error SMC VWAP {symbol}: {chart_err}")

                            caption = format_caption(smc_sig)
                            if chart_buf:
                                await bot.send_photo(chat_id=TEST_CHANNEL_ID, photo=chart_buf, caption=caption, parse_mode="HTML")
                            else:
                                await bot.send_message(chat_id=TEST_CHANNEL_ID, text=caption, parse_mode="HTML")

                            active_signals[smc_key] = smc_sig
                            save_signals_history(active_signals)
                            signals_sent += 1
                            log.info(f"SMC-VWAP Institutional Signal sent: {symbol} {market} {smc_sig['side']} AI_Score={smc_sig['ai_score']}")

                # 5. فحص استراتيجية قنص التشبعات واصطياد السيولة (Extreme Dip & Sweep Engine - 4H & 1H)
                raw_4h = fetch_klines(symbol, "4h", 120, market)
                raw_1h_dip = fetch_klines(symbol, "1h", 120, market)
                for tf_data, tf_name in [(raw_4h, "4H"), (raw_1h_dip, "1H")]:
                    if tf_data:
                        dip_sig = analyze_dip_sweep(
                            symbol,
                            tf_data["closes"], tf_data["highs"],
                            tf_data["lows"],   tf_data["opens"],
                            tf_data["volumes"], market
                        )
                        if dip_sig:
                            dip_sig["tf"] = tf_name
                            dip_key = f"DIP_{symbol}_{market}_{tf_name}_{dip_sig['side']}"
                            if dip_key not in active_signals:
                                chart_buf = None
                                try:
                                    chart_buf = generate_chart(
                                        symbol,
                                        tf_data["closes"], tf_data["highs"],
                                        tf_data["lows"],   tf_data["opens"],
                                        tf_data["volumes"],
                                        dip_sig, lookback=80
                                    )
                                except Exception as chart_err:
                                    log.warning(f"Chart error Dip Sweep {symbol}: {chart_err}")

                                caption = format_caption(dip_sig)
                                if chart_buf:
                                    await bot.send_photo(chat_id=TEST_CHANNEL_ID, photo=chart_buf, caption=caption, parse_mode="HTML")
                                else:
                                    await bot.send_message(chat_id=TEST_CHANNEL_ID, text=caption, parse_mode="HTML")

                                active_signals[dip_key] = dip_sig
                                save_signals_history(active_signals)
                                signals_sent += 1
                                log.info(f"Extreme Dip & Sweep Signal sent: {symbol} {market} {tf_name} {dip_sig['side']} RSI={dip_sig['rsi']}")

                await asyncio.sleep(0.3)
            except Exception as e:
                log.error(f"Error {symbol}/{market}: {e}")
            await asyncio.sleep(0.2)

    log.info(f"Scan done. Signals sent: {signals_sent}")

    # ── فحص ومتابعة الصفقات السابقة وإرسال إشعارات الهدف والستوب بشارت ─────
    closed_signals, near_alerts = check_signal_outcomes()

    # 1. إشعارات قرب الهدف أو الستوب (0.30% أو أقل)
    for n_data in near_alerts:
        try:
            near_msg = format_near_alert_message(n_data)
            sym = n_data.get("symbol", "")
            mkt = n_data.get("market", "FUTURES").lower()

            near_chart_buf = None
            try:
                raw_data = fetch_klines(sym, "5m", 80, mkt)
                if raw_data:
                    near_chart_buf = generate_near_alert_chart(
                        sym,
                        raw_data["closes"], raw_data["highs"],
                        raw_data["lows"],   raw_data["opens"],
                        raw_data["volumes"],
                        n_data,
                        n_data.get("alert_type", "NEAR_TP"),
                        n_data.get("rem_tp_pct", 0.0),
                        n_data.get("rem_sl_pct", 0.0),
                        lookback=60
                    )
            except Exception as chart_err:
                log.warning(f"Near alert chart error {sym}: {chart_err}")

            if near_chart_buf:
                await bot.send_photo(chat_id=TEST_CHANNEL_ID, photo=near_chart_buf, caption=near_msg, parse_mode="HTML")
            else:
                await bot.send_message(chat_id=TEST_CHANNEL_ID, text=near_msg, parse_mode="HTML")

            log.info(f"Near Alert Sent: {sym} -> {n_data.get('alert_type')} (rem_tp={n_data.get('rem_tp_pct')}%, rem_sl={n_data.get('rem_sl_pct')}%)")
        except Exception as near_err:
            log.error(f"Error sending near alert: {near_err}")

    # 2. إشعارات الإغلاق التام (WIN / LOSS)
    for sig_data in closed_signals:
        try:
            outcome_msg = format_outcome_message(sig_data)
            sym = sig_data.get("symbol", "")
            mkt = sig_data.get("market", "FUTURES").lower()

            outcome_chart_buf = None
            try:
                raw_data = fetch_klines(sym, "5m", 80, mkt)
                if raw_data:
                    outcome_chart_buf = generate_outcome_chart(
                        sym,
                        raw_data["closes"], raw_data["highs"],
                        raw_data["lows"],   raw_data["opens"],
                        raw_data["volumes"],
                        sig_data, lookback=60
                    )
            except Exception as chart_err:
                log.warning(f"Outcome chart error {sym}: {chart_err}")

            if outcome_chart_buf:
                await bot.send_photo(chat_id=TEST_CHANNEL_ID, photo=outcome_chart_buf, caption=outcome_msg, parse_mode="HTML")
            else:
                await bot.send_message(chat_id=TEST_CHANNEL_ID, text=outcome_msg, parse_mode="HTML")

            log.info(f"Outcome Notification Sent: {sym} -> {sig_data.get('status')} ({sig_data.get('pnl_pct')}%)")
        except Exception as notify_err:
            log.error(f"Error sending outcome notification: {notify_err}")

    return signals_sent




# ── إرسال ملخص يومي ──────────────────────────────────────────
async def send_summary(bot: Bot, scan_count: int, total_sigs: int):
    closed_count = len([s for s in active_signals.values() if s.get("result")])
    msg = (
        f"📊 **ملخص DEVEL_MASTER**\n\n"
        f"🔄 عمليات المسح: {scan_count}\n"
        f"📨 إشارات أُرسلت: {total_sigs}\n"
        f"⏳ إشارات نشطة: {len(active_signals)}\n\n"
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    await bot.send_message(chat_id=TEST_CHANNEL_ID, text=msg, parse_mode="Markdown")

# ── الحلقة الرئيسية ──────────────────────────────────────────
async def main():
    if not TELEGRAM_TOKEN:
        log.error("TEST_BOT_TOKEN is required!")
        return
    if not TEST_CHANNEL_ID:
        log.error("TEST_CHANNEL_ID is required!")
        return

    bot = Bot(token=TELEGRAM_TOKEN)
    scan_count  = 0
    total_sigs  = 0
    scan_interval = SCAN_INTERVAL_H * 3600

    await bot.send_message(
        chat_id=TEST_CHANNEL_ID,
        text=(
            "<b>DEVEL_MASTER Test Bot</b> started!\n\n"
            "Settings:\n"
            f"  TF: 4H\n"
            f"  Min Score: {MIN_SCORE}\n"
            f"  Futures RR: 1:{RR_FUTURES}\n"
            f"  Spot RR: 1:{RR_SPOT}\n"
            f"  Futures: {'on' if ENABLE_FUTURES else 'off'}\n"
            f"  Spot: {'on' if ENABLE_SPOT else 'off'}\n"
            f"  Symbols: {len(SYMBOLS)}\n"
            f"  Scan every: {SCAN_INTERVAL_H}h\n\n"
            "Waiting for first scan..."
        ),
        parse_mode="HTML"
    )

    while True:
        try:
            sigs = await run_scanner(bot)
            total_sigs += sigs
            scan_count  += 1
            if scan_count % 6 == 0:  # ملخص كل 24 ساعة (6 × 4h)
                await send_summary(bot, scan_count, total_sigs)
            await asyncio.sleep(scan_interval)
        except Exception as e:
            log.error(f"Main loop error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
