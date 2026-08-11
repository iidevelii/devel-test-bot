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
from telegram import Bot

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
    "INJUSDT","APTUSDT","ARBUSDT","SYNUSDT","HBARUSDT","KAITOUSDT","BANKUSDT",
    "COHRUSDT","1000BONKUSDT","SHELLUSDT","FETCHUSDT","STXUSDT","RUNEUSDT",
    "ICPUSDT","FTMUSDT","OPUSDT","TIAUSDT","MATICUSDT","ATOMUSDT","UNIUSDT",
    "RENDERUSDT","WIFUSDT","PEPEUSDT","FLOKIUSDT","ETCUSDT","DOGEUSDT",
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
EMOJI = {"LONG": "🟢", "SHORT": "🔴"}
MKT_EMOJI = {"futures": "📊", "spot": "💰"}

def format_signal(sig: dict) -> str:
    side  = sig["side"]
    mkt   = sig["market"].lower()
    entry = sig["entry"]
    sl    = sig["sl"]
    tp    = sig["tp"]
    rr    = sig["rr"]
    score = sig["score"]
    src   = ", ".join(sig.get("sources", [])[:5])
    rsi   = sig.get("rsi", 0)
    adx   = sig.get("adx", 0)
    sl_pct= abs(entry-sl)/entry*100
    tp_pct= abs(tp-entry)/entry*100
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return (
        f"{EMOJI[side]} <b>DEVEL MASTER — {side}</b>\n"
        f"{MKT_EMOJI[mkt]} {sig['symbol']} | {mkt.upper()}\n\n"
        f"📍 <b>Entry:</b> <code>{entry:.4f}</code>\n"
        f"🛡 <b>SL:</b> <code>{sl:.4f}</code> (-{sl_pct:.1f}%)\n"
        f"🎯 <b>TP:</b> <code>{tp:.4f}</code> (+{tp_pct:.1f}%)\n"
        f"⚖️ <b>R:R:</b> 1:{rr}\n\n"
        f"📊 <b>Score:</b> {score:.1f}/10\n"
        f"📈 RSI: {rsi:.0f} | ADX: {adx:.0f}\n"
        f"🔍 <b>Sources:</b> {src}\n\n"
        f"⏰ {now}\n"
        f"<i>Test signal — not financial advice</i>"
    )

# ── تتبع الإشارات (في الذاكرة) ──────────────────────────────
active_signals: dict = {}  # symbol → signal_data

def check_signal_outcomes():
    """يتحقق من نتائج الإشارات النشطة"""
    to_remove = []
    for sym, sig_data in active_signals.items():
        market = sig_data["market"].lower()
        data = fetch_klines(sym, "4h", 10, market)
        if data is None: continue
        curr = data["closes"][-1]
        high = max(data["highs"][-5:])
        low  = min(data["lows"][-5:])
        side = sig_data["side"]
        sl   = sig_data["sl"]
        tp   = sig_data["tp"]
        entry= sig_data["entry"]

        result = None
        if side == "LONG":
            if low <= sl:    result = "LOSS"
            elif high >= tp: result = "WIN"
        else:
            if high >= sl:   result = "LOSS"
            elif low <= tp:  result = "WIN"

        if result:
            sig_data["result"] = result
            sig_data["exit_price"] = curr
            to_remove.append(sym)

    return [(sym, active_signals.pop(sym)) for sym in to_remove]

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
                sig = analyze(symbol, market, tf)
                if sig is None: continue

                # لا ترسل إذا كانت إشارة نشطة لنفس الرمز
                key = f"{symbol}_{market}"
                if key in active_signals: continue

                msg = format_signal(sig)
                await bot.send_message(
                    chat_id=TEST_CHANNEL_ID,
                    text=msg,
                    parse_mode="HTML"
                )
                active_signals[key] = sig
                signals_sent += 1
                log.info(f"Signal sent: {symbol} {market} {sig['side']} score={sig['score']:.1f}")
                await asyncio.sleep(0.5)
            except Exception as e:
                log.error(f"Error {symbol}/{market}: {e}")
            await asyncio.sleep(0.2)

    log.info(f"Scan done. Signals sent: {signals_sent}")

    # تحقق من نتائج الإشارات السابقة
    closed = check_signal_outcomes()
    for sym, sig_data in closed:
        result = sig_data.get("result","OPEN")
        emoji  = "✅" if result=="WIN" else "❌"
        entry  = sig_data["entry"]
        exit_p = sig_data.get("exit_price",0)
        msg = (
            f"{emoji} **نتيجة {sym}**\n"
            f"الدخول: `{entry:.4f}` → الخروج: `{exit_p:.4f}`\n"
            f"النتيجة: **{result}**"
        )
        try:
            await bot.send_message(chat_id=TEST_CHANNEL_ID, text=msg, parse_mode="HTML")
        except: pass

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
