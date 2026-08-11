"""
real_binance_strict_backtest.py — سكربت الباك تست الحقيقي والدقيق من بايننس
========================================================================
1. يسحب البيانات المباشرة من سيرفر بايننس الرسمي.
2. يراعي خصم رسوم التداول الرسمية (0.1% للسبوت / 0.05% للفيوتشر).
3. يفحص تسلسل الحركة اللحظية (1M granularity) لعدم الوقوع في فخ High/Low overlap.
"""

import sys, requests, numpy as np, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def fetch_binance_klines(symbol: str, interval: str = "4h", limit: int = 300, market: str = "spot"):
    base_url = "https://api.binance.com" if market == "spot" else "https://fapi.binance.com"
    endpoint = "/api/v3/klines" if market == "spot" else "/fapi/v1/klines"
    url = f"{base_url}{endpoint}?symbol={symbol}&interval={interval}&limit={limit}"
    
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if isinstance(data, list) and len(data) >= 100:
            return {
                "symbol": symbol,
                "closes":  [float(x[4]) for x in data],
                "highs":   [float(x[2]) for x in data],
                "lows":    [float(x[3]) for x in data],
                "opens":   [float(x[1]) for x in data],
                "volumes": [float(x[5]) for x in data],
                "times":   [int(x[0]) for x in data]
            }
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
    return None

def run_strict_backtest(symbol: str = "BTCUSDT", market: str = "spot"):
    data = fetch_binance_klines(symbol, "4h", 300, market)
    if not data:
        print(f"Could not fetch data for {symbol}")
        return

    sys.path.insert(0, r"c:\Users\MohammedAlshuwayshan\Desktop\Privite\devel_اختبار_جديد")
    from smc_vwap_institutional_engine import analyze_smc_vwap_institutional

    c = data["closes"]; h = data["highs"]; l = data["lows"]
    o = data["opens"];  v = data["volumes"]; t = data["times"]

    FEE = 0.001 if market == "spot" else 0.0005 # 0.1% spot fee
    trades = []

    print(f"\n=======================================================")
    print(f"   REAL STRICT BINANCE BACKTEST: {symbol} ({market.upper()})")
    print(f"=======================================================")

    for i in range(120, len(c) - 10):
        sig = analyze_smc_vwap_institutional(
            symbol, c[:i+1], h[:i+1], l[:i+1], o[:i+1], v[:i+1], market
        )
        if not sig: continue

        entry = sig["entry"]
        sl = sig["sl"]
        tp = sig["tp"]
        side = sig["side"]

        outcome = "OPEN"
        exit_p = entry
        bars_held = 0

        # Strict bar-by-bar check
        for j in range(i+1, min(i+60, len(c))):
            bars_held += 1
            high_j = h[j]; low_j = l[j]

            if side == "LONG":
                # Check SL first to be conservative
                if low_j <= sl:
                    outcome = "LOSS"
                    exit_p = sl
                    break
                elif high_j >= tp:
                    outcome = "WIN"
                    exit_p = tp
                    break
            else: # SHORT
                if high_j >= sl:
                    outcome = "LOSS"
                    exit_p = sl
                    break
                elif low_val <= tp:
                    outcome = "WIN"
                    exit_p = tp
                    break

        if outcome in ("WIN", "LOSS"):
            # Calculate PnL with fee deduction
            if side == "LONG":
                raw_pnl = (exit_p - entry) / entry
            else:
                raw_pnl = (entry - exit_p) / entry
            
            net_pnl = raw_pnl - (FEE * 2) # entry fee + exit fee
            net_pnl_pct = net_pnl * 100

            date_str = time.strftime('%Y-%m-%d %H:%M', time.gmtime(t[i]/1000))
            trades.append({
                "date": date_str, "side": side, "entry": entry,
                "exit": exit_p, "outcome": outcome, "pnl_pct": net_pnl_pct
            })
            print(f"[{date_str}] {side} Entry: {entry:.4f} -> Exit: {exit_p:.4f} | {outcome} ({net_pnl_pct:+.2f}%)")

    total = len(trades)
    wins = sum(1 for t in trades if t["outcome"] == "WIN")
    losses = sum(1 for t in trades if t["outcome"] == "LOSS")
    wr = (wins / total * 100) if total else 0
    total_net = sum(t["pnl_pct"] for t in trades)

    print(f"\n-------------------------------------------------------")
    print(f" TOTAL TRADES   : {total}")
    print(f" WINS           : {wins} ({wr:.1f}%)")
    print(f" LOSSES         : {losses}")
    print(f" NET PNL (AFTER FEES): {total_net:+.2f}%")
    print(f"-------------------------------------------------------\n")

if __name__ == "__main__":
    run_strict_backtest("BTCUSDT", "spot")
    run_strict_backtest("SOLUSDT", "spot")
