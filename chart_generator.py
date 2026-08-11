"""
chart_generator.py — رسم الشارت مع تحليل DEVEL_MASTER
=======================================================
يرسم شارت Candlestick مع:
  - مناطق الدخول والخروج (Entry / SL / TP)
  - Order Blocks (ICT)
  - Fibonacci مستويات
  - خطوط الترند (Classic)
  - مستويات S/R
  - Fair Value Gaps
"""

import io
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

# ── ألوان ────────────────────────────────────────────────────
BG_COLOR     = "#0d1117"
GRID_COLOR   = "#1e2733"
UP_COLOR     = "#00e676"
DOWN_COLOR   = "#ff1744"
ENTRY_COLOR  = "#29b6f6"
SL_COLOR     = "#ff5252"
TP_COLOR     = "#69f0ae"
OB_BULL_CLR  = "rgba(0,230,118,0.15)"
OB_BEAR_CLR  = "rgba(255,23,68,0.15)"
FIB_COLOR    = "#ffd54f"
SR_COLOR     = "#ce93d8"
TREND_COLOR  = "#ff9800"
TEXT_COLOR   = "#e0e0e0"
FVG_BULL_CLR = "#00897b"
FVG_BEAR_CLR = "#c62828"


def _swing_highs(h, lb=5):
    return [i for i in range(lb, len(h)-lb)
            if h[i] == max(h[i-lb:i+lb+1])]

def _swing_lows(l, lb=5):
    return [i for i in range(lb, len(l)-lb)
            if l[i] == min(l[i-lb:i+lb+1])]

def _find_sr_levels(h, l, tol=0.005, min_touches=2):
    all_prices = list(h) + list(l)
    groups = []
    for price in all_prices:
        merged = False
        for g in groups:
            if abs(price - g["p"]) / g["p"] < tol:
                g["n"] += 1; merged = True; break
        if not merged:
            groups.append({"p": float(price), "n": 1})
    return [g["p"] for g in groups if g["n"] >= min_touches]


def generate_chart(
    symbol: str,
    closes: list, highs: list, lows: list, opens: list, volumes: list,
    signal: dict,
    lookback: int = 80,
) -> io.BytesIO:
    """
    يُرجع BytesIO يحتوي على صورة PNG للشارت.
    """
    # ── اختصار البيانات لآخر lookback شمعة ─────────────────
    n = min(lookback, len(closes))
    C = np.array(closes[-n:],  dtype=float)
    H = np.array(highs[-n:],   dtype=float)
    L = np.array(lows[-n:],    dtype=float)
    O = np.array(opens[-n:],   dtype=float)
    V = np.array(volumes[-n:], dtype=float)
    X = np.arange(len(C))

    entry  = signal["entry"]
    sl     = signal["sl"]
    tp     = signal["tp"]
    side   = signal["side"]
    score  = signal["score"]
    sources= signal.get("sources", [])
    market = signal.get("market", "FUTURES")

    # ── إعداد الشكل ─────────────────────────────────────────
    fig, (ax, ax_vol) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [4, 1]},
        facecolor=BG_COLOR
    )
    for a in (ax, ax_vol):
        a.set_facecolor(BG_COLOR)
        a.tick_params(colors=TEXT_COLOR, labelsize=8)
        a.spines[["top","right","left","bottom"]].set_color(GRID_COLOR)
        a.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.7)
        a.yaxis.tick_right()

    # ── رسم الشموع ──────────────────────────────────────────
    for i in X:
        color = UP_COLOR if C[i] >= O[i] else DOWN_COLOR
        # الذيل
        ax.plot([i, i], [L[i], H[i]], color=color, linewidth=0.8, alpha=0.9)
        # الجسم
        body_h = abs(C[i] - O[i])
        body_y = min(C[i], O[i])
        rect = plt.Rectangle(
            (i - 0.35, body_y), 0.7, max(body_h, (H[i]-L[i])*0.01),
            color=color, alpha=0.9
        )
        ax.add_patch(rect)

    # ── حجم التداول ─────────────────────────────────────────
    for i in X:
        color = UP_COLOR if C[i] >= O[i] else DOWN_COLOR
        ax_vol.bar(i, V[i], color=color, alpha=0.5, width=0.7)
    ax_vol.set_xlim(-1, len(X))
    ax_vol.set_ylabel("Vol", color=TEXT_COLOR, fontsize=7)

    # ── خطوط Entry / SL / TP ────────────────────────────────
    rr   = signal.get("rr", 1.3)
    risk = abs(entry - sl)
    gain = abs(tp - entry)

    # Entry
    ax.axhline(entry, color=ENTRY_COLOR, linewidth=1.5, linestyle="--", alpha=0.9)
    ax.text(len(X)+0.3, entry, f" ENTRY\n {entry:.4f}", color=ENTRY_COLOR,
            fontsize=8, va="center", fontweight="bold")

    # SL
    ax.axhline(sl, color=SL_COLOR, linewidth=1.2, linestyle=":", alpha=0.9)
    ax.text(len(X)+0.3, sl, f" SL\n {sl:.4f}", color=SL_COLOR,
            fontsize=8, va="center")

    # TP
    ax.axhline(tp, color=TP_COLOR, linewidth=1.2, linestyle=":", alpha=0.9)
    ax.text(len(X)+0.3, tp, f" TP\n {tp:.4f}", color=TP_COLOR,
            fontsize=8, va="center")

    # منطقة المخاطرة
    y_min = min(sl, tp); y_max = max(sl, tp)
    ax.axhspan(entry, sl, alpha=0.07, color=SL_COLOR)
    ax.axhspan(entry, tp, alpha=0.07, color=TP_COLOR)

    # سهم الاتجاه
    arrow_x = len(X) - 1
    if side == "LONG":
        ax.annotate("", xy=(arrow_x, tp), xytext=(arrow_x, entry),
                    arrowprops=dict(arrowstyle="->", color=TP_COLOR, lw=2))
    else:
        ax.annotate("", xy=(arrow_x, tp), xytext=(arrow_x, entry),
                    arrowprops=dict(arrowstyle="->", color=SL_COLOR, lw=2))

    # ── رسم بناءً على المصادر ───────────────────────────────
    src_str = " ".join(sources).upper()

    # 1. خط الترند الهابط (Classic)
    if any(x in src_str for x in ["DOWNTREND", "UPTREND", "CLASSIC"]):
        _draw_trendline(ax, H, L, C, X, side, sources)

    # 2. Fibonacci
    if "FIB" in src_str:
        _draw_fibonacci(ax, H, L, X)

    # 3. S/R Levels
    if "SR" in src_str or "RETEST" in src_str:
        _draw_sr_levels(ax, H, L, X)

    # 4. Order Block
    if "OB" in src_str:
        _draw_order_blocks(ax, O, H, L, C, X)

    # 5. FVG
    if "FVG" in src_str:
        _draw_fvg(ax, H, L, X)

    # 6. Liquidity Sweep
    if "SWEEP" in src_str:
        _draw_sweep(ax, H, L, X)

    # 7. BOS/CHoCH
    if "BOS" in src_str:
        _draw_bos(ax, H, L, C, X, side)

    # ── EMA سريعة ───────────────────────────────────────────
    def ema(c, p):
        r = np.full(len(c), np.nan)
        if len(c) < p: return r
        r[p-1] = np.mean(c[:p]); k = 2/(p+1)
        for i in range(p, len(c)): r[i] = c[i]*k + r[i-1]*(1-k)
        return r

    e20 = ema(C, 20); e50 = ema(C, 50)
    ax.plot(X, e20, color="#42a5f5", linewidth=1.0, alpha=0.7, label="EMA20")
    ax.plot(X, e50, color="#ef9a9a", linewidth=1.0, alpha=0.7, label="EMA50")

    # ── حدود المحاور ─────────────────────────────────────────
    price_range = max(H) - min(L)
    ax.set_ylim(min(L) - price_range*0.05, max(H) + price_range*0.15)
    ax.set_xlim(-1, len(X) + 5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    tf_display = str(signal.get("tf", "4H")).upper()

    # ── العنوان ─────────────────────────────────────────────
    side_label = "LONG" if side == "LONG" else "SHORT"
    sl_pct  = abs(entry-sl)/entry*100
    tp_pct  = abs(tp-entry)/entry*100
    src_display = " | ".join(sources[:4]) if sources else "DEVEL_MASTER"

    ax.set_title(
        f"DEVEL_MASTER  |  {symbol}  |  {market}  |  {tf_display}\n"
        f"[{side_label}]   Entry: {entry:.4f}   SL: -{sl_pct:.1f}%   TP: +{tp_pct:.1f}%   R:R 1:{rr}\n"
        f"Score: {score:.1f}/10  |  {src_display}",
        color=TEXT_COLOR, fontsize=9, pad=10,
        fontfamily="sans-serif",
        loc="left"
    )

    # ── Legend ───────────────────────────────────────────────
    legend_elements = [
        Line2D([0],[0], color=ENTRY_COLOR, lw=2, linestyle="--", label="Entry"),
        Line2D([0],[0], color=SL_COLOR,   lw=2, linestyle=":", label=f"SL (-{sl_pct:.1f}%)"),
        Line2D([0],[0], color=TP_COLOR,   lw=2, linestyle=":", label=f"TP (+{tp_pct:.1f}%)"),
        Line2D([0],[0], color="#42a5f5",  lw=1, label="EMA20"),
        Line2D([0],[0], color="#ef9a9a",  lw=1, label="EMA50"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              facecolor="#1e2733", edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=7)

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    # ── حفظ الصورة ──────────────────────────────────────────
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════
# دوال الرسم المتخصصة
# ══════════════════════════════════════════════════════════════

def _draw_trendline(ax, H, L, C, X, side, sources):
    """رسم خط الترند الهابط/الصاعد + نقطة الاختراق"""
    src_str = " ".join(sources).upper()
    lb = 5
    if side == "LONG" or "DOWNTREND" in src_str:
        # خط الترند الهابط: يربط قمم متناقصة
        sh = _swing_highs(H, lb)
        if len(sh) >= 2:
            x1, x2 = sh[-2], sh[-1]
            y1, y2  = H[x1], H[x2]
            # مد الخط للأمام
            slope = (y2-y1)/(x2-x1) if x2!=x1 else 0
            x_end = len(X)-1
            y_end = y2 + slope*(x_end-x2)
            ax.plot([x1, x_end], [y1, y_end],
                    color=TREND_COLOR, linewidth=1.5,
                    linestyle="--", alpha=0.85, label="Downtrend")
            # نقطة الاختراق
            ax.scatter([len(X)-2], [C[-2]],
                       color=UP_COLOR, s=80, zorder=5, marker="^",
                       label="Breakout")
            ax.text(len(X)-2, C[-2]*1.002, "BRK",
                    color=UP_COLOR, fontsize=7, ha="center")
    else:
        # خط ترند صاعد
        sl_ = _swing_lows(L, lb)
        if len(sl_) >= 2:
            x1, x2 = sl_[-2], sl_[-1]
            y1, y2  = L[x1], L[x2]
            slope = (y2-y1)/(x2-x1) if x2!=x1 else 0
            x_end = len(X)-1
            y_end = y2 + slope*(x_end-x2)
            ax.plot([x1, x_end], [y1, y_end],
                    color=TREND_COLOR, linewidth=1.5,
                    linestyle="--", alpha=0.85, label="Uptrend")
            ax.scatter([len(X)-2], [C[-2]],
                       color=DOWN_COLOR, s=80, zorder=5, marker="v",
                       label="Breakdown")
            ax.text(len(X)-2, C[-2]*0.998, "BRK",
                    color=DOWN_COLOR, fontsize=7, ha="center")


def _draw_fibonacci(ax, H, L, X):
    """رسم مستويات Fibonacci من أعلى قمة لأدنى قاع"""
    hh = float(np.max(H)); ll = float(np.min(L))
    rng = hh - ll
    if rng < 1e-10: return
    levels = {
        "0.236": ll + rng*0.236,
        "0.382": ll + rng*0.382,
        "0.500": ll + rng*0.500,
        "0.618": ll + rng*0.618,
        "0.786": ll + rng*0.786,
    }
    golden = {"0.618", "0.786"}
    for name, level in levels.items():
        lw    = 1.5 if name in golden else 0.8
        alpha = 0.9 if name in golden else 0.55
        ax.axhline(level, color=FIB_COLOR, linewidth=lw,
                   linestyle="-.", alpha=alpha)
        ax.text(-0.5, level, f"  Fib {name}", color=FIB_COLOR,
                fontsize=6.5, va="center", alpha=alpha)


def _draw_sr_levels(ax, H, L, X):
    """رسم مستويات Support/Resistance"""
    levels = _find_sr_levels(H, L, min_touches=2)
    curr = float(H[-1] + L[-1]) / 2
    for lv in levels:
        kind = "R" if lv > curr else "S"
        ax.axhline(lv, color=SR_COLOR, linewidth=0.9,
                   linestyle=":", alpha=0.7)
        ax.text(0.5, lv, f"  {kind} {lv:.4f}",
                color=SR_COLOR, fontsize=6, va="bottom", alpha=0.8)


def _draw_order_blocks(ax, O, H, L, C, X):
    """رسم Order Blocks كمستطيلات شفافة"""
    n = len(C)
    for i in range(max(1, n-25), n-3):
        if i >= n: break
        if C[i] < O[i]:  # شمعة هابطة
            up = all(C[i+k] > C[i+k-1] for k in range(1,4) if i+k < n)
            if up:
                rect = plt.Rectangle(
                    (i-0.4, L[i]), 0.8, H[i]-L[i],
                    facecolor=UP_COLOR, alpha=0.12, linewidth=0.8,
                    edgecolor=UP_COLOR
                )
                ax.add_patch(rect)
                ax.text(i, H[i], "  OB↑", color=UP_COLOR,
                        fontsize=6.5, va="bottom")
        elif C[i] > O[i]:  # شمعة صاعدة
            dn = all(C[i+k] < C[i+k-1] for k in range(1,4) if i+k < n)
            if dn:
                rect = plt.Rectangle(
                    (i-0.4, L[i]), 0.8, H[i]-L[i],
                    facecolor=DOWN_COLOR, alpha=0.12, linewidth=0.8,
                    edgecolor=DOWN_COLOR
                )
                ax.add_patch(rect)
                ax.text(i, L[i], "  OB↓", color=DOWN_COLOR,
                        fontsize=6.5, va="top")


def _draw_fvg(ax, H, L, X):
    """رسم Fair Value Gaps كمناطق ملوّنة"""
    n = len(H)
    for i in range(max(0, n-20), n-2):
        if i+2 >= n: break
        if H[i] < L[i+2]:  # Bullish FVG
            rect = plt.Rectangle(
                (i, H[i]), 2, L[i+2]-H[i],
                color=FVG_BULL_CLR, alpha=0.25, linewidth=0
            )
            ax.add_patch(rect)
            ax.text(i+1, (H[i]+L[i+2])/2, "FVG",
                    color="#80cbc4", fontsize=6, ha="center", va="center")
        elif L[i] > H[i+2]:  # Bearish FVG
            rect = plt.Rectangle(
                (i, H[i+2]), 2, L[i]-H[i+2],
                color=FVG_BEAR_CLR, alpha=0.25, linewidth=0
            )
            ax.add_patch(rect)
            ax.text(i+1, (H[i+2]+L[i])/2, "FVG",
                    color="#ef9a9a", fontsize=6, ha="center", va="center")


def _draw_sweep(ax, H, L, X):
    """رسم Liquidity Sweep"""
    if len(H) < 25: return
    ph = float(np.max(H[-23:-3]))
    pl = float(np.min(L[-23:-3]))
    ax.axhline(ph, color="#ffa726", linewidth=1.0, linestyle=":",
               alpha=0.8, label="Liquidity High")
    ax.axhline(pl, color="#ffa726", linewidth=1.0, linestyle=":",
               alpha=0.8, label="Liquidity Low")
    ax.text(0.5, ph, "  Liq High", color="#ffa726",
            fontsize=6.5, va="bottom")
    ax.text(0.5, pl, "  Liq Low",  color="#ffa726",
            fontsize=6.5, va="top")


def _draw_bos(ax, H, L, C, X, side):
    """رسم BOS (Break of Structure)"""
    sh = _swing_highs(H, 5); sl_ = _swing_lows(L, 5)
    if side == "LONG" and sh:
        lsh = float(H[sh[-1]])
        ax.axhline(lsh, color="#ce93d8", linewidth=1.2,
                   linestyle="-.", alpha=0.8)
        ax.text(sh[-1], lsh, "  BOS", color="#ce93d8",
                fontsize=7, va="bottom", fontweight="bold")
    elif sl_:
        lsl = float(L[sl_[-1]])
        ax.axhline(lsl, color="#ce93d8", linewidth=1.2,
                   linestyle="-.", alpha=0.8)
        ax.text(sl_[-1], lsl, "  BOS", color="#ce93d8",
                fontsize=7, va="top", fontweight="bold")


# ══════════════════════════════════════════════════════════════
# شارت نتيجة الصفقة (Outcome Chart مع تحديد شمعة الدخول والخروج)
# ══════════════════════════════════════════════════════════════

def generate_outcome_chart(
    symbol: str,
    closes: list, highs: list, lows: list, opens: list, volumes: list,
    sig_data: dict,
    lookback: int = 60
) -> io.BytesIO:
    """
    يرسم شارت يوضح مسار الصفقة من شمعة الدخول إلى شمعة الخروج (TP أو SL)
    مع رسم سهم الدخول وسهم الخروج ومسار الصفقة.
    """
    n = min(lookback, len(closes))
    C = np.array(closes[-n:], dtype=float)
    H = np.array(highs[-n:], dtype=float)
    L = np.array(lows[-n:], dtype=float)
    O = np.array(opens[-n:], dtype=float)
    V = np.array(volumes[-n:], dtype=float)
    X = np.arange(len(C))

    entry   = sig_data["entry"]
    sl      = sig_data["sl"]
    tp      = sig_data["tp"]
    exit_p  = sig_data.get("exit_price", C[-1])
    side    = sig_data["side"]
    status  = sig_data.get("status", "WIN")
    pnl_pct = sig_data.get("pnl_pct", 0.0)
    market  = sig_data.get("market", "FUTURES")

    # ── إعداد الشكل ─────────────────────────────────────────
    fig, (ax, ax_vol) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [4, 1]},
        facecolor=BG_COLOR
    )
    for a in (ax, ax_vol):
        a.set_facecolor(BG_COLOR)
        a.tick_params(colors=TEXT_COLOR, labelsize=8)
        a.spines[["top","right","left","bottom"]].set_color(GRID_COLOR)
        a.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.7)
        a.yaxis.tick_right()

    # ── رسم الشموع ──────────────────────────────────────────
    for i in X:
        color = UP_COLOR if C[i] >= O[i] else DOWN_COLOR
        ax.plot([i, i], [L[i], H[i]], color=color, linewidth=0.8, alpha=0.9)
        body_h = abs(C[i] - O[i])
        body_y = min(C[i], O[i])
        rect = plt.Rectangle(
            (i - 0.35, body_y), 0.7, max(body_h, (H[i]-L[i])*0.01),
            facecolor=color, alpha=0.9, edgecolor=color
        )
        ax.add_patch(rect)

    # ── حجم التداول ─────────────────────────────────────────
    for i in X:
        color = UP_COLOR if C[i] >= O[i] else DOWN_COLOR
        ax_vol.bar(i, V[i], color=color, alpha=0.5, width=0.7)
    ax_vol.set_xlim(-1, len(X))

    # ── خطوط Entry / SL / TP ────────────────────────────────
    ax.axhline(entry, color=ENTRY_COLOR, linewidth=1.5, linestyle="--", alpha=0.85)
    ax.axhline(sl,    color=SL_COLOR,    linewidth=1.2, linestyle=":",  alpha=0.85)
    ax.axhline(tp,    color=TP_COLOR,    linewidth=1.2, linestyle=":",  alpha=0.85)

    # ── تمييز شمعة الدخول وشمعة الخروج ─────────────────────
    entry_idx = max(0, len(X) - 15) # شمعة الدخول التقريبية
    exit_idx  = len(X) - 1           # شمعة الخروج الأحدث

    # علامة شمعة الدخول
    ax.scatter([entry_idx], [entry], color=ENTRY_COLOR, s=120, zorder=6, marker="o")
    ax.text(entry_idx, entry * (1.003 if side=="LONG" else 0.997), " ENTRY",
            color=ENTRY_COLOR, fontsize=8.5, fontweight="bold", ha="center")

    # علامة شمعة الخروج
    exit_color = TP_COLOR if status == "WIN" else SL_COLOR
    exit_marker = "^" if status == "WIN" else "v"
    exit_label = "🎯 TARGET HIT" if status == "WIN" else "🛑 SL HIT"

    ax.scatter([exit_idx], [exit_p], color=exit_color, s=150, zorder=7, marker=exit_marker)
    ax.text(exit_idx, exit_p * (1.004 if side=="LONG" else 0.996), f" {exit_label}",
            color=exit_color, fontsize=9, fontweight="bold", ha="right")

    # رسم المسار بين الدخول والخروج
    path_color = TP_COLOR if status == "WIN" else SL_COLOR
    ax.plot([entry_idx, exit_idx], [entry, exit_p],
            color=path_color, linewidth=2.5, linestyle="-", alpha=0.85)

    # ── حدود المحاور والعنوان ─────────────────────────────────
    price_range = max(H) - min(L)
    ax.set_ylim(min(L) - price_range*0.05, max(H) + price_range*0.15)
    ax.set_xlim(-1, len(X) + 3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    status_txt = f"TARGET HIT (+{pnl_pct:.2f}%)" if status == "WIN" else f"STOP LOSS HIT ({pnl_pct:.2f}%)"
    status_clr = TP_COLOR if status == "WIN" else SL_COLOR

    ax.set_title(
        f"DEVEL_MASTER OUTCOME  |  {symbol}  |  {market}  |  {side}\n"
        f"RESULT: {status_txt}   Entry: {entry:.4f} -> Exit: {exit_p:.4f}",
        color=status_clr, fontsize=10, pad=10,
        fontweight="bold", fontfamily="sans-serif",
        loc="left"
    )

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════
# شارت تنبيه الاقتراب من الهدف أو الستوب (Near TP / Near SL Chart)
# ══════════════════════════════════════════════════════════════

def generate_near_alert_chart(
    symbol: str,
    closes: list, highs: list, lows: list, opens: list, volumes: list,
    sig_data: dict,
    alert_type: str, # "NEAR_TP" أو "NEAR_SL"
    rem_tp_pct: float,
    rem_sl_pct: float,
    lookback: int = 60
) -> io.BytesIO:
    """
    يرسم شارت يوضح مدى قرب السعر الحالي من الهدف أو الستوب مع إظهار المتبقي بالـ %
    """
    n = min(lookback, len(closes))
    C = np.array(closes[-n:], dtype=float)
    H = np.array(highs[-n:], dtype=float)
    L = np.array(lows[-n:], dtype=float)
    O = np.array(opens[-n:], dtype=float)
    V = np.array(volumes[-n:], dtype=float)
    X = np.arange(len(C))

    entry  = sig_data["entry"]
    sl     = sig_data["sl"]
    tp     = sig_data["tp"]
    curr_p = C[-1]
    side   = sig_data["side"]
    market = sig_data.get("market", "FUTURES")

    fig, (ax, ax_vol) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [4, 1]},
        facecolor=BG_COLOR
    )
    for a in (ax, ax_vol):
        a.set_facecolor(BG_COLOR)
        a.tick_params(colors=TEXT_COLOR, labelsize=8)
        a.spines[["top","right","left","bottom"]].set_color(GRID_COLOR)
        a.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.7)
        a.yaxis.tick_right()

    # رسم الشموع
    for i in X:
        color = UP_COLOR if C[i] >= O[i] else DOWN_COLOR
        ax.plot([i, i], [L[i], H[i]], color=color, linewidth=0.8, alpha=0.9)
        body_h = abs(C[i] - O[i])
        body_y = min(C[i], O[i])
        rect = plt.Rectangle(
            (i - 0.35, body_y), 0.7, max(body_h, (H[i]-L[i])*0.01),
            facecolor=color, alpha=0.9, edgecolor=color
        )
        ax.add_patch(rect)

    for i in X:
        color = UP_COLOR if C[i] >= O[i] else DOWN_COLOR
        ax.bar(i, V[i], color=color, alpha=0.5, width=0.7)
    ax_vol.set_xlim(-1, len(X))

    # خطوط Entry / SL / TP
    ax.axhline(entry, color=ENTRY_COLOR, linewidth=1.5, linestyle="--", alpha=0.85)
    ax.axhline(sl,    color=SL_COLOR,    linewidth=1.2, linestyle=":",  alpha=0.85)
    ax.axhline(tp,    color=TP_COLOR,    linewidth=1.2, linestyle=":",  alpha=0.85)

    # خط السعر الحالي المميز
    curr_color = "#ffd54f" # أصفر مميز للسعر الحالي
    ax.axhline(curr_p, color=curr_color, linewidth=2.0, linestyle="-", alpha=0.9)

    last_idx = len(X) - 1
    ax.scatter([last_idx], [curr_p], color=curr_color, s=140, zorder=7, marker="o")

    # إضافة الملاحظة التوضيحية للمتبقي
    if alert_type == "NEAR_TP":
        hdr_txt = f"⏳ NEAR TARGET!  Only {rem_tp_pct:.2f}% to TP!"
        hdr_clr = TP_COLOR
        ax.annotate(f"  MEMBER ALERT: {rem_tp_pct:.2f}% to TP!",
                    xy=(last_idx, curr_p), xytext=(last_idx-15, curr_p + (tp-curr_p)*0.5),
                    arrowprops=dict(arrowstyle="->", color=TP_COLOR, lw=2),
                    color=TP_COLOR, fontsize=9.5, fontweight="bold")
    else:
        hdr_txt = f"⚠️ NEAR STOP LOSS!  Only {rem_sl_pct:.2f}% to SL!"
        hdr_clr = SL_COLOR
        ax.annotate(f"  WARNING: {rem_sl_pct:.2f}% to SL!",
                    xy=(last_idx, curr_p), xytext=(last_idx-15, curr_p - (curr_p-sl)*0.5),
                    arrowprops=dict(arrowstyle="->", color=SL_COLOR, lw=2),
                    color=SL_COLOR, fontsize=9.5, fontweight="bold")

    price_range = max(H) - min(L)
    ax.set_ylim(min(L) - price_range*0.05, max(H) + price_range*0.15)
    ax.set_xlim(-1, len(X) + 3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    ax.set_title(
        f"DEVEL_MASTER ALERT  |  {symbol}  |  {market}  |  {side}\n"
        f"{hdr_txt}   Current: {curr_p:.4f}   Target: {tp:.4f}   SL: {sl:.4f}",
        color=hdr_clr, fontsize=10, pad=10,
        fontweight="bold", fontfamily="sans-serif",
        loc="left"
    )

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


