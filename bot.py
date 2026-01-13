import os
import time
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple
from urllib.parse import quote_plus, urlparse

import pandas as pd
import yfinance as yf
import pandas_ta as ta
import feedparser

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode, ChatType
from telegram.ext import Application, CommandHandler, ContextTypes

from storage import get_watchlist, set_watchlist, load_json, save_json
from news_scraper import scrape_article
from gemini_client import summarize_article


# =========================
# CONFIG
# =========================
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
WATCHLIST_FILE = os.getenv("WATCHLIST_FILE", "watchlists.json")
NEWS_STATE_FILE = os.getenv("NEWS_STATE_FILE", "news_state.json")

GROUP_ADMIN_ONLY = os.getenv("GROUP_ADMIN_ONLY", "true").lower() in ("1", "true", "yes")
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", "15"))

HIST_PERIOD = os.getenv("HIST_PERIOD", "9mo")
HIST_INTERVAL = os.getenv("HIST_INTERVAL", "1d")

AUTO_NEWS_ENABLED = os.getenv("AUTO_NEWS_ENABLED", "true").lower() in ("1", "true", "yes")
AUTO_NEWS_INTERVAL_SECONDS = int(os.getenv("AUTO_NEWS_INTERVAL_SECONDS", "900"))
AUTO_NEWS_PER_TICKER_MAX_NEW = int(os.getenv("AUTO_NEWS_PER_TICKER_MAX_NEW", "1"))
AUTO_NEWS_PER_CHAT_MAX_PER_RUN = int(os.getenv("AUTO_NEWS_PER_CHAT_MAX_PER_RUN", "5"))

QUIET_HOURS_WIB = os.getenv("QUIET_HOURS_WIB", "").strip()
NEWS_LIMIT_TOTAL = int(os.getenv("NEWS_LIMIT_TOTAL", "6"))


# =========================
# NEWS SOURCES (LENGKAP)
# =========================
NEWS_SOURCES = {
    "IDXChannel": {"domain": "idxchannel.com", "paths": ["/market", "/market-news", "/tag/corporate-action"]},
    "CNBC": {"domain": "cnbcindonesia.com", "paths": ["/market"]},
    "Kontan": {"domain": "kontan.co.id", "paths": ["/topik/corporate-action", "/investasi", "/insight"]},
    "InvestorID": {"domain": "investor.id", "paths": ["/corporate-action"]},
    "Bisnis": {"domain": "bisnis.com", "paths": ["/market", "/bursa-saham"]},
    "Liputan6": {"domain": "liputan6.com", "paths": ["/saham"]},
    "TradingView": {"domain": "tradingview.com", "paths": ["/news/markets/stocks"]},
    "BloombergTechnoz": {"domain": "bloombergtechnoz.com", "paths": ["/kanal/market"]},
    "KabarBursa": {"domain": "kabarbursa.com", "paths": ["/market-hari-ini"]},
    "MarketNewsID": {"domain": "marketnews.id", "paths": ["/kategori/corporate-action"]},
    "StockWatch": {"domain": "stockwatch.id", "paths": ["/category/market"]},
    "IDNSaham": {"domain": "idnsaham.com", "paths": ["/market/corporate-action"]},
    "FortuneIDN": {"domain": "fortuneidn.com", "paths": ["/tag/corporate-action"]},
    "Investing": {"domain": "id.investing.com", "paths": ["/news/stock-market-news"]},
    "SindoNews": {"domain": "sindonews.com", "paths": ["/topic/707/saham"]},
}


# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("idx-telegram-bot")


# =========================
# HELPERS
# =========================
_LAST_CALL: Dict[Tuple[int, str], float] = {}

def _rate_limit_ok(user_id: int, key: str) -> bool:
    now = time.time()
    last = _LAST_CALL.get((user_id, key), 0)
    if now - last < RATE_LIMIT_SECONDS:
        return False
    _LAST_CALL[(user_id, key)] = now
    return True

def _safe_float(x: Any) -> float:
    try:
        if isinstance(x, pd.Series):
            return float(x.iloc[0])
        return float(x)
    except Exception:
        return float("nan")

def _fmt_pct(x: Any) -> str:
    try:
        return f"{_safe_float(x):.2f}%"
    except Exception:
        return "n/a"

def _human_num(x: Any) -> str:
    x = _safe_float(x)
    ax = abs(x)
    if ax >= 1e12: return f"{x/1e12:.2f}T"
    if ax >= 1e9: return f"{x/1e9:.2f}B"
    if ax >= 1e6: return f"{x/1e6:.2f}M"
    if ax >= 1e3: return f"{x/1e3:.2f}K"
    return f"{x:.2f}"

def _google_news_rss(q: str) -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=id&gl=ID&ceid=ID:id"

def _normalize_idx_ticker(code: str) -> str:
    return code.upper() if code.upper().endswith(".JK") else f"{code.upper()}.JK"

def _code_only(ticker: str) -> str:
    return ticker.replace(".JK", "").upper()

def _within_quiet_hours_wib() -> bool:
    if not QUIET_HOURS_WIB:
        return True
    try:
        start_s, end_s = QUIET_HOURS_WIB.split("-")
        sh, sm = map(int, start_s.split(":"))
        eh, em = map(int, end_s.split(":"))
        now = datetime.now()
        return now.replace(hour=sh, minute=sm) <= now <= now.replace(hour=eh, minute=em)
    except Exception:
        return True


# =========================
# MARKET DATA
# =========================
def _fetch_price_history(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=HIST_PERIOD, interval=HIST_INTERVAL, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df.columns = df.columns.str.title()
    return df

def _compute_technical(df: pd.DataFrame) -> pd.DataFrame:
    df["MA20"] = ta.sma(df["Close"], 20)
    df["MA50"] = ta.sma(df["Close"], 50)
    df["RSI14"] = ta.rsi(df["Close"], 14)
    df["RET_1D"] = df["Close"].pct_change(1) * 100
    df["RET_5D"] = df["Close"].pct_change(5) * 100
    df["RET_20D"] = df["Close"].pct_change(20) * 100
    return df

def _technical_signal(last: pd.Series) -> str:
    for k in ("MA20", "MA50", "RSI14"):
        if pd.isna(last[k]):
            return "Data belum cukup untuk indikator."
    close, ma20, ma50, rsi = map(_safe_float, (last["Close"], last["MA20"], last["MA50"], last["RSI14"]))
    if close > ma20 > ma50 and rsi < 70:
        return "Bullish (trend naik, RSI sehat)."
    if close < ma20 < ma50 and rsi > 30:
        return "Bearish (trend turun)."
    if rsi >= 70:
        return "RSI overbought."
    if rsi <= 30:
        return "RSI oversold."
    return "Netral / konsolidasi."


# =========================
# ANALYZE FORMAT
# =========================
def _format_analyze(ticker: str, df: pd.DataFrame, f: Dict[str, Any]) -> str:
    last = df.iloc[-1]
    return (
        f"*{ticker}*\n"
        f"Close: {_safe_float(last['Close']):.2f}\n"
        f"Return 1D / 5D / 20D: {_fmt_pct(last['RET_1D'])} / {_fmt_pct(last['RET_5D'])} / {_fmt_pct(last['RET_20D'])}\n"
        f"MA20 / MA50: {_safe_float(last['MA20']):.2f} / {_safe_float(last['MA50']):.2f}\n"
        f"RSI14: {_safe_float(last['RSI14']):.2f}\n"
        f"Teknikal: {_technical_signal(last)}\n\n"
        f"Market Cap: {_human_num(f.get('MarketCap'))}\n"
        f"PER / PBV: {f.get('PER')} / {f.get('PBV')}\n"
        f"ROE / EPS: {f.get('ROE')} / {f.get('EPS')}\n"
        f"Div Yield: {f.get('DivYield')}\n"
        f"_Bukan rekomendasi investasi._"
    )


# =========================
# NEWS FETCH
# =========================
def _fetch_news_for_code(code: str) -> List[Tuple[str, str, str]]:
    results = []
    for label, cfg in NEWS_SOURCES.items():
        feed = feedparser.parse(_google_news_rss(f"{code} saham site:{cfg['domain']}"))
        for e in feed.entries or []:
            link = getattr(e, "link", "")
            title = getattr(e, "title", "")
            parsed = urlparse(link)
            if cfg["domain"] not in parsed.netloc:
                continue
            if not any(parsed.path.startswith(p) for p in cfg["paths"]):
                continue
            results.append((label, title, link))
    uniq, seen = [], set()
    for r in results:
        if r[2] not in seen:
            seen.add(r[2])
            uniq.append(r)
    return uniq[:NEWS_LIMIT_TOTAL]


# =========================
# AUTO NEWS JOB
# =========================
async def _job_auto_news(context: ContextTypes.DEFAULT_TYPE):
    if not AUTO_NEWS_ENABLED or not _within_quiet_hours_wib():
        return

    state = load_json(NEWS_STATE_FILE, {})
    watch_data = load_json(WATCHLIST_FILE, {"chats": {}})

    for chat_id_str, tickers in watch_data.get("chats", {}).items():
        chat_id = int(chat_id_str)
        state.setdefault(chat_id_str, {})
        sent = 0

        for tkr in tickers:
            code = _code_only(tkr)
            state[chat_id_str].setdefault(code, [])
            seen = set(state[chat_id_str][code])

            for src, title, link in _fetch_news_for_code(code):
                if link in seen:
                    continue

                art = scrape_article(link)
                summary = summarize_article(
                    title=art.title if art else title,
                    source=src,
                    ticker=code,
                    url=link,
                    article_text=art.text if art else "",
                )

                msg = (
                    f"📰 *News update {code}*\n"
                    f"*Sumber:* {src}\n"
                    f"*Judul:* {title}\n\n"
                    f"{summary}\n\n"
                    f"{link}"
                )

                await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode=ParseMode.MARKDOWN)
                state[chat_id_str][code].append(link)
                sent += 1
                await asyncio.sleep(1)

                if sent >= AUTO_NEWS_PER_CHAT_MAX_PER_RUN:
                    break

    save_json(NEWS_STATE_FILE, state)


# =========================
# COMMANDS
# =========================
async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Format: /analyze BBCA")
        return
    ticker = _normalize_idx_ticker(context.args[0])
    df = _fetch_price_history(ticker)
    if df.empty:
        await update.message.reply_text("Data tidak tersedia.")
        return
    df = _compute_technical(df)
    f = yf.Ticker(ticker).info or {}
    await update.message.reply_text(_format_analyze(ticker, df, f), parse_mode=ParseMode.MARKDOWN)

async def cmd_news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Format: /news BBCA")
        return
    code = _code_only(context.args[0])
    items = _fetch_news_for_code(code)
    if not items:
        await update.message.reply_text("Belum ada berita relevan.")
        return
    lines = [f"*Berita {code}*"]
    for src, title, link in items:
        lines.append(f"- [{src}] {title}\n{link}")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


# =========================
# MAIN
# =========================
def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN belum diset.")

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("news", cmd_news))

    if AUTO_NEWS_ENABLED:
        app.job_queue.run_repeating(_job_auto_news, interval=AUTO_NEWS_INTERVAL_SECONDS, first=60)

    app.run_polling()

if __name__ == "__main__":
    main()
