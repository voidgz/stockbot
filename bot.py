import os
import time
import asyncio
import logging
from datetime import datetime
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

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WATCHLIST_FILE = "watchlists.json"
NEWS_STATE_FILE = "news_state.json"

AUTO_NEWS_INTERVAL = 900  # 15 menit

# =========================
# NEWS SOURCES (FROM EXCEL)
# =========================
NEWS_SOURCES = {
    "IDXChannel": {
        "domain": "idxchannel.com",
        "paths": ["/market", "/market-news", "/tag/corporate-action"],
    },
    "CNBC": {
        "domain": "cnbcindonesia.com",
        "paths": ["/market"],
    },
    "Kontan": {
        "domain": "kontan.co.id",
        "paths": ["/topik/corporate-action", "/investasi", "/insight"],
    },
    "InvestorID": {
        "domain": "investor.id",
        "paths": ["/corporate-action"],
    },
    "Bisnis": {
        "domain": "bisnis.com",
        "paths": ["/market", "/bursa-saham"],
    },
    "Liputan6": {
        "domain": "liputan6.com",
        "paths": ["/saham"],
    },
    "TradingView": {
        "domain": "tradingview.com",
        "paths": ["/news/markets/stocks"],
    },
    "BloombergTechnoz": {
        "domain": "bloombergtechnoz.com",
        "paths": ["/kanal/market"],
    },
    "KabarBursa": {
        "domain": "kabarbursa.com",
        "paths": ["/market-hari-ini"],
    },
    "MarketNewsID": {
        "domain": "marketnews.id",
        "paths": ["/kategori/corporate-action"],
    },
    "StockWatch": {
        "domain": "stockwatch.id",
        "paths": ["/category/market"],
    },
    "IDNSaham": {
        "domain": "idnsaham.com",
        "paths": ["/market/corporate-action"],
    },
}

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("idx-telegram-bot")

# =========================
# HELPERS (SAFE FOR pandas 2.2+)
# =========================
def safe_float(x):
    try:
        if isinstance(x, pd.Series):
            return float(x.iloc[0])
        return float(x)
    except Exception:
        return float("nan")

def safe_pct(x):
    try:
        return f"{safe_float(x):.2f}%"
    except Exception:
        return "n/a"

def google_news_rss(q):
    return f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=id&gl=ID&ceid=ID:id"

# =========================
# FETCH NEWS (PATH AWARE)
# =========================
def fetch_news(code):
    results = []
    for label, cfg in NEWS_SOURCES.items():
        query = f"{code} saham site:{cfg['domain']}"
        feed = feedparser.parse(google_news_rss(query))

        for e in feed.entries or []:
            title = getattr(e, "title", "")
            link = getattr(e, "link", "")
            if not title or not link:
                continue

            parsed = urlparse(link)
            if cfg["domain"] not in parsed.netloc:
                continue
            if not any(parsed.path.startswith(p) for p in cfg["paths"]):
                continue

            results.append((label, title, link))

    # dedup
    seen = set()
    uniq = []
    for r in results:
        if r[2] not in seen:
            seen.add(r[2])
            uniq.append(r)

    return uniq[:6]

# =========================
# ANALYSIS
# =========================
def analyze_stock(ticker):
    df = yf.download(ticker, period="9mo", auto_adjust=True, progress=False)
    if df.empty:
        return None

    df["MA20"] = ta.sma(df["Close"], 20)
    df["MA50"] = ta.sma(df["Close"], 50)
    df["RSI"] = ta.rsi(df["Close"], 14)

    last = df.iloc[-1]

    return (
        f"*{ticker}*\n"
        f"Close: {safe_float(last['Close']):.2f}\n"
        f"MA20 / MA50: {safe_float(last['MA20']):.2f} / {safe_float(last['MA50']):.2f}\n"
        f"RSI: {safe_float(last['RSI']):.2f}\n"
    )

# =========================
# COMMANDS
# =========================
async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Format: /analyze BBCA")
        return

    ticker = context.args[0].upper() + ".JK"
    result = analyze_stock(ticker)
    if not result:
        await update.message.reply_text("Data tidak tersedia.")
        return

    await update.message.reply_text(result, parse_mode=ParseMode.MARKDOWN)

# =========================
# AUTO NEWS JOB
# =========================
async def auto_news_job(context: ContextTypes.DEFAULT_TYPE):
    state = load_json(NEWS_STATE_FILE, {})
    watchlists = load_json(WATCHLIST_FILE, {}).get("chats", {})

    for chat_id, tickers in watchlists.items():
        state.setdefault(chat_id, {})
        for t in tickers:
            code = t.replace(".JK", "")
            state[chat_id].setdefault(code, [])

            for src, title, link in fetch_news(code):
                if link in state[chat_id][code]:
                    continue

                article = scrape_article(link)
                summary = summarize_article(
                    title=title,
                    source=src,
                    ticker=code,
                    url=link,
                    article_text=article.text if article else "",
                )

                msg = (
                    f"📰 *{code}*\n"
                    f"*{src}*\n\n"
                    f"{summary}\n\n"
                    f"{link}"
                )

                await context.bot.send_message(chat_id=int(chat_id), text=msg, parse_mode=ParseMode.MARKDOWN)
                state[chat_id][code].append(link)
                await asyncio.sleep(1)

    save_json(NEWS_STATE_FILE, state)

# =========================
# MAIN
# =========================
def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("analyze", cmd_analyze))

    app.job_queue.run_repeating(auto_news_job, interval=AUTO_NEWS_INTERVAL, first=60)

    app.run_polling()

if __name__ == "__main__":
    main()
