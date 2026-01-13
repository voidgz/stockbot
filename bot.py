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
# Config
# =========================
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
WATCHLIST_FILE = os.getenv("WATCHLIST_FILE", "watchlists.json")
NEWS_STATE_FILE = os.getenv("NEWS_STATE_FILE", "news_state.json")

ALLOWED_GROUP_IDS_RAW = os.getenv("ALLOWED_GROUP_IDS", "").strip()
ALLOWED_GROUP_IDS: set[int] = set()
if ALLOWED_GROUP_IDS_RAW:
    for x in ALLOWED_GROUP_IDS_RAW.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            ALLOWED_GROUP_IDS.add(int(x))
        except ValueError:
            pass

GROUP_ADMIN_ONLY = os.getenv("GROUP_ADMIN_ONLY", "true").strip().lower() in ("1", "true", "yes", "y")
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", "15"))

HIST_PERIOD = os.getenv("HIST_PERIOD", "9mo")
HIST_INTERVAL = os.getenv("HIST_INTERVAL", "1d")

AUTO_NEWS_ENABLED = os.getenv("AUTO_NEWS_ENABLED", "true").strip().lower() in ("1", "true", "yes", "y")
AUTO_NEWS_INTERVAL_SECONDS = int(os.getenv("AUTO_NEWS_INTERVAL_SECONDS", "900"))
AUTO_NEWS_PER_TICKER_MAX_NEW = int(os.getenv("AUTO_NEWS_PER_TICKER_MAX_NEW", "1"))
AUTO_NEWS_PER_CHAT_MAX_PER_RUN = int(os.getenv("AUTO_NEWS_PER_CHAT_MAX_PER_RUN", "5"))

# WIB allow-window: "07:30-16:30" (auto-news runs only inside this window). Empty => always allowed.
QUIET_HOURS_WIB = os.getenv("QUIET_HOURS_WIB", "").strip()

NEWS_LIMIT_TOTAL = int(os.getenv("NEWS_LIMIT_TOTAL", "6"))


# =========================
# News sources (full from Excel)
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
    "FortuneIDN": {
        "domain": "fortuneidn.com",
        "paths": ["/tag/corporate-action"],
    },
    "Investing": {
        "domain": "id.investing.com",
        "paths": ["/news/stock-market-news"],
    },
    "SindoNews": {
        "domain": "sindonews.com",
        "paths": ["/topic/707/saham"],
    },
}


# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("stockbot")


# =========================
# Rate limit
# =========================
_LAST_CALL: Dict[Tuple[int, str], float] = {}


def _rate_limit_ok(user_id: int, key: str, seconds: int = RATE_LIMIT_SECONDS) -> bool:
    now = time.time()
    k = (user_id, key)
    last = _LAST_CALL.get(k, 0.0)
    if now - last < seconds:
        return False
    _LAST_CALL[k] = now
    return True


# =========================
# Helpers
# =========================
def _safe_float(x: Any) -> float:
    try:
        if isinstance(x, pd.Series):
            return float(x.iloc[0])
        return float(x)
    except Exception:
        return float("nan")


def _safe_str(x: Any, fallback: str = "n/a") -> str:
    if x is None:
        return fallback
    try:
        if pd.isna(x):
            return fallback
    except Exception:
        pass
    s = str(x).strip()
    return s if s else fallback


def _fmt_num(x: Any) -> str:
    try:
        v = _safe_float(x)
        if pd.isna(v):
            return "n/a"
        return f"{v:.2f}"
    except Exception:
        return "n/a"


def _fmt_pct_from_fraction(x: Any) -> str:
    try:
        v = _safe_float(x)
        if pd.isna(v):
            return "n/a"
        return f"{v * 100:.2f}%"
    except Exception:
        return "n/a"


def _fmt_pct_from_percent(x: Any) -> str:
    try:
        v = _safe_float(x)
        if pd.isna(v):
            return "n/a"
        return f"{v:.2f}%"
    except Exception:
        return "n/a"


def _human_num(x: Any) -> str:
    try:
        v = _safe_float(x)
        if pd.isna(v):
            return "n/a"
    except Exception:
        return "n/a"
    av = abs(v)
    if av >= 1e12:
        return f"{v / 1e12:.2f}T"
    if av >= 1e9:
        return f"{v / 1e9:.2f}B"
    if av >= 1e6:
        return f"{v / 1e6:.2f}M"
    if av >= 1e3:
        return f"{v / 1e3:.2f}K"
    return f"{v:.2f}"


def _google_news_rss(query: str) -> str:
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=id&gl=ID&ceid=ID:id"


def _normalize_idx_ticker(code: str) -> str:
    c = code.strip().upper()
    return c if c.endswith(".JK") else f"{c}.JK"


def _code_only(ticker: str) -> str:
    return ticker.strip().upper().replace(".JK", "")


def _group_allowed(update: Update) -> bool:
    chat = update.effective_chat
    if not chat:
        return False
    if chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        if ALLOWED_GROUP_IDS and chat.id not in ALLOWED_GROUP_IDS:
            return False
    return True


async def _is_group_admin(update: Update) -> bool:
    chat = update.effective_chat
    user = update.effective_user
    if not chat or not user:
        return False
    try:
        m = await chat.get_member(user.id)
        return m.status in ("administrator", "creator")
    except Exception:
        return False


async def _guard(update: Update, rate_key: str) -> bool:
    if not _group_allowed(update):
        return False

    uid = update.effective_user.id if update.effective_user else 0
    if uid and not _rate_limit_ok(uid, rate_key):
        return False

    chat = update.effective_chat
    if chat and chat.type in (ChatType.GROUP, ChatType.SUPERGROUP) and GROUP_ADMIN_ONLY:
        if not await _is_group_admin(update):
            return False

    return True


def _within_quiet_hours_wib() -> bool:
    if not QUIET_HOURS_WIB:
        return True
    try:
        start_s, end_s = QUIET_HOURS_WIB.split("-", 1)
        sh, sm = map(int, start_s.split(":"))
        eh, em = map(int, end_s.split(":"))
        now = datetime.now()
        start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
        end = now.replace(hour=eh, minute=em, second=0, microsecond=0)
        return start <= now <= end
    except Exception:
        return True


def _load_news_state() -> Dict[str, Any]:
    return load_json(NEWS_STATE_FILE, {})


def _save_news_state(state: Dict[str, Any]) -> None:
    save_json(NEWS_STATE_FILE, state)


def _autonews_enabled_for_chat(chat_id: int) -> bool:
    state = _load_news_state()
    meta = state.get("_meta", {})
    per_chat = meta.get("autonews_chat", {})
    return bool(per_chat.get(str(chat_id), True))


# =========================
# Market data
# =========================
def _fetch_price_history(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=HIST_PERIOD,
        interval=HIST_INTERVAL,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)
    df.index = pd.to_datetime(df.index)
    return df


def _compute_technical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MA20"] = ta.sma(out["Close"], length=20)
    out["MA50"] = ta.sma(out["Close"], length=50)
    out["RSI14"] = ta.rsi(out["Close"], length=14)
    out["RET_1D"] = out["Close"].pct_change(1) * 100
    out["RET_5D"] = out["Close"].pct_change(5) * 100
    out["RET_20D"] = out["Close"].pct_change(20) * 100
    return out


def _technical_signal(last: pd.Series) -> str:
    for k in ("MA20", "MA50", "RSI14"):
        v = last.get(k)
        if pd.isna(v):
            return "Data belum cukup untuk indikator (butuh ~50 hari)."

    close = _safe_float(last["Close"])
    ma20 = _safe_float(last["MA20"])
    ma50 = _safe_float(last["MA50"])
    rsi = _safe_float(last["RSI14"])

    if close > ma20 > ma50 and rsi < 70:
        return "Bullish (close > MA20 > MA50) dan RSI belum overbought."
    if close < ma20 < ma50 and rsi > 30:
        return "Bearish (close < MA20 < MA50) dan RSI belum oversold."
    if rsi >= 70:
        return "RSI overbought (>=70), rawan koreksi."
    if rsi <= 30:
        return "RSI oversold (<=30), rawan rebound."
    return "Netral / transisi (butuh konfirmasi)."


def _fetch_fundamental(ticker: str) -> Dict[str, Any]:
    t = yf.Ticker(ticker)
    info = t.info or {}
    return {
        "Sector": info.get("sector"),
        "MarketCap": info.get("marketCap"),
        "PER": info.get("trailingPE"),
        "PBV": info.get("priceToBook"),
        "ROE": info.get("returnOnEquity"),
        "EPS": info.get("trailingEps"),
        "DivYield": info.get("dividendYield"),
    }


def _fundamental_signal(f: Dict[str, Any]) -> str:
    notes: List[str] = []
    try:
        per = f.get("PER")
        if per is not None and not pd.isna(per) and float(per) > 0 and float(per) < 15:
            notes.append("PER relatif rendah (<15).")
    except Exception:
        pass
    try:
        pbv = f.get("PBV")
        if pbv is not None and not pd.isna(pbv) and float(pbv) > 0 and float(pbv) < 1.5:
            notes.append("PBV relatif rendah (<1.5).")
    except Exception:
        pass
    try:
        roe = f.get("ROE")
        if roe is not None and not pd.isna(roe) and float(roe) > 0.15:
            notes.append("ROE kuat (>15%).")
    except Exception:
        pass
    try:
        dy = f.get("DivYield")
        if dy is not None and not pd.isna(dy) and float(dy) >= 0.03:
            notes.append("Dividend yield menarik (>=3%).")
    except Exception:
        pass
    return " ".join(notes) if notes else "Fundamental netral / data terbatas."


def _format_analyze(ticker: str, df: pd.DataFrame, f: Dict[str, Any]) -> str:
    last = df.iloc[-1]
    dt = df.index[-1].to_pydatetime().date().isoformat()

    per = f.get("PER")
    pbv = f.get("PBV")
    roe = f.get("ROE")
    eps = f.get("EPS")
    dy = f.get("DivYield")

    return (
        f"*{ticker}*\n"
        f"**Tanggal data:** {dt}\n"
        f"**Close:** {_fmt_num(last.get('Close'))}\n"
        f"**Return 1D / 5D / 20D:** "
        f"{_fmt_pct_from_percent(last.get('RET_1D'))} / {_fmt_pct_from_percent(last.get('RET_5D'))} / {_fmt_pct_from_percent(last.get('RET_20D'))}\n"
        f"**MA20 / MA50:** {_fmt_num(last.get('MA20'))} / {_fmt_num(last.get('MA50'))}\n"
        f"**RSI14:** {_fmt_num(last.get('RSI14'))}\n"
        f"**Teknikal:** {_technical_signal(last)}\n\n"
        f"*Fundamental*\n"
        f"**Sektor:** {_safe_str(f.get('Sector'))}\n"
        f"**Market cap:** {_human_num(f.get('MarketCap'))}\n"
        f"**PER / PBV:** {_fmt_num(per)} / {_fmt_num(pbv)}\n"
        f"**ROE / EPS:** {_fmt_pct_from_fraction(roe)} / {_fmt_num(eps)}\n"
        f"**Div yield:** {_fmt_pct_from_fraction(dy)}\n"
        f"**Fundamental:** {_fundamental_signal(f)}\n\n"
        f"_Catatan: ringkasan otomatis, bukan rekomendasi investasi._"
    )


# =========================
# News fetching (domain + path aware)
# =========================
def _fetch_news_for_code(code: str) -> List[Tuple[str, str, str]]:
    code = code.strip().upper()
    results: List[Tuple[str, str, str]] = []

    for label, cfg in NEWS_SOURCES.items():
        domain = cfg["domain"]
        paths = cfg["paths"]

        q = f'{code} saham site:{domain}'
        feed = feedparser.parse(_google_news_rss(q))

        for e in feed.entries or []:
            title = getattr(e, "title", "") or ""
            link = getattr(e, "link", "") or ""
            if not title or not link:
                continue

            parsed = urlparse(link)
            if domain not in parsed.netloc:
                continue
            if not any(parsed.path.startswith(p) for p in paths):
                continue

            results.append((label, title.strip(), link.strip()))

    seen = set()
    uniq: List[Tuple[str, str, str]] = []
    for item in results:
        if item[2] in seen:
            continue
        seen.add(item[2])
        uniq.append(item)

    return uniq[:NEWS_LIMIT_TOTAL]


def _format_news_list(code: str, items: List[Tuple[str, str, str]]) -> str:
    lines = [f"*Berita {code}*"]
    for src, title, link in items:
        lines.append(f"- [{src}] {title}\n  {link}")
    return "\n".join(lines)


# =========================
# Auto-news job (with per-chat toggle)
# =========================
async def _job_auto_news(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not AUTO_NEWS_ENABLED:
        return
    if not _within_quiet_hours_wib():
        return

    state = _load_news_state()
    watch_data = load_json(WATCHLIST_FILE, {"chats": {}})
    chats: Dict[str, List[str]] = watch_data.get("chats", {}) if isinstance(watch_data, dict) else {}
    if not chats:
        return

    for chat_id_str, tickers in chats.items():
        try:
            chat_id = int(chat_id_str)
        except Exception:
            continue

        if ALLOWED_GROUP_IDS and chat_id not in ALLOWED_GROUP_IDS:
            continue
        if not _autonews_enabled_for_chat(chat_id):
            continue

        state.setdefault(chat_id_str, {})
        sent_this_run = 0

        tickers = tickers[:25] if isinstance(tickers, list) else []
        for tkr in tickers:
            code = _code_only(tkr)
            state[chat_id_str].setdefault(code, [])
            already = set(state[chat_id_str][code])

            items = _fetch_news_for_code(code)
            new_sent_for_ticker = 0

            for src, title, link in items:
                if link in already:
                    continue

                art = scrape_article(link)
                summary = None
                if art:
                    summary = summarize_article(
                        title=art.title or title,
                        source=src,
                        ticker=code,
                        url=link,
                        article_text=art.text,
                    )

                if summary:
                    msg = (
                        f"*News update {code}*\n"
                        f"**Sumber:** {src}\n"
                        f"**Judul:** {title}\n\n"
                        f"{summary}\n\n"
                        f"🔗 {link}\n"
                        f"_Ringkasan AI (parafrase), baca sumber untuk detail._"
                    )
                else:
                    msg = (
                        f"*News update {code}*\n"
                        f"**Sumber:** {src}\n"
                        f"**Judul:** {title}\n"
                        f"🔗 {link}"
                    )

                try:
                    await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode=ParseMode.MARKDOWN)
                except Exception:
                    await context.bot.send_message(chat_id=chat_id, text=msg.replace("*", "").replace("_", ""))

                state[chat_id_str][code].append(link)
                state[chat_id_str][code] = state[chat_id_str][code][-40:]

                new_sent_for_ticker += 1
                sent_this_run += 1

                await asyncio.sleep(1.0)

                if new_sent_for_ticker >= AUTO_NEWS_PER_TICKER_MAX_NEW:
                    break
                if sent_this_run >= AUTO_NEWS_PER_CHAT_MAX_PER_RUN:
                    break

            if sent_this_run >= AUTO_NEWS_PER_CHAT_MAX_PER_RUN:
                break

    _save_news_state(state)


# =========================
# Commands
# =========================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, "start"):
        return
    msg = (
        "Stockbot aktif.\n\n"
        "Perintah:\n"
        "- /analyze BBCA\n"
        "- /news BBCA\n"
        "- /watch add BBCA | /watch remove BBCA | /watch list\n"
        "- /autonews on | /autonews off\n"
        "- /help\n\n"
        "Catatan: auto-news berjalan untuk saham yang ada di watchlist chat."
    )
    await update.message.reply_text(msg)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, "help"):
        return
    await cmd_start(update, context)


async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, "analyze"):
        return
    if not context.args:
        await update.message.reply_text("Format: /analyze BBCA")
        return

    ticker = _normalize_idx_ticker(context.args[0])
    try:
        df = _fetch_price_history(ticker)
        if df.empty:
            await update.message.reply_text(f"Tidak ada data harga untuk {ticker}.")
            return
        df = _compute_technical(df)
        f = _fetch_fundamental(ticker)
        await update.message.reply_text(_format_analyze(ticker, df, f), parse_mode=ParseMode.MARKDOWN)
    except Exception:
        log.exception("analyze failed")
        await update.message.reply_text("Gagal ambil data (sementara). Coba lagi.")


async def cmd_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, "news"):
        return
    if not context.args:
        await update.message.reply_text("Format: /news BBCA")
        return

    code = context.args[0].strip().upper().replace(".JK", "")
    try:
        items = _fetch_news_for_code(code)
        if not items:
            await update.message.reply_text(f"Belum ketemu berita untuk {code} dari sumber yang ditentukan.")
            return
        await update.message.reply_text(_format_news_list(code, items), parse_mode=ParseMode.MARKDOWN)
    except Exception:
        log.exception("news failed")
        await update.message.reply_text("Gagal ambil berita (sementara). Coba lagi.")


async def cmd_watch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, "watch"):
        return
    if not context.args:
        await update.message.reply_text("Format: /watch add BBCA | /watch remove BBCA | /watch list")
        return

    action = context.args[0].strip().lower()
    chat_id = update.effective_chat.id
    wl = get_watchlist(WATCHLIST_FILE, chat_id)

    if action == "list":
        await update.message.reply_text("Watchlist kosong." if not wl else "Watchlist:\n" + "\n".join(wl))
        return

    if len(context.args) < 2:
        await update.message.reply_text("Format: /watch add BBCA | /watch remove BBCA")
        return

    ticker = _normalize_idx_ticker(context.args[1])

    if action == "add":
        wl = sorted(set(wl + [ticker]))
        set_watchlist(WATCHLIST_FILE, chat_id, wl)
        await update.message.reply_text(f"Ditambahkan: {ticker}")
        return

    if action == "remove":
        wl = [x for x in wl if x != ticker]
        set_watchlist(WATCHLIST_FILE, chat_id, wl)
        await update.message.reply_text(f"Dihapus: {ticker}")
        return

    await update.message.reply_text("Aksi tidak dikenal. Pakai: add/remove/list")


async def cmd_autonews(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, "autonews"):
        return
    if not context.args:
        await update.message.reply_text("Format: /autonews on | /autonews off")
        return

    action = context.args[0].strip().lower()
    state = _load_news_state()
    meta = state.setdefault("_meta", {})
    per_chat = meta.setdefault("autonews_chat", {})
    chat_id = update.effective_chat.id

    if action == "on":
        per_chat[str(chat_id)] = True
        _save_news_state(state)
        await update.message.reply_text("Auto-news: ON (chat ini)")
        return

    if action == "off":
        per_chat[str(chat_id)] = False
        _save_news_state(state)
        await update.message.reply_text("Auto-news: OFF (chat ini)")
        return

    await update.message.reply_text("Format: /autonews on | /autonews off")


# =========================
# Main
# =========================
def main() -> None:
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN belum diset (cek .env).")

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("news", cmd_news))
    app.add_handler(CommandHandler("watch", cmd_watch))
    app.add_handler(CommandHandler("autonews", cmd_autonews))

    if AUTO_NEWS_ENABLED:
        app.job_queue.run_repeating(
            _job_auto_news,
            interval=AUTO_NEWS_INTERVAL_SECONDS,
            first=60,
            name="auto_news",
        )

    log.info("Stockbot running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
