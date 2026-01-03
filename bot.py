import os
import time
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple
from urllib.parse import quote_plus

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
ALLOWED_GROUP_IDS = set()
if ALLOWED_GROUP_IDS_RAW:
    for x in ALLOWED_GROUP_IDS_RAW.split(","):
        x = x.strip()
        if x:
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

# WIB quiet hours format: "07:30-16:30" (inclusive window where auto-news IS allowed).
QUIET_HOURS_WIB = os.getenv("QUIET_HOURS_WIB", "").strip()

NEWS_DOMAINS = {
    "IDX": "idx.co.id",
    "InvestorID": "investor.id",
    "CNBC": "cnbcindonesia.com",
    "RTI": "rti.co.id",
}
NEWS_ITEMS_PER_SOURCE = int(os.getenv("NEWS_ITEMS_PER_SOURCE", "2"))
NEWS_LIMIT_TOTAL = int(os.getenv("NEWS_LIMIT_TOTAL", "6"))


# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("idx-telegram-bot")


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
def _normalize_idx_ticker(code: str) -> str:
    c = code.strip().upper()
    return c if c.endswith(".JK") else f"{c}.JK"


def _code_only(ticker: str) -> str:
    return ticker.strip().upper().replace(".JK", "")


def _fmt_pct(x: Any) -> str:
    if x is None:
        return "n/a"
    try:
        if pd.isna(x):
            return "n/a"
    except Exception:
        pass
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return "n/a"


def _human_num(x: Any) -> str:
    if x is None:
        return "n/a"
    try:
        if pd.isna(x):
            return "n/a"
    except Exception:
        pass
    try:
        x = float(x)
    except Exception:
        return "n/a"
    ax = abs(x)
    if ax >= 1e12:
        return f"{x/1e12:.2f}T"
    if ax >= 1e9:
        return f"{x/1e9:.2f}B"
    if ax >= 1e6:
        return f"{x/1e6:.2f}M"
    if ax >= 1e3:
        return f"{x/1e3:.2f}K"
    return f"{x:.2f}"


def _google_news_rss(query: str) -> str:
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=id&gl=ID&ceid=ID:id"


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
    """
    If QUIET_HOURS_WIB is set, auto-news runs only inside that window.
    Note: uses server local time if server is set to WIB; recommended set server TZ to Asia/Jakarta.
    """
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
    if any(pd.isna(last.get(k)) for k in ("MA20", "MA50", "RSI14")):
        return "Data belum cukup untuk indikator (butuh ~50 hari)."
    close = float(last["Close"])
    ma20 = float(last["MA20"])
    ma50 = float(last["MA50"])
    rsi = float(last["RSI14"])

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

    def nf(x: Any) -> str:
        if x is None:
            return "n/a"
        try:
            if pd.isna(x):
                return "n/a"
        except Exception:
            pass
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "n/a"

    per = f.get("PER")
    pbv = f.get("PBV")
    roe = f.get("ROE")
    eps = f.get("EPS")
    dy = f.get("DivYield")

    per_s = "n/a" if per is None or (isinstance(per, float) and pd.isna(per)) else f"{float(per):.2f}"
    pbv_s = "n/a" if pbv is None or (isinstance(pbv, float) and pd.isna(pbv)) else f"{float(pbv):.2f}"
    roe_s = "n/a" if roe is None or (isinstance(roe, float) and pd.isna(roe)) else f"{float(roe)*100:.2f}%"
    eps_s = "n/a" if eps is None or (isinstance(eps, float) and pd.isna(eps)) else f"{float(eps):.2f}"
    dy_s = "n/a" if dy is None or (isinstance(dy, float) and pd.isna(dy)) else f"{float(dy)*100:.2f}%"

    return (
        f"*{ticker}*\n"
        f"**Tanggal data:** {dt}\n"
        f"**Close:** {float(last['Close']):.2f}\n"
        f"**Return 1D / 5D / 20D:** {_fmt_pct(last.get('RET_1D'))} / {_fmt_pct(last.get('RET_5D'))} / {_fmt_pct(last.get('RET_20D'))}\n"
        f"**MA20 / MA50:** {nf(last.get('MA20'))} / {nf(last.get('MA50'))}\n"
        f"**RSI14:** {nf(last.get('RSI14'))}\n"
        f"**Teknikal:** {_technical_signal(last)}\n\n"
        f"*Fundamental*\n"
        f"**Sektor:** {f.get('Sector') or 'n/a'}\n"
        f"**Market cap:** {_human_num(f.get('MarketCap'))}\n"
        f"**PER / PBV:** {per_s} / {pbv_s}\n"
        f"**ROE / EPS:** {roe_s} / {eps_s}\n"
        f"**Div yield:** {dy_s}\n"
        f"**Fundamental:** {_fundamental_signal(f)}\n\n"
        f"_Catatan: ringkasan otomatis, bukan rekomendasi investasi._"
    )


# =========================
# News fetching (domain-filtered) + Auto push with Gemini summary
# =========================
def _fetch_news_for_code(code: str) -> List[Tuple[str, str, str]]:
    code = code.strip().upper()
    results: List[Tuple[str, str, str]] = []

    for label, domain in NEWS_DOMAINS.items():
        q = f'{code} saham site:{domain}'
        feed = feedparser.parse(_google_news_rss(q))
        entries = feed.entries or []
        count = 0
        for e in entries:
            title = getattr(e, "title", "") or ""
            link = getattr(e, "link", "") or ""
            if not title or not link:
                continue
            results.append((label, title.strip(), link.strip()))
            count += 1
            if count >= NEWS_ITEMS_PER_SOURCE:
                break

    # de-dup by link
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


def _load_news_state() -> Dict[str, Any]:
    # {chat_id: {CODE: [link1, link2, ...]}}
    return load_json(NEWS_STATE_FILE, {})


def _save_news_state(state: Dict[str, Any]) -> None:
    save_json(NEWS_STATE_FILE, state)


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

        state.setdefault(chat_id_str, {})
        sent_this_run = 0

        # Limit tickers per run to avoid bursts
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

                # Attempt scrape -> Gemini summarize; if scrape fails, fallback to headline-only push
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
                        f"📰 *News update {code}*\n"
                        f"**Sumber:** {src}\n"
                        f"**Judul:** {title}\n\n"
                        f"{summary}\n\n"
                        f"🔗 {link}\n"
                        f"_Ringkasan AI (parafrase), baca sumber untuk detail._"
                    )
                else:
                    msg = (
                        f"📰 *News update {code}*\n"
                        f"**Sumber:** {src}\n"
                        f"**Judul:** {title}\n"
                        f"🔗 {link}"
                    )

                try:
                    await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode=ParseMode.MARKDOWN)
                except Exception:
                    # If markdown breaks (rare), retry plain text
                    await context.bot.send_message(chat_id=chat_id, text=msg.replace("*", "").replace("_", ""))

                # Update state (dedup)
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
        "Perintah:\n"
        "- /analyze BBCA\n"
        "- /news BBCA\n"
        "- /watch add BBCA | /watch remove BBCA | /watch list\n"
        "- /autonews on | /autonews off (admin)\n\n"
        "Catatan: auto-news jalan untuk saham yang ada di watchlist group."
    )
    await update.message.reply_text(msg)


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
    # This command toggles env-like runtime flag stored in NEWS_STATE_FILE meta.
    # Still guarded by admin-only in group (via _guard).
    if not await _guard(update, "autonews"):
        return
    if not context.args:
        await update.message.reply_text("Format: /autonews on | /autonews off")
        return

    action = context.args[0].strip().lower()
    state = _load_news_state()
    meta = state.setdefault("_meta", {})
    chat_id = update.effective_chat.id
    per_chat = meta.setdefault("autonews_chat", {})
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


def _autonews_enabled_for_chat(chat_id: int) -> bool:
    state = _load_news_state()
    meta = state.get("_meta", {})
    per_chat = meta.get("autonews_chat", {})
    # default: ON (if AUTO_NEWS_ENABLED), unless explicitly disabled
    return bool(per_chat.get(str(chat_id), True))


async def _job_auto_news_wrapper(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Wrapper that respects per-chat toggle without reworking the whole pipeline.
    """
    if not AUTO_NEWS_ENABLED:
        return
    if not _within_quiet_hours_wib():
        return

    watch_data = load_json(WATCHLIST_FILE, {"chats": {}})
    chats = watch_data.get("chats", {}) if isinstance(watch_data, dict) else {}
    # Temporarily filter chats based on toggle by cloning watchlists in-memory
    filtered = {"chats": {}}
    for chat_id_str, tickers in (chats or {}).items():
        try:
            chat_id = int(chat_id_str)
        except Exception:
            continue
        if not _autonews_enabled_for_chat(chat_id):
            continue
        filtered["chats"][chat_id_str] = tickers

    # save filtered snapshot to context for reuse by _job_auto_news
    # easiest: swap file read by setting context.job.data, but we keep simple:
    # monkey: write to temp file is overkill; so just run a small customized loop here.
    # To keep the code small, we call the main job but temporarily override WATCHLIST_FILE is not safe.
    # So we re-run minimal logic: emulate _job_auto_news using filtered dict.

    state = _load_news_state()

    for chat_id_str, tickers in filtered["chats"].items():
        try:
            chat_id = int(chat_id_str)
        except Exception:
            continue

        if ALLOWED_GROUP_IDS and chat_id not in ALLOWED_GROUP_IDS:
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
                        f"📰 *News update {code}*\n"
                        f"**Sumber:** {src}\n"
                        f"**Judul:** {title}\n\n"
                        f"{summary}\n\n"
                        f"🔗 {link}\n"
                        f"_Ringkasan AI (parafrase), baca sumber untuk detail._"
                    )
                else:
                    msg = (
                        f"📰 *News update {code}*\n"
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
# Main
# =========================
def main() -> None:
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN belum diset (cek .env).")

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("news", cmd_news))
    app.add_handler(CommandHandler("watch", cmd_watch))
    app.add_handler(CommandHandler("autonews", cmd_autonews))

    if AUTO_NEWS_ENABLED:
        app.job_queue.run_repeating(
            _job_auto_news_wrapper,
            interval=AUTO_NEWS_INTERVAL_SECONDS,
            first=60,
            name="auto_news",
        )

    log.info("Bot running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
