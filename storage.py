import json
import os
from typing import Any, Dict, List


def load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, data: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def get_watchlist(watchlist_file: str, chat_id: int) -> List[str]:
    data = load_json(watchlist_file, {"chats": {}})
    chats: Dict[str, Any] = data.setdefault("chats", {})
    wl = chats.get(str(chat_id), [])
    if not isinstance(wl, list):
        wl = []
    wl = sorted(set(str(x).upper().strip() for x in wl if str(x).strip()))
    return wl


def set_watchlist(watchlist_file: str, chat_id: int, tickers: List[str]) -> None:
    data = load_json(watchlist_file, {"chats": {}})
    chats: Dict[str, Any] = data.setdefault("chats", {})
    chats[str(chat_id)] = sorted(set(tickers))
    save_json(watchlist_file, data)
