import json
import os

def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def get_watchlist(path, chat_id):
    data = load_json(path, {"chats": {}})
    return data.get("chats", {}).get(str(chat_id), [])

def set_watchlist(path, chat_id, tickers):
    data = load_json(path, {"chats": {}})
    data.setdefault("chats", {})[str(chat_id)] = tickers
    save_json(path, data)
