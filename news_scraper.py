import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass

@dataclass
class Article:
    title: str
    text: str

def scrape_article(url):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "lxml")

        title = soup.title.text if soup.title else ""
        article = soup.find("article") or soup.find("main")
        if not article:
            return None

        text = "\n".join(p.get_text(strip=True) for p in article.find_all("p"))
        if len(text) < 300:
            return None

        return Article(title=title, text=text[:12000])
    except Exception:
        return None
