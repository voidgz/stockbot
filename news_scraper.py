import re
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


@dataclass
class Article:
    title: str
    text: str


def _get_html(url: str, timeout: int = 15) -> Optional[str]:
    try:
        r = requests.get(url, headers=UA_HEADERS, timeout=timeout)
        if r.status_code >= 400:
            return None
        return r.text
    except Exception:
        return None


def _clean_text(s: str) -> str:
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def _extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""


def _extract_paragraphs(container) -> str:
    if not container:
        return ""
    ps = container.find_all("p")
    text = "\n".join(p.get_text(" ", strip=True) for p in ps if p.get_text(strip=True))
    return _clean_text(text)


def _scrape_idx(url: str, soup: BeautifulSoup) -> Article:
    # IDX press release / news pages: try common content containers
    title = _extract_title(soup)
    main = (
        soup.find("main")
        or soup.find("div", class_=re.compile(r"(content|article|detail|container)", re.I))
    )
    text = _extract_paragraphs(main)
    return Article(title=title, text=text)


def _scrape_cnbc(url: str, soup: BeautifulSoup) -> Article:
    title = _extract_title(soup)
    # CNBC Indonesia commonly uses detail_text, but can change.
    container = (
        soup.find("div", class_=re.compile(r"detail_text", re.I))
        or soup.find("div", class_=re.compile(r"(content|article|detail)", re.I))
        or soup.find("article")
    )
    text = _extract_paragraphs(container)
    return Article(title=title, text=text)


def _scrape_investorid(url: str, soup: BeautifulSoup) -> Article:
    title = _extract_title(soup)
    container = (
        soup.find("div", class_=re.compile(r"(content|article|detail|body)", re.I))
        or soup.find("article")
        or soup.find("main")
    )
    text = _extract_paragraphs(container)
    return Article(title=title, text=text)


def _scrape_rti(url: str, soup: BeautifulSoup) -> Article:
    title = _extract_title(soup)
    container = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=re.compile(r"(content|article|detail|body)", re.I))
    )
    text = _extract_paragraphs(container)
    return Article(title=title, text=text)


def scrape_article(url: str) -> Optional[Article]:
    """
    Best-effort scraper. If markup changes, this may return None or short text.
    Keep usage respectful: rate-limit + avoid massive crawling.
    """
    html = _get_html(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "lxml")
    host = (urlparse(url).netloc or "").lower()

    if "idx.co.id" in host:
        art = _scrape_idx(url, soup)
    elif "cnbcindonesia.com" in host:
        art = _scrape_cnbc(url, soup)
    elif "investor.id" in host:
        art = _scrape_investorid(url, soup)
    elif "rti.co.id" in host:
        art = _scrape_rti(url, soup)
    else:
        # Unknown domain: fallback to article/main
        title = _extract_title(soup)
        container = soup.find("article") or soup.find("main")
        text = _extract_paragraphs(container)
        art = Article(title=title, text=text)

    # Guardrails: need enough text to summarize
    text = (art.text or "").strip()
    if len(text) < 300:
        return None

    # Cap to avoid sending too much to Gemini
    capped = text[:12000]
    return Article(title=(art.title or "").strip(), text=capped)
