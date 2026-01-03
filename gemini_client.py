import os
from typing import Optional

import google.generativeai as genai


def _configured() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def summarize_article(
    *,
    title: str,
    source: str,
    ticker: str,
    url: str,
    article_text: str,
    model_name: str = "gemini-2.5-flash",
) -> Optional[str]:
    """
    Returns a concise investor-focused summary.
    We only output AI summary + source link (avoid reproducing full article).
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Keep prompt strict: short, non-copyright-infringing, actionable.
    prompt = f"""

Anda adalah analis pasar modal Indonesia.

Tugas Anda:
1. Kumpulkan berita terbaru hari ini terkait seluruh saham yang terdaftar di Bursa Efek Indonesia (BEI).
2. Fokus pada berita yang berdampak terhadap harga saham, kinerja perusahaan, atau sentimen pasar.
3. Identifikasi dan soroti corporate action jika ada, seperti:
   - Dividen
   - Stock split / reverse split
   - Right issue
   - Buyback saham
   - Akuisisi / merger
   - IPO anak usaha
   - Backdoor
   - Perubahan direksi atau komisaris
4. Abaikan berita yang tidak relevan atau bersifat opini tanpa data.

Aturan:
- Jangan menyalin teks artikel panjang; parafrase saja.
- Fokus pada dampak ke emiten {ticker} (positif/negatif/netral) + alasan.
- Sertakan "Apa yang perlu dipantau" (1-3 poin).
- Jika isi artikel tidak relevan dengan {ticker}, jawab: "Tidak cukup relevan."

Metadata:
- Judul: {title}
- Sumber: {source}
- URL: {url}

Teks artikel (untuk dipahami, bukan untuk disalin):
{article_text}
""".strip()

    resp = model.generate_content(prompt)
    text = (getattr(resp, "text", None) or "").strip()
    return text or None
