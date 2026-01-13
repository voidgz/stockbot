import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")


def summarize_article(
    *,
    title: str,
    source: str,
    ticker: str,
    url: str,
    article_text: str,
) -> str:
    """
    Investor-focused, safe, structured summary.
    """

    if not article_text or len(article_text) < 300:
        return (
            "Berita terdeteksi, namun isi artikel terlalu singkat atau tidak dapat diproses.\n"
            "Silakan baca langsung dari sumber."
        )

    prompt = f"""
Kamu adalah analis pasar modal Indonesia.

Tugas:
Ringkas berita berikut untuk investor ritel saham Indonesia.

Aturan WAJIB:
- Jangan menyalin kalimat artikel secara langsung.
- Gunakan parafrase.
- Jangan memberi rekomendasi beli/jual.
- Fokus pada dampak ke saham {ticker}.
- Jika berita tidak relevan dengan {ticker}, jawab: "Tidak cukup relevan untuk emiten ini."

Format output HARUS seperti ini:

Ringkasan:
- (2–4 poin ringkas tentang isi utama berita)

Dampak ke {ticker}:
- Positif / Negatif / Netral
- Alasan singkat (1–2 poin)

Hal yang Perlu Dipantau:
- (1–3 poin konkret, jika ada)

Metadata:
Judul: {title}
Sumber: {source}
URL: {url}

Teks artikel (untuk dipahami, bukan disalin):
{article_text}
""".strip()

    try:
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception:
        return (
            "Gagal memproses ringkasan AI.\n"
            "Silakan baca berita langsung dari sumber."
        )
