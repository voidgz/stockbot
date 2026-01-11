import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

def summarize_article(title, source, ticker, url, article_text):
    if not article_text:
        return "Berita relevan, silakan baca sumber."

    prompt = f"""
Ringkas berita berikut untuk investor saham Indonesia ({ticker}).
Fokus pada dampak ke saham dan sentimen.

Judul: {title}
Sumber: {source}

{article_text}
"""
    return model.generate_content(prompt).text.strip()

