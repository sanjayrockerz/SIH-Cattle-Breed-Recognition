"""
Translation utility for Streamlit Cattle Breed App
Uses googletrans for breed description translation
Modular and non-intrusive

Usage:
    from src.utils.translate import translate_text
    translated = translate_text(text, dest_lang="hi")

Supported dest_lang codes:
- en: English
- hi: Hindi
- ta: Tamil
- te: Telugu
- kn: Kannada
"""
from googletrans import Translator

LANG_MAP = {
    "English": "en",
    "हिंदी (Hindi)": "hi",
    "தமிழ் (Tamil)": "ta",
    "తెలుగు (Telugu)": "te",
    "ಕನ್ನಡ (Kannada)": "kn",
}

def translate_text(text: str, dest_lang: str = "en") -> str:
    """Translate text to the destination language code."""
    translator = Translator()
    try:
        result = translator.translate(text, dest=dest_lang)
        return result.text
    except Exception:
        return text  # fallback to original if translation fails
