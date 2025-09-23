"""
Google Text-to-Speech (TTS) utility for Streamlit
Modular, non-intrusive integration for any UI section

Usage:
    from src.utils.tts import tts_read_aloud
    tts_read_aloud(text, key="main_desc")

Maintainers:
- This utility does NOT affect ML model or prediction workflow.
- Audio file is deleted after playback to prevent clutter.
"""
import streamlit as st
from gtts import gTTS
from io import BytesIO

LANG_MAP = {
    "English": "en",
    "हिंदी (Hindi)": "hi",
    "தமிழ் (Tamil)": "ta",
    "తెలుగు (Telugu)": "te",
    "ಕನ್ನಡ (Kannada)": "kn",
}

def tts_read_aloud(text: str, key: str = "tts", language: str = "English"):
    """
    Streamlit button to read aloud any text using Google TTS.
    Args:
        text (str): Text to read aloud
        key (str): Unique key for Streamlit widget
        language (str): Language for TTS (English, Hindi, Tamil, Telugu, Kannada)
    """
    if st.button("🔊 Read aloud", key=key):
        lang_code = LANG_MAP.get(language, "en")
        tts = gTTS(text, lang=lang_code)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        st.audio(mp3_fp.read(), format="audio/mp3")
