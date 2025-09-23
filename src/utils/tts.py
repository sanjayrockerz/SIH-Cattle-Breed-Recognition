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
import os
from gtts import gTTS
import streamlit as st
import tempfile

def tts_read_aloud(text: str, key: str = "tts"):
    """
    Streamlit button to read aloud any text using Google TTS.
    Args:
        text (str): Text to read aloud (English only)
        key (str): Unique key for Streamlit widget
    """
    if st.button("🔊 Read aloud", key=key):
        # Generate temporary mp3 file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts = gTTS(text)
            tmp_file_path = tmp_file.name
            tts.save(tmp_file_path)
        # Play audio in Streamlit
        with open(tmp_file_path, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")
        # Delete file after playback
        try:
            os.remove(tmp_file_path)
        except Exception:
            pass
