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
    Uses session state for reliable playback after rerun.
    Args:
        text (str): Text to read aloud
        key (str): Unique key for Streamlit widget
        language (str): Language for TTS (English, Hindi, Tamil, Telugu, Kannada)
    """
    audio_key = f"tts_audio_{key}"
    play_key = f"tts_play_{key}"
    if st.button("🔊 Read aloud", key=key):
        lang_code = LANG_MAP.get(language, "en")
        tts = gTTS(text, lang=lang_code)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        st.session_state[audio_key] = mp3_fp.read()
        st.session_state[play_key] = True
    # Play audio if available and play flag is set
    if st.session_state.get(play_key) and st.session_state.get(audio_key):
        st.audio(st.session_state[audio_key], format="audio/mp3")
        st.session_state[play_key] = False
