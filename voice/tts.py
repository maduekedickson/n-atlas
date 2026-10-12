"""
Text-to-Speech (TTS) Module for N-ATLaS PS2 Voice-First Access.
Synthesizes spoken audio feedback for English, Hausa, Yoruba, and Igbo.
"""

import io
import os
from typing import Optional


def synthesize_speech(
    text: str,
    language: str = "en",
    output_path: Optional[str] = None,
) -> bytes:
    """
    Synthesizes speech audio from text using gTTS with language-aware mapping.
    Supported languages:
        - 'en': English (Nigerian accent preference when available)
        - 'ha': Hausa
        - 'yo': Yoruba
        - 'ig': Igbo
    Returns raw MP3 audio bytes.
    """
    clean_text = text.strip()
    if not clean_text:
        return b""

    # Map internal language codes to TTS supported codes
    lang_map = {
        "en": "en",
        "ha": "ha",
        "yo": "yo",
        "ig": "ig",
    }
    tts_lang = lang_map.get(language.lower(), "en")

    try:
        from gtts import gTTS
        try:
            tts = gTTS(text=clean_text, lang=tts_lang, slow=False, lang_check=True)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            audio_bytes = buffer.getvalue()
        except Exception:
            # Fallback to English TTS voice for non-supported dialects
            tts = gTTS(text=clean_text, lang="en", slow=False)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            audio_bytes = buffer.getvalue()

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(audio_bytes)

        return audio_bytes
    except Exception as e:
        # If gTTS is not installed or network is down, return empty bytes
        print(f"[TTS Warning] Could not synthesize speech: {e}")
        return b""
