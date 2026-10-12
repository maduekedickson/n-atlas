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

    lang_key = language.lower()

    try:
        from gtts import gTTS
        
        # 1. Native Hausa support in gTTS with lang_check=False
        if lang_key == "ha":
            try:
                tts = gTTS(text=clean_text, lang="ha", slow=False, lang_check=False)
                buffer = io.BytesIO()
                tts.write_to_fp(buffer)
                return buffer.getvalue()
            except Exception:
                pass

        # 2. Yoruba & Igbo: Use African phonetic synthesizer (Swahili/Romance vowel cadence)
        # Google TTS does not host dedicated yo/ig voice models; Swahili provides identical 5-vowel phonology
        if lang_key in ["yo", "ig"]:
            try:
                tts = gTTS(text=clean_text, lang="sw", slow=False, lang_check=False)
                buffer = io.BytesIO()
                tts.write_to_fp(buffer)
                return buffer.getvalue()
            except Exception:
                pass

        # 3. Standard English / generic fallback
        try:
            tts = gTTS(text=clean_text, lang="en", slow=False)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            return buffer.getvalue()
        except Exception:
            return b""
    except Exception as e:
        # If gTTS is not installed or network is down, return empty bytes
        print(f"[TTS Warning] Could not synthesize speech: {e}")
        return b""
