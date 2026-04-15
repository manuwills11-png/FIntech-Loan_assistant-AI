"""
Text-to-Speech service using gTTS (Google Text-to-Speech).

gTTS is completely free — no API key, no billing, no account needed.
It uses the same Google Translate TTS endpoint that the browser uses.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Map simple language codes to gTTS supported codes
_LANG_MAP = {
    "en": "en",
    "hi": "hi",
    "ta": "ta",
    "te": "te",
    "kn": "kn",
    "mr": "mr",
    "bn": "bn",
    "ml": "ml",
}


def synthesize_speech(
    text: str,
    language_code: str = "en-IN",
    speaking_rate: float = 0.9,
) -> Optional[str]:
    """
    Convert text to speech using gTTS and return base64-encoded MP3.

    Args:
        text:           Text to convert to speech.
        language_code:  Language code e.g. 'hi-IN' or 'hi' or 'en'.
        speaking_rate:  Ignored by gTTS (always normal speed).

    Returns:
        Base64-encoded MP3 string, or None on failure.
    """
    try:
        from gtts import gTTS  # type: ignore

        # Normalise language code: 'hi-IN' → 'hi'
        lang = language_code.split("-")[0].lower()
        lang = _LANG_MAP.get(lang, "en")

        tts = gTTS(text=text, lang=lang, slow=False)

        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)

        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as exc:
        logger.error("gTTS synthesis failed: %s", exc)
        return None


def language_code_for(lang: str) -> str:
    """Return simple language code for gTTS."""
    return lang.split("-")[0].lower()
