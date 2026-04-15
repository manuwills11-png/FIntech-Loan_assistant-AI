"""
Translation service using deep-translator (free, no API key required).
All calls have a 5-second timeout so they never hang the API.
"""

from __future__ import annotations

import logging
import signal
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 5


def _translate_with_timeout(source: str, target: str, text: str) -> Optional[str]:
    """Run a GoogleTranslator call with a timeout using a thread."""
    import concurrent.futures
    from deep_translator import GoogleTranslator  # type: ignore

    def _do_translate():
        return GoogleTranslator(source=source, target=target).translate(text)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_translate)
        try:
            return future.result(timeout=_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            logger.warning("Translation timed out after %ds", _TIMEOUT_SECONDS)
            return None
        except Exception as exc:
            logger.warning("Translation error: %s", exc)
            return None


def detect_language(text: str) -> str:
    """Detect the language of input text. Returns language code e.g. 'hi', 'en'."""
    try:
        import langdetect  # type: ignore
        return langdetect.detect(text)
    except Exception:
        return "en"


def translate_to_english(text: str, source_language: Optional[str] = None) -> Tuple[str, str]:
    """Translate text to English. Returns (translated_text, source_language_code)."""
    if not text.strip():
        return text, source_language or "en"

    source = source_language or "auto"
    if source == "en":
        return text, "en"

    result = _translate_with_timeout(source=source, target="en", text=text)
    return (result or text), (source_language or "en")


def translate_from_english(text: str, target_language: str) -> str:
    """Translate English text to the target language."""
    if not text.strip() or target_language == "en":
        return text

    result = _translate_with_timeout(source="en", target=target_language, text=text)
    return result or text
