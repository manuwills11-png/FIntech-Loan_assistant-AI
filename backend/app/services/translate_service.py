"""
Translation service using deep-translator (free, no API key required).

Batch endpoint uses one worker thread and a single GoogleTranslator instance so
many UI strings do not each spawn a new thread + 5s timeout (which caused slow
requests, proxy timeouts, and perceived 500s).
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_SINGLE_TIMEOUT_SECONDS = 20
_BATCH_TIMEOUT_SECONDS = 120

# Frontend / ISO codes → Google Translate target code (deep-translator).
# Unsupported / low-resource UI codes fall back to Hindi.
_GOOGLE_TARGET_ALIASES: dict[str, str] = {
    "kok": "hi",
    "mni": "hi",
    "ks": "hi",
}


def normalize_target_lang(target_language: str) -> str:
    code = (target_language or "en").split("-")[0].strip().lower()
    if not code or code == "en":
        return "en"
    return _GOOGLE_TARGET_ALIASES.get(code, code)


def _translate_with_timeout(source: str, target: str, text: str) -> Optional[str]:
    """Run a GoogleTranslator call with a timeout using a thread."""
    from deep_translator import GoogleTranslator  # type: ignore

    def _do_translate():
        return GoogleTranslator(source=source, target=target).translate(text)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_translate)
        try:
            return future.result(timeout=_SINGLE_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            logger.warning("Translation timed out after %ds", _SINGLE_TIMEOUT_SECONDS)
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

    target = normalize_target_lang(target_language)
    if target == "en":
        return text

    result = _translate_with_timeout(source="en", target=target, text=text)
    return result or text


def translate_batch_from_english(texts: list[str], target_language: str) -> list[str]:
    """
    Translate many English strings in one worker (one translator instance).
    On any failure, leaves that string unchanged (English).
    """
    if target_language == "en":
        return list(texts)
    target = normalize_target_lang(target_language)
    if target == "en":
        return list(texts)

    results = list(texts)

    def _run_batch() -> None:
        from deep_translator import GoogleTranslator  # type: ignore

        translator = GoogleTranslator(source="en", target=target)
        for i, raw in enumerate(texts):
            text = raw if isinstance(raw, str) else ""
            if not text.strip():
                continue
            try:
                results[i] = translator.translate(text)
            except Exception as exc:
                logger.warning("translate_batch item %s failed: %s", i, exc)
                results[i] = text

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_batch)
        try:
            future.result(timeout=_BATCH_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            logger.warning("translate_batch timed out after %ds", _BATCH_TIMEOUT_SECONDS)
        except Exception as exc:
            logger.warning("translate_batch failed: %s", exc)

    return results
