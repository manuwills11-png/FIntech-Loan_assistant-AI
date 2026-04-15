"""
Speech-to-Text service using Groq Whisper API (free tier).

Groq offers whisper-large-v3 for free as part of their API.
Uses the same GROQ_API_KEY already configured in .env
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_bytes: bytes,
    language_code: str = "en",
    encoding: str = "LINEAR16",
) -> Optional[str]:
    """
    Transcribe audio bytes to text using Groq Whisper API,
    then post-correct with LLM for better accuracy on Indian languages.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("GROQ_API_KEY not set – cannot transcribe audio.")
        return None

    try:
        from groq import Groq  # type: ignore

        client = Groq(api_key=api_key)

        lang = _simple_lang_code(language_code)

        response = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=("audio.webm", audio_bytes, "audio/webm"),
            language=lang,
            prompt=_get_lang_prompt(lang),
        )

        raw_text = response.text.strip() if response.text else None
        if not raw_text:
            return None

        # Post-correct with LLM for non-English languages
        if lang != "en":
            raw_text = _correct_transcription(client, raw_text, lang)

        return raw_text

    except Exception as exc:
        logger.error("Groq Whisper transcription failed: %s", exc)
        return None


def _correct_transcription(client: object, text: str, lang: str) -> str:
    """
    Use Groq LLM to fix misrecognized financial terms in transcription.
    This dramatically improves accuracy for Indian languages.
    """
    lang_names = {
        "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
        "ml": "Malayalam", "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati",
        "pa": "Punjabi", "ur": "Urdu",
    }
    lang_name = lang_names.get(lang, lang)

    try:
        from groq import Groq  # type: ignore
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a transcription corrector for {lang_name}. "
                        f"The user asked a financial question in {lang_name}. "
                        "Fix any speech-to-text errors in the transcription below. "
                        "Keep the same language and meaning. "
                        "Return ONLY the corrected text, nothing else. "
                        "Common topics: loan (ലോൺ/कर्ज/கடன்), EMI, interest (പലിശ/ब्याज), "
                        "savings (സേവിങ്സ്/बचत), bank (ബാങ്ക്/बैंक), risk (റിസ്ക്/जोखिम)."
                    ),
                },
                {"role": "user", "content": f"Correct this transcription: {text}"},
            ],
            max_tokens=200,
            timeout=10,
        )
        corrected = response.choices[0].message.content.strip()
        return corrected if corrected else text
    except Exception as exc:
        logger.warning("Transcription correction failed: %s", exc)
        return text


def _simple_lang_code(lang: str) -> str:
    """Strip region suffix: 'hi-IN' → 'hi', 'en-IN' → 'en'."""
    return lang.split("-")[0].lower()


def language_code_for(lang: str) -> str:
    """Return simple language code for Whisper (no region suffix needed)."""
    return _simple_lang_code(lang)


def _get_lang_prompt(lang: str) -> str:
    """Return a prompt hint in the target language to help Whisper accuracy."""
    prompts = {
        "hi": "यह एक वित्तीय सहायक है।",
        "ta": "இது ஒரு நிதி உதவியாளர்.",
        "te": "ఇది ఒక ఆర్థిక సహాయకుడు.",
        "kn": "ಇದು ಒಂದು ಹಣಕಾಸು ಸಹಾಯಕ.",
        "ml": "ഇത് ഒരു സാമ്പത്തിക സഹായിയാണ്.",
        "mr": "हे एक आर्थिक सहाय्यक आहे.",
        "bn": "এটি একটি আর্থিক সহকারী।",
        "gu": "આ એક નાણાકીય સહાયક છે.",
        "pa": "ਇਹ ਇੱਕ ਵਿੱਤੀ ਸਹਾਇਕ ਹੈ।",
        "ur": "یہ ایک مالی معاون ہے۔",
    }
    code = _simple_lang_code(lang)
    return prompts.get(code, "")
