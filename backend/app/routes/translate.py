"""
POST /translate/batch  – Translate a batch of strings from English to a target language.
"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.translate_service import translate_batch_from_english

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/translate", tags=["Translation"])


class BatchTranslateRequest(BaseModel):
    texts: list[str]
    target_language: str


class BatchTranslateResponse(BaseModel):
    translations: list[str]


@router.post("/batch", response_model=BatchTranslateResponse)
async def batch_translate(data: BatchTranslateRequest) -> BatchTranslateResponse:
    """Translate multiple English strings to the target language at once."""
    try:
        texts = [t if isinstance(t, str) else "" for t in data.texts]
        translations = translate_batch_from_english(texts, data.target_language)
        return BatchTranslateResponse(translations=translations)
    except Exception as exc:
        logger.exception("batch_translate failed: %s", exc)
        return BatchTranslateResponse(translations=list(data.texts))
