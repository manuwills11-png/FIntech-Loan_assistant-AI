"""
POST /translate/batch  – Translate a batch of strings from English to a target language.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.translate_service import translate_from_english

router = APIRouter(prefix="/translate", tags=["Translation"])


class BatchTranslateRequest(BaseModel):
    texts: list[str]
    target_language: str


class BatchTranslateResponse(BaseModel):
    translations: list[str]


@router.post("/batch", response_model=BatchTranslateResponse)
async def batch_translate(data: BatchTranslateRequest) -> BatchTranslateResponse:
    """Translate multiple English strings to the target language at once."""
    if data.target_language == "en":
        return BatchTranslateResponse(translations=data.texts)

    translations = [
        translate_from_english(text, data.target_language)
        for text in data.texts
    ]
    return BatchTranslateResponse(translations=translations)
