"""
POST /predict-risk  – Loan risk prediction endpoint.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import LoanRiskInput, LoanRiskOutput, SupportedLanguage
from app.services import risk_service
from app.services.translate_service import translate_from_english

router = APIRouter(prefix="/predict-risk", tags=["Risk Prediction"])


@router.post("", response_model=LoanRiskOutput, summary="Predict loan risk")
async def predict_risk(data: LoanRiskInput) -> LoanRiskOutput:
    """
    Predict the financial risk of a loan application.

    Returns a risk score (0–100), category (Low/Medium/High),
    plain-language explanation, and key contributing factors.

    Explanation is translated into the requested language.
    """
    try:
        result = risk_service.predict_risk(data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # Translate explanation and recommendation if non-English requested
    lang = data.language.value
    if lang != "en":
        result.explanation = translate_from_english(result.explanation, lang)
        result.recommendation = translate_from_english(result.recommendation, lang)
        result.key_factors = [translate_from_english(f, lang) for f in result.key_factors]

    return result
