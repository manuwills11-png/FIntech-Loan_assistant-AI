"""
GET /demo-user/{user_id}  – Pre-loaded demo user profiles for judges and testing.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    DemoUserOutput,
    LoanRiskInput,
    RiskCategory,
    SupportedLanguage,
)
from app.services import risk_service

router = APIRouter(prefix="/demo-user", tags=["Demo Mode"])


# ── Pre-loaded demo profiles ──────────────────────────────────────────────────

_DEMO_USERS = {
    "farmer": {
        "user_id": "farmer",
        "name": "Ramesh Kumar",
        "profile": "Small farmer from rural Maharashtra with irregular seasonal income.",
        "loan_data": LoanRiskInput(
            monthly_income=18000,
            monthly_expenses=12000,
            existing_loans=50000,
            emi_amount=3500,
            repayment_history_score=55,
            loan_amount_requested=100000,
            loan_tenure_months=36,
            employment_type="farmer",
            language=SupportedLanguage.HINDI,
        ),
    },
    "student": {
        "user_id": "student",
        "name": "Priya Sharma",
        "profile": "Engineering student seeking education loan with no income, dependent on family.",
        "loan_data": LoanRiskInput(
            monthly_income=0,
            monthly_expenses=8000,
            existing_loans=0,
            emi_amount=0,
            repayment_history_score=0,
            loan_amount_requested=500000,
            loan_tenure_months=60,
            employment_type="student",
            language=SupportedLanguage.ENGLISH,
        ),
    },
    "salaried": {
        "user_id": "salaried",
        "name": "Anita Desai",
        "profile": "Salaried software professional in Bengaluru with good repayment history.",
        "loan_data": LoanRiskInput(
            monthly_income=75000,
            monthly_expenses=35000,
            existing_loans=200000,
            emi_amount=12000,
            repayment_history_score=85,
            loan_amount_requested=1000000,
            loan_tenure_months=60,
            employment_type="salaried",
            language=SupportedLanguage.ENGLISH,
        ),
    },
    "high_risk": {
        "user_id": "high_risk",
        "name": "Suresh Patel",
        "profile": "Self-employed vendor with high existing debt and poor repayment record.",
        "loan_data": LoanRiskInput(
            monthly_income=22000,
            monthly_expenses=20000,
            existing_loans=300000,
            emi_amount=9000,
            repayment_history_score=30,
            loan_amount_requested=200000,
            loan_tenure_months=24,
            employment_type="self_employed",
            language=SupportedLanguage.ENGLISH,
        ),
    },
}


@router.get(
    "/{user_id}",
    response_model=DemoUserOutput,
    summary="Load a pre-configured demo user profile",
)
async def get_demo_user(user_id: str) -> DemoUserOutput:
    """
    Load a pre-configured demo profile and run a live risk assessment.

    Available profiles:
    - **farmer** – Rural farmer with irregular income (Hindi)
    - **student** – Student with no income seeking education loan
    - **salaried** – Well-paid professional with good history
    - **high_risk** – Over-leveraged self-employed individual

    Returns the demo user's financial data along with a freshly computed risk result.
    """
    profile = _DEMO_USERS.get(user_id.lower())
    if not profile:
        available = ", ".join(_DEMO_USERS.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Demo user '{user_id}' not found. Available: {available}",
        )

    loan_data: LoanRiskInput = profile["loan_data"]

    try:
        risk_result = risk_service.predict_risk(loan_data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Risk computation failed: {exc}")

    return DemoUserOutput(
        user_id=profile["user_id"],
        name=profile["name"],
        profile=profile["profile"],
        loan_data=loan_data,
        risk_result=risk_result,
    )


@router.get("", summary="List all available demo users")
async def list_demo_users() -> list[dict]:
    """Return all available demo user IDs and names."""
    return [
        {"user_id": uid, "name": p["name"], "profile": p["profile"]}
        for uid, p in _DEMO_USERS.items()
    ]
