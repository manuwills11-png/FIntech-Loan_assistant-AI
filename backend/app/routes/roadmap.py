"""
POST /roadmap  – Financial roadmap generator endpoint.

Produces:
  - Month-by-month repayment schedule
  - Expense reduction tips
  - Income improvement tips
  - LLM-generated plain-language summary
"""

from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException

from app.models.schemas import MonthlyPlanItem, RoadmapInput, RoadmapOutput
from app.services import ai_service
from app.services.translate_service import translate_from_english
from app.utils.helpers import format_inr, safe_divide

router = APIRouter(prefix="/roadmap", tags=["Financial Roadmap"])

# Annual interest rates by risk category (approximate)
_INTEREST_RATE_MAP = {
    "Low": 0.10,    # 10% p.a.
    "Medium": 0.14, # 14% p.a.
    "High": 0.18,   # 18% p.a.
}


def _compute_emi(principal: float, annual_rate: float, months: int) -> float:
    """Standard EMI formula: P × r × (1+r)^n / ((1+r)^n – 1)."""
    if annual_rate == 0:
        return principal / months
    r = annual_rate / 12
    emi = principal * r * (1 + r) ** months / ((1 + r) ** months - 1)
    return round(emi, 2)


def _build_repayment_plan(
    principal: float,
    annual_rate: float,
    months: int,
    emi: float,
) -> list[MonthlyPlanItem]:
    plan = []
    balance = principal
    r = annual_rate / 12

    for month in range(1, months + 1):
        interest = round(balance * r, 2)
        principal_payment = round(emi - interest, 2)
        closing = round(max(balance - principal_payment, 0), 2)

        plan.append(
            MonthlyPlanItem(
                month=month,
                opening_balance=round(balance, 2),
                emi_payment=emi,
                closing_balance=closing,
            )
        )
        balance = closing
        if balance <= 0:
            break

    return plan


def _expense_tips(data: RoadmapInput) -> list[str]:
    tips = []
    expense_ratio = safe_divide(data.monthly_expenses, data.monthly_income)

    if expense_ratio > 0.7:
        tips.append("Your expenses are over 70% of income. Try to cut discretionary spending by 20%.")
    if expense_ratio > 0.5:
        tips.append("Track all spending for one month to identify where money is going.")
    tips.append("Avoid taking new credit or consumer loans during the repayment period.")
    tips.append("Cook at home more often to reduce food expenses.")
    tips.append("Review subscriptions and memberships — cancel unused ones.")

    savings_gap = data.monthly_income - data.monthly_expenses - data.emi_amount
    if savings_gap < 0:
        tips.append(
            f"You currently have a monthly shortfall of {format_inr(abs(savings_gap))}. "
            "Consider part-time work to bridge this gap."
        )
    return tips[:5]


def _income_tips(data: RoadmapInput) -> list[str]:
    tips = [
        "Explore government skill-development programs (PMKVY, DDU-GKY) to improve earning potential.",
        "Consider small side income: tutoring, tailoring, delivery services, or seasonal farm labour.",
        "If you have land, explore agricultural subsidies or crop loans with lower interest rates.",
        "Digital literacy can open access to gig economy platforms and online freelance work.",
        "Form or join a Self-Help Group (SHG) for access to group credit at lower rates.",
    ]
    return tips


@router.post("", response_model=RoadmapOutput, summary="Generate financial roadmap")
async def generate_roadmap(data: RoadmapInput) -> RoadmapOutput:
    """
    Generate a personalised financial roadmap.

    Includes a month-by-month EMI schedule, expense reduction tips,
    income improvement suggestions, and a plain-language narrative summary.
    """
    # Determine interest rate from risk score
    if data.risk_score is None:
        rate = _INTEREST_RATE_MAP["Medium"]
    elif data.risk_score < 40:
        rate = _INTEREST_RATE_MAP["Low"]
    elif data.risk_score < 70:
        rate = _INTEREST_RATE_MAP["Medium"]
    else:
        rate = _INTEREST_RATE_MAP["High"]

    emi = _compute_emi(data.loan_amount_requested, rate, data.loan_tenure_months)
    plan = _build_repayment_plan(data.loan_amount_requested, rate, data.loan_tenure_months, emi)
    total_paid = round(emi * len(plan), 2)
    total_interest = round(total_paid - data.loan_amount_requested, 2)

    expense_tips = _expense_tips(data)
    income_tips = _income_tips(data)

    # Generate human-readable summary via LLM
    roadmap_context = {
        "loan_amount": data.loan_amount_requested,
        "emi": emi,
        "tenure_months": data.loan_tenure_months,
        "total_interest": total_interest,
        "monthly_income": data.monthly_income,
        "monthly_expenses": data.monthly_expenses,
        "expense_tips": expense_tips[:2],
        "income_tips": income_tips[:2],
    }
    lang = data.language.value
    summary_en = ai_service.generate_roadmap_narrative(roadmap_context, language=lang)
    summary = translate_from_english(summary_en, lang) if lang != "en" else summary_en

    return RoadmapOutput(
        repayment_plan=plan,
        expense_reduction_tips=expense_tips,
        income_improvement_tips=income_tips,
        summary=summary,
        total_interest_payable=total_interest,
        suggested_emi=emi,
    )
