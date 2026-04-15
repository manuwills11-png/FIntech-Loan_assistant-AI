"""
Risk prediction service.

Computes a CIBIL-inspired 7-factor risk score (0–100) blended with an
optional ML model signal, then calls Groq for personalised AI advice.

Factors & weights (mirror RBI/CIBIL methodology):
  1. CIBIL / Credit Score      30 %
  2. EMI Burden                25 %
  3. Debt Load                 15 %
  4. Savings Buffer            15 %
  5. Loan Size Risk            10 %
  6. Age Eligibility            3 %
  7. Employment Stability       2 %
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np

from app.models.schemas import (
    FactorScore,
    LoanRiskInput,
    LoanRiskOutput,
    RiskCategory,
)
from app.utils.helpers import clamp, safe_divide, score_to_category

logger = logging.getLogger(__name__)

# ── Model path (optional – blended in if available) ───────────────────────────
_MODEL_PATH = Path(os.getenv("MODEL_PATH", "./ml/loan_risk_model.pkl"))
_model = None  # lazy-loaded


def _load_model():
    global _model
    if _model is None:
        if not _MODEL_PATH.exists():
            return None
        _model = joblib.load(_MODEL_PATH)
    return _model


# ── CIBIL-style factor scoring ────────────────────────────────────────────────

def _cibil_to_risk(cibil: int) -> float:
    """Map CIBIL score (300–900) to risk (0–100). Higher CIBIL = lower risk."""
    # 750–900 → 0–20 risk (good)
    # 650–749 → 20–60 risk (fair)
    # 300–649 → 60–100 risk (poor)
    if cibil >= 750:
        return clamp((900 - cibil) / 150.0 * 20.0, 0.0, 20.0)
    elif cibil >= 650:
        return clamp(20.0 + (750 - cibil) / 100.0 * 40.0, 20.0, 60.0)
    else:
        return clamp(60.0 + (650 - cibil) / 350.0 * 40.0, 60.0, 100.0)


def _compute_cibil_factors(data: LoanRiskInput) -> Tuple[float, List[FactorScore]]:
    """
    Compute a weighted 0–100 risk score across 7 CIBIL-inspired factors.
    Each factor score is 0–100 (higher = worse).
    """
    factors: List[FactorScore] = []

    # 1. CIBIL / Credit Score (30 %)
    #    Use actual CIBIL if provided, else fall back to repayment_history_score
    if data.cibil_score is not None:
        credit_risk = _cibil_to_risk(data.cibil_score)
        credit_label = f"CIBIL Score ({data.cibil_score})"
    else:
        credit_risk = clamp(100.0 - data.repayment_history_score, 0.0, 100.0)
        credit_label = "Repayment History"
    factors.append(FactorScore(
        name="credit_score",
        label=credit_label,
        score=round(credit_risk, 1),
        weight=30,
        status=_status(credit_risk),
    ))

    # 2. EMI Burden (25 %)
    #    eti = emi / income; 0% → 0 risk; ≥ 50% → 100 risk
    eti = safe_divide(data.emi_amount, data.monthly_income)
    emi_risk = clamp(eti * 200.0, 0.0, 100.0)
    factors.append(FactorScore(
        name="emi_burden",
        label="EMI Burden",
        score=round(emi_risk, 1),
        weight=25,
        status=_status(emi_risk),
    ))

    # 3. Debt Load (15 %)
    #    dti = existing_loans / (income × 12); ≥ 67% of annual income → 100 risk
    dti = safe_divide(data.existing_loans, data.monthly_income * 12.0)
    debt_risk = clamp(dti * 150.0, 0.0, 100.0)
    factors.append(FactorScore(
        name="debt_load",
        label="Debt Load",
        score=round(debt_risk, 1),
        weight=15,
        status=_status(debt_risk),
    ))

    # 4. Savings Buffer (15 %)
    #    ≥ 20% savings → 0 risk; negative → 100 risk
    savings = data.monthly_income - data.monthly_expenses - data.emi_amount
    savings_ratio = safe_divide(savings, data.monthly_income)
    savings_risk = clamp((0.20 - savings_ratio) * 200.0, 0.0, 100.0)
    factors.append(FactorScore(
        name="savings_buffer",
        label="Savings Buffer",
        score=round(savings_risk, 1),
        weight=15,
        status=_status(savings_risk),
    ))

    # 5. Loan Size Risk (10 %)
    #    < 2× annual income → 0 risk; ≥ 10× → 100 risk
    lir = safe_divide(data.loan_amount_requested, data.monthly_income * 12.0)
    loan_risk = clamp((lir - 2.0) / 8.0 * 100.0, 0.0, 100.0)
    factors.append(FactorScore(
        name="loan_size",
        label="Loan Size",
        score=round(loan_risk, 1),
        weight=10,
        status=_status(loan_risk),
    ))

    # 6. Age Eligibility (3 %)
    #    Banks prefer 25–55; too young or near retirement = higher risk
    if data.age is not None:
        if data.age < 21:
            age_risk = 80.0
        elif data.age < 25:
            age_risk = 40.0
        elif data.age <= 55:
            age_risk = 0.0
        elif data.age <= 60:
            age_risk = 30.0
        else:
            age_risk = 70.0
    else:
        age_risk = 20.0  # neutral if not provided
    factors.append(FactorScore(
        name="age_eligibility",
        label="Age Eligibility",
        score=round(age_risk, 1),
        weight=3,
        status=_status(age_risk),
    ))

    # 7. Employment Stability (2 %)
    #    < 1 yr = high risk; 1–2 yr = fair; > 2 yr = good
    if data.employment_stability_years is not None:
        yrs = data.employment_stability_years
        if yrs < 1:
            stab_risk = 80.0
        elif yrs < 2:
            stab_risk = 50.0
        elif yrs < 5:
            stab_risk = 20.0
        else:
            stab_risk = 0.0
    else:
        stab_risk = 30.0  # neutral if not provided
    factors.append(FactorScore(
        name="employment_stability",
        label="Employment Stability",
        score=round(stab_risk, 1),
        weight=2,
        status=_status(stab_risk),
    ))

    # ── Co-applicant boost ───────────────────────────────────────────────────
    # If a co-applicant is present, reduce formula score proportionally.
    # Combined income improves EMI burden; co-applicant CIBIL is averaged in.
    co_income = getattr(data, "co_applicant_income", None) or 0.0
    co_cibil = getattr(data, "co_applicant_cibil_score", None)

    if co_income > 0:
        # Effective combined income lowers EMI-burden and savings-buffer risk
        combined_income = data.monthly_income + co_income
        eti_combined = safe_divide(data.emi_amount, combined_income)
        savings_combined = combined_income - data.monthly_expenses - data.emi_amount
        savings_ratio_combined = safe_divide(savings_combined, combined_income)

        for f in factors:
            if f.name == "emi_burden":
                f.score = round(clamp(eti_combined * 200.0, 0.0, 100.0), 1)
                f.status = _status(f.score)
            if f.name == "savings_buffer":
                f.score = round(clamp((0.20 - savings_ratio_combined) * 200.0, 0.0, 100.0), 1)
                f.status = _status(f.score)

    if co_cibil is not None and data.cibil_score is not None:
        # Blend primary and co-applicant CIBIL (primary weighs more: 70/30)
        blended_cibil = round(0.70 * data.cibil_score + 0.30 * co_cibil)
        blended_risk = _cibil_to_risk(blended_cibil)
        for f in factors:
            if f.name == "credit_score":
                f.score = round(blended_risk, 1)
                f.label = f"CIBIL Score (Primary {data.cibil_score} / Co-app {co_cibil})"
                f.status = _status(f.score)

    formula_score = sum(f.score * f.weight / 100.0 for f in factors)
    return round(formula_score, 2), factors


def _status(score: float) -> str:
    """Map a factor risk score to a traffic-light status."""
    if score < 40:
        return "good"
    if score < 70:
        return "fair"
    return "poor"


# ── ML feature builder (matches training feature order) ──────────────────────

def _build_features(data: LoanRiskInput) -> np.ndarray:
    dti = safe_divide(data.existing_loans, data.monthly_income * 12)
    eti = safe_divide(data.emi_amount, data.monthly_income)
    savings_ratio = safe_divide(
        data.monthly_income - data.monthly_expenses - data.emi_amount,
        data.monthly_income,
    )
    loan_income_ratio = safe_divide(data.loan_amount_requested, data.monthly_income * 12)
    employment_map = {
        "salaried": 0, "self_employed": 1, "farmer": 2, "student": 3, "other": 4
    }
    emp_code = employment_map.get(data.employment_type.lower(), 4)
    return np.array([[
        data.monthly_income,
        data.monthly_expenses,
        data.existing_loans,
        data.emi_amount,
        data.repayment_history_score,
        data.loan_amount_requested,
        data.loan_tenure_months,
        dti, eti, savings_ratio, loan_income_ratio, emp_code,
    ]])


# ── Rule-based key-factor extractor ──────────────────────────────────────────

def _identify_key_factors(data: LoanRiskInput, factors: List[FactorScore]) -> List[str]:
    """Convert FactorScore objects into human-readable sentences."""
    messages: List[str] = []

    eti = safe_divide(data.emi_amount, data.monthly_income)
    dti = safe_divide(data.existing_loans, data.monthly_income * 12)
    savings = data.monthly_income - data.monthly_expenses - data.emi_amount

    if eti > 0.5:
        messages.append("Your EMI is more than 50% of your income, which is very high.")
    elif eti > 0.35:
        messages.append("Your EMI takes up a large portion of your monthly income.")

    if dti > 0.4:
        messages.append("You have high existing debt compared to your annual income.")

    if savings < 0:
        messages.append("Your monthly expenses exceed your income — you are spending more than you earn.")
    elif savings < data.monthly_income * 0.1:
        messages.append("You have very little monthly savings left after expenses and EMI.")

    if data.cibil_score is not None:
        if data.cibil_score < 650:
            messages.append(f"Your CIBIL score of {data.cibil_score} is low. Most banks require 700+.")
        elif data.cibil_score < 700:
            messages.append(f"Your CIBIL score of {data.cibil_score} is below the preferred 750 threshold.")
        elif data.cibil_score >= 750:
            messages.append(f"Your CIBIL score of {data.cibil_score} is excellent — banks will view this positively.")
    elif data.repayment_history_score < 50:
        messages.append("Your past loan repayment record is weak.")
    elif data.repayment_history_score < 70:
        messages.append("Your repayment history could be improved.")

    if data.age is not None and data.age > 58:
        messages.append(f"At age {data.age}, loan tenure options may be limited by banks.")
    if data.employment_stability_years is not None and data.employment_stability_years < 1:
        messages.append("Less than 1 year in current job is a red flag for most lenders.")

    if data.employment_type.lower() in ("farmer", "self_employed"):
        messages.append("Irregular income from self-employment or farming adds uncertainty.")

    lir = safe_divide(data.loan_amount_requested, data.monthly_income * 12)
    if lir > 5:
        messages.append("The loan amount requested is very large relative to your annual income.")

    co_income = getattr(data, "co_applicant_income", None) or 0.0
    if co_income > 0:
        messages.append(f"Co-applicant income of ₹{co_income:,.0f}/month strengthens your application.")

    if not messages:
        messages.append("Your overall financial indicators are within acceptable range.")

    return messages


# ── Explanation & recommendation generators ───────────────────────────────────

def _generate_explanation(score: float, factors: List[FactorScore]) -> str:
    category = score_to_category(score)
    poor = [f.label for f in factors if f.status == "poor"]

    if category == "Low":
        base = "Great news! Your financial profile looks healthy."
    elif category == "Medium":
        base = "Your loan application has moderate risk. A few areas need attention."
    else:
        base = "Your application currently shows high financial risk."

    if poor:
        base += f" Weak areas: {', '.join(poor)}."
    return base


def _generate_recommendation(score: float, factors: List[FactorScore]) -> str:
    category = score_to_category(score)
    if category == "Low":
        return "You are likely eligible for this loan. Maintain your good repayment habits."
    elif category == "Medium":
        return (
            "Consider reducing your existing EMI or increasing income before applying. "
            "A smaller loan amount may get approved more easily."
        )
    else:
        return (
            "We recommend reducing existing debts, improving your savings, "
            "and building a stronger repayment record before reapplying."
        )


# ── AI-powered personalised advice ───────────────────────────────────────────

def _get_ai_advice(data: LoanRiskInput, score: float, factors: List[FactorScore]) -> str | None:
    """
    Call Groq (via ai_service) for a personalised 2–3 sentence advice.
    Returns None on any error so the API can still respond without AI.
    """
    try:
        from app.services.ai_service import generate_response  # avoid circular import

        poor_labels = [f.label for f in factors if f.status == "poor"]
        fair_labels = [f.label for f in factors if f.status == "fair"]
        lang = getattr(data, "language", "en")
        lang_code = lang.value if hasattr(lang, "value") else str(lang)

        prompt = (
            f"Loan applicant financial summary:\n"
            f"- Monthly income: ₹{data.monthly_income:,.0f}\n"
            f"- Monthly expenses: ₹{data.monthly_expenses:,.0f}\n"
            f"- Current EMI: ₹{data.emi_amount:,.0f}\n"
            f"- Existing loans outstanding: ₹{data.existing_loans:,.0f}\n"
            f"- Loan amount requested: ₹{data.loan_amount_requested:,.0f} "
            f"over {data.loan_tenure_months} months\n"
            f"- Repayment history score: {data.repayment_history_score:.0f}/100\n"
            f"- Employment type: {data.employment_type}\n"
            f"- Overall risk score: {score:.0f}/100\n"
            f"- Poor factors: {', '.join(poor_labels) if poor_labels else 'None'}\n"
            f"- Fair factors: {', '.join(fair_labels) if fair_labels else 'None'}\n\n"
            f"Give 2–3 specific, practical tips to improve loan eligibility. "
            f"Use very simple words for a rural Indian user. Be concise and encouraging."
        )

        return generate_response(prompt, language=lang_code)
    except Exception as exc:
        logger.warning("AI advice generation failed: %s", exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def predict_risk(data: LoanRiskInput) -> LoanRiskOutput:
    """
    Run CIBIL-style formula (primary) + optional ML blend → full risk assessment.
    """
    # 1. Formula-based score and factor breakdown
    formula_score, factor_breakdown = _compute_cibil_factors(data)

    # 2. Optionally blend in ML model probability (40 % weight if available)
    ml_score: float | None = None
    try:
        model = _load_model()
        if model is not None:
            features = _build_features(data)
            prob = model.predict_proba(features)[0]
            ml_score = clamp(float(prob[1]) * 100.0, 0.0, 100.0)
    except Exception as exc:
        logger.debug("ML model skipped: %s", exc)

    if ml_score is not None:
        risk_score = round(0.60 * formula_score + 0.40 * ml_score, 2)
    else:
        risk_score = formula_score

    # 3. Derived ratios
    dti = safe_divide(data.existing_loans, data.monthly_income * 12.0)
    eti = safe_divide(data.emi_amount, data.monthly_income)

    # 4. Human-readable outputs
    key_factors = _identify_key_factors(data, factor_breakdown)
    explanation = _generate_explanation(risk_score, factor_breakdown)
    recommendation = _generate_recommendation(risk_score, factor_breakdown)

    # 5. AI advice (non-blocking – None if Groq unavailable)
    ai_advice = _get_ai_advice(data, risk_score, factor_breakdown)

    return LoanRiskOutput(
        risk_score=round(risk_score, 2),
        risk_category=RiskCategory(score_to_category(risk_score)),
        explanation=explanation,
        key_factors=key_factors,
        recommendation=recommendation,
        debt_to_income_ratio=round(dti, 4),
        emi_to_income_ratio=round(eti, 4),
        factor_breakdown=factor_breakdown,
        ai_advice=ai_advice,
    )
