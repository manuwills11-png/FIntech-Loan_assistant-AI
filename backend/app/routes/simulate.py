"""
POST /simulate  – Credit simulator endpoint.

Allows users to modify loan parameters and instantly see
how the risk score changes, enabling "what-if" scenarios.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import LoanRiskInput, RiskCategory, SimulateInput, SimulateOutput
from app.services import risk_service
from app.utils.helpers import score_to_category

router = APIRouter(prefix="/simulate", tags=["Credit Simulator"])


def _improvement_tips(original: float, simulated: float, data: SimulateInput) -> list[str]:
    """Generate contextual improvement tips based on simulated changes."""
    tips = []
    delta = original - simulated  # positive = improvement

    if delta > 10:
        tips.append("Great improvement! Continuing this path will increase your loan eligibility.")
    elif delta > 0:
        tips.append("Slight improvement. Keep optimising your financial inputs.")
    elif delta == 0:
        tips.append("No change in risk. Try adjusting income, EMI, or loan amount.")
    else:
        tips.append("Risk increased. Consider requesting a smaller loan amount or reducing EMI.")

    # Contextual tips
    eti = data.emi_amount / max(data.monthly_income, 1)
    if eti > 0.5:
        tips.append("Your EMI-to-income ratio is above 50%. Reducing EMI is the fastest way to lower risk.")
    elif eti > 0.35:
        tips.append("Try to bring your EMI below 35% of monthly income.")

    savings = data.monthly_income - data.monthly_expenses - data.emi_amount
    if savings < 0:
        tips.append("You are spending more than you earn. Cutting non-essential expenses is critical.")

    if data.repayment_history_score < 70:
        tips.append("Improving your repayment history score above 70 will significantly reduce risk.")

    return tips[:5]


@router.post("", response_model=SimulateOutput, summary="Simulate risk with modified inputs")
async def simulate(data: SimulateInput) -> SimulateOutput:
    """
    Run a 'what-if' simulation.

    Modify any input parameter (income, EMI, loan amount, etc.)
    and instantly see the updated risk score and delta from a baseline.

    The baseline is computed with the same inputs minus any
    beneficial modifications the user is exploring.
    """
    try:
        simulated_result = risk_service.predict_risk(data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}")

    # Compute a conservative baseline with no optimism:
    # baseline = same data but with repayment_history_score at 50 (neutral)
    baseline_data = data.model_copy(update={"repayment_history_score": 50.0})
    try:
        baseline_result = risk_service.predict_risk(baseline_data)
        original_score = baseline_result.risk_score
    except Exception:
        original_score = simulated_result.risk_score  # fallback

    tips = _improvement_tips(original_score, simulated_result.risk_score, data)

    return SimulateOutput(
        original_score=round(original_score, 2),
        simulated_score=round(simulated_result.risk_score, 2),
        score_delta=round(original_score - simulated_result.risk_score, 2),
        risk_category=RiskCategory(score_to_category(simulated_result.risk_score)),
        improvement_tips=tips,
    )
