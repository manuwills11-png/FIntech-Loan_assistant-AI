"""
Pydantic schemas for all API request/response models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────
# Shared enums
# ─────────────────────────────────────────────

class RiskCategory(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class SupportedLanguage(str, Enum):
    ENGLISH = "en"
    HINDI = "hi"
    TAMIL = "ta"
    TELUGU = "te"
    KANNADA = "kn"
    MARATHI = "mr"
    BENGALI = "bn"
    MALAYALAM = "ml"
    GUJARATI = "gu"
    URDU = "ur"
    PUNJABI = "pa"
    ASSAMESE = "as"
    ODIA = "or"


# ─────────────────────────────────────────────
# Loan Risk Prediction
# ─────────────────────────────────────────────

class LoanRiskInput(BaseModel):
    monthly_income: float = Field(..., ge=0, description="Monthly income in INR")
    monthly_expenses: float = Field(..., ge=0, description="Monthly expenses in INR")
    existing_loans: float = Field(..., ge=0, description="Total outstanding loan amount in INR")
    emi_amount: float = Field(..., ge=0, description="Current monthly EMI in INR")
    repayment_history_score: float = Field(
        default=75.0, ge=0, le=100,
        description="Repayment history score 0–100 (100 = perfect). Derived from CIBIL if provided."
    )
    loan_amount_requested: float = Field(..., gt=0, description="New loan amount requested in INR")
    loan_tenure_months: int = Field(..., gt=0, description="Requested loan tenure in months")
    employment_type: str = Field(
        default="salaried",
        description="Employment type: salaried | self_employed | farmer | student | other"
    )
    language: SupportedLanguage = Field(default=SupportedLanguage.ENGLISH)

    # New eligibility fields
    cibil_score: Optional[int] = Field(
        None, ge=300, le=900,
        description="CIBIL/credit score 300–900. If provided, overrides repayment_history_score."
    )
    age: Optional[int] = Field(None, ge=18, le=75, description="Applicant age in years")
    loan_purpose: Optional[str] = Field(
        None,
        description="Purpose: home | personal | business | education | vehicle | agriculture | other"
    )
    employment_stability_years: Optional[float] = Field(
        None, ge=0,
        description="Years in current job or business"
    )

    # Co-applicant fields
    co_applicant_income: Optional[float] = Field(
        None, ge=0,
        description="Co-applicant monthly income in INR (spouse/parent/sibling)"
    )
    co_applicant_employment_type: Optional[str] = Field(
        None,
        description="Co-applicant employment: salaried | self_employed | farmer | student | other"
    )
    co_applicant_cibil_score: Optional[int] = Field(
        None, ge=300, le=900,
        description="Co-applicant CIBIL score 300–900"
    )

    @field_validator("monthly_expenses")
    @classmethod
    def expenses_lt_income(cls, v: float, info: Any) -> float:
        return v


class FactorScore(BaseModel):
    """Single factor in the CIBIL-style risk breakdown."""
    name: str = Field(..., description="Internal factor identifier")
    label: str = Field(..., description="Human-readable factor name")
    score: float = Field(..., description="Factor risk score 0–100 (100 = worst)")
    weight: int = Field(..., description="Percentage weight in overall score (sums to 100)")
    status: str = Field(..., description="'good' | 'fair' | 'poor'")


class LoanRiskOutput(BaseModel):
    risk_score: float = Field(..., description="Risk score 0–100 (100 = highest risk)")
    risk_category: RiskCategory
    explanation: str = Field(..., description="Human-readable explanation in requested language")
    key_factors: List[str] = Field(..., description="Top contributing risk factors")
    recommendation: str = Field(..., description="Short actionable recommendation")
    debt_to_income_ratio: float
    emi_to_income_ratio: float
    factor_breakdown: List[FactorScore] = Field(
        default_factory=list,
        description="CIBIL-style per-factor risk scores with weights"
    )
    ai_advice: Optional[str] = Field(
        None, description="AI-generated personalised advice from Groq"
    )


# ─────────────────────────────────────────────
# Credit Simulator
# ─────────────────────────────────────────────

class SimulateInput(LoanRiskInput):
    """Same as LoanRiskInput – client modifies values to see updated risk."""
    pass


class SimulateOutput(BaseModel):
    original_score: float
    simulated_score: float
    score_delta: float
    risk_category: RiskCategory
    improvement_tips: List[str]


# ─────────────────────────────────────────────
# Chat / Conversational AI
# ─────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class UserFinancialContext(BaseModel):
    monthly_income: Optional[float] = None
    monthly_expenses: Optional[float] = None
    existing_loans: Optional[float] = None
    emi_amount: Optional[float] = None
    loan_amount_requested: Optional[float] = None
    loan_tenure_months: Optional[int] = None
    risk_score: Optional[float] = None
    risk_category: Optional[str] = None
    cibil_score: Optional[int] = None
    age: Optional[int] = None
    loan_purpose: Optional[str] = None
    employment_type: Optional[str] = None
    employment_stability_years: Optional[float] = None
    gold_weight_grams: Optional[float] = None
    gold_purity_karats: Optional[int] = None
    co_applicant_income: Optional[float] = None
    co_applicant_employment_type: Optional[str] = None
    co_applicant_cibil_score: Optional[int] = None


class ChatInput(BaseModel):
    message: Optional[str] = Field(None, description="Text message from user")
    language: SupportedLanguage = Field(default=SupportedLanguage.ENGLISH)
    conversation_history: List[ChatMessage] = Field(
        default_factory=list,
        description="Previous turns for context (up to 10)"
    )
    return_audio: bool = Field(default=False, description="Return TTS audio in response")
    user_context: Optional[UserFinancialContext] = Field(
        None, description="User's financial data from the app"
    )


class ChatOutput(BaseModel):
    reply: str = Field(..., description="AI response in user's language")
    detected_language: str
    translated_reply: Optional[str] = None
    audio_base64: Optional[str] = Field(None, description="Base64-encoded MP3 if return_audio=True")
    conversation_history: List[ChatMessage]
    actions: List[Any] = Field(
        default_factory=list,
        description="Structured UI actions the frontend should execute (navigate, prefill, submit, etc.)"
    )


# ─────────────────────────────────────────────
# Financial Roadmap
# ─────────────────────────────────────────────

class RoadmapInput(BaseModel):
    monthly_income: float = Field(..., gt=0)
    monthly_expenses: float = Field(..., ge=0)
    existing_loans: float = Field(..., ge=0)
    emi_amount: float = Field(..., ge=0)
    loan_amount_requested: float = Field(..., gt=0)
    loan_tenure_months: int = Field(..., gt=0)
    risk_score: Optional[float] = Field(None, description="Pre-computed risk score (0–100)")
    language: SupportedLanguage = Field(default=SupportedLanguage.ENGLISH)


class MonthlyPlanItem(BaseModel):
    month: int
    opening_balance: float
    emi_payment: float
    closing_balance: float


class RoadmapOutput(BaseModel):
    repayment_plan: List[MonthlyPlanItem]
    expense_reduction_tips: List[str]
    income_improvement_tips: List[str]
    summary: str = Field(..., description="Human-readable summary in user's language")
    total_interest_payable: float
    suggested_emi: float


# ─────────────────────────────────────────────
# Reminder System
# ─────────────────────────────────────────────

class ReminderInput(BaseModel):
    phone_number: str = Field(..., description="WhatsApp number with country code, e.g. +919876543210")
    message: str = Field(..., description="Reminder message text")
    remind_at: str = Field(
        ...,
        description="ISO-8601 datetime string, e.g. 2024-06-15T09:00:00"
    )
    repeat: Optional[str] = Field(
        None,
        description="Cron expression for repeating reminders, e.g. '0 9 * * *'"
    )


class ReminderOutput(BaseModel):
    job_id: str
    status: str
    remind_at: str
    message: str


# ─────────────────────────────────────────────
# WhatsApp Messaging
# ─────────────────────────────────────────────

class WhatsAppInput(BaseModel):
    to: str = Field(..., description="Recipient WhatsApp number, e.g. +919876543210")
    message: str = Field(..., description="Message body")


class WhatsAppOutput(BaseModel):
    sid: str
    status: str
    to: str


# ─────────────────────────────────────────────
# Document / OCR
# ─────────────────────────────────────────────

class Transaction(BaseModel):
    date: Optional[str] = None
    description: str
    amount: float
    type: str = Field(..., description="'credit' or 'debit'")


class DocumentExtractOutput(BaseModel):
    document_type: str = Field(
        default="financial",
        description="'pan_card' | 'aadhaar' | 'cibil_report' | 'salary_slip' | 'bank_statement'"
    )
    confidence: str = Field(..., description="'high' | 'medium' | 'low'")
    raw_text_preview: str = Field(..., description="First 500 chars of OCR output for verification")

    # Financial document fields
    estimated_income: Optional[float] = None
    monthly_expenses: Optional[float] = None
    transactions: List[Transaction] = Field(default_factory=list)

    # PAN card fields
    pan_number: Optional[str] = None
    pan_name: Optional[str] = None
    pan_dob: Optional[str] = None
    pan_father_name: Optional[str] = None

    # Aadhaar fields
    aadhaar_name: Optional[str] = None
    aadhaar_dob: Optional[str] = None
    aadhaar_gender: Optional[str] = None
    aadhaar_address: Optional[str] = None

    # Shared identity field
    age_from_dob: Optional[int] = None

    # CIBIL / credit report fields
    cibil_score: Optional[int] = None
    cibil_report_date: Optional[str] = None
    active_loans_count: Optional[int] = None
    total_outstanding: Optional[float] = None
    total_monthly_emi: Optional[float] = None
    credit_accounts: List[dict] = Field(default_factory=list)

    # Salary slip fields
    gross_salary: Optional[float] = None
    net_salary: Optional[float] = None
    basic_salary: Optional[float] = None
    employer_name: Optional[str] = None
    salary_month: Optional[str] = None
    total_deductions: Optional[float] = None


# ─────────────────────────────────────────────
# Demo Users
# ─────────────────────────────────────────────

class DemoUserOutput(BaseModel):
    user_id: str
    name: str
    profile: str
    loan_data: LoanRiskInput
    risk_result: LoanRiskOutput
