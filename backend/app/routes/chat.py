"""
POST /chat        – Agentic conversational AI.
POST /chat/voice  – Voice-based chat (STT → agent → TTS).

Agent loop with tool calling:
  SERVER-SIDE tools (executed here, result fed back to AI):
    • assess_loan_risk   – runs ML risk model
    • generate_roadmap   – builds repayment schedule
    • get_loan_schemes   – returns matching govt/bank loan schemes
    • contact_bank       – drafts formal inquiry + AI-simulated bank response

  CLIENT-SIDE tools (returned as actions to frontend):
    • save_user_data     – persists collected values into the app store
    • show_risk_result   – navigates to loan-risk page with data + auto-submit
    • navigate_to_page   – go to any page
    • save_phone         – store WhatsApp number
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.models.schemas import (
    ChatInput, ChatMessage, ChatOutput, LoanRiskInput,
    SupportedLanguage, UserFinancialContext,
)
from app.services import speech_service, tts_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Conversational AI"])

MAX_ITER = 5

LANG_NAMES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "mr": "Marathi", "bn": "Bengali", "ml": "Malayalam",
    "gu": "Gujarati", "ur": "Urdu", "pa": "Punjabi", "as": "Assamese", "or": "Odia",
}

# ── Tool schemas ───────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_user_data",
            "description": (
                "Save the financial values the user just provided into the app. "
                "Call this IMMEDIATELY whenever the user gives any financial numbers or personal details. "
                "Only include fields the user actually mentioned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "income":                    {"type": "number", "description": "Monthly income in ₹"},
                    "expenses":                  {"type": "number", "description": "Monthly expenses in ₹"},
                    "existing_loans":            {"type": "number", "description": "Total outstanding loans in ₹"},
                    "emi":                       {"type": "number", "description": "Current monthly EMI in ₹"},
                    "loan_amount":               {"type": "number", "description": "Loan amount requested in ₹"},
                    "tenure_months":             {"type": "integer", "description": "Loan repayment period in months"},
                    "employment_type":           {"type": "string", "description": "salaried | self_employed | farmer | student | other"},
                    "cibil_score":               {"type": "integer", "description": "CIBIL/credit score 300–900"},
                    "age":                       {"type": "integer", "description": "Applicant age in years"},
                    "loan_purpose":              {"type": "string", "description": "home | personal | business | vehicle | education | gold | agriculture | other"},
                    "employment_stability_years":{"type": "number", "description": "Years in current job or business"},
                    "gold_weight_grams":         {"type": "number", "description": "Weight of gold ornaments/coins in grams (gold loans only)"},
                    "gold_purity_karats":        {"type": "integer", "description": "Gold purity: 18, 22, or 24 karats (gold loans only)"},
                    "co_applicant_income":        {"type": "number", "description": "Co-applicant monthly income in ₹ (spouse/parent/sibling)"},
                    "co_applicant_employment_type":{"type": "string", "description": "Co-applicant employment type"},
                    "co_applicant_cibil_score":   {"type": "integer", "description": "Co-applicant CIBIL score 300–900"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_loan_risk",
            "description": (
                "Run the full loan risk assessment AND show the result on screen. "
                "Call this as soon as you have income, expenses, existing_loans, emi, loan_amount, tenure_months. "
                "Pass any additional known fields (cibil_score, age, loan_purpose, employment_stability_years)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "income":                    {"type": "number"},
                    "expenses":                  {"type": "number"},
                    "existing_loans":            {"type": "number"},
                    "emi":                       {"type": "number"},
                    "loan_amount":               {"type": "number"},
                    "tenure_months":             {"type": "integer"},
                    "employment_type":           {"type": "string"},
                    "cibil_score":               {"type": "integer"},
                    "age":                       {"type": "integer"},
                    "loan_purpose":              {"type": "string"},
                    "employment_stability_years":{"type": "number"},
                    "co_applicant_income":       {"type": "number"},
                    "co_applicant_employment_type":{"type": "string"},
                    "co_applicant_cibil_score":  {"type": "integer"},
                },
                "required": ["income", "expenses", "existing_loans", "emi", "loan_amount", "tenure_months"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_roadmap",
            "description": "Generate a month-by-month loan repayment plan. Call after assess_loan_risk or when user asks for repayment plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "income":         {"type": "number"},
                    "expenses":       {"type": "number"},
                    "existing_loans": {"type": "number"},
                    "emi":            {"type": "number"},
                    "loan_amount":    {"type": "number"},
                    "tenure_months":  {"type": "integer"},
                    "risk_score":     {"type": "number"},
                },
                "required": ["income", "expenses", "loan_amount", "tenure_months"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_loan_schemes",
            "description": "Fetch relevant government and bank loan schemes for the user. Call when user asks about available loans, schemes, or subsidies, or which bank to approach.",
            "parameters": {
                "type": "object",
                "properties": {
                    "employment_type": {"type": "string"},
                    "loan_purpose":    {"type": "string", "description": "home | business | agriculture | education | personal | vehicle"},
                    "income":          {"type": "number"},
                },
                "required": ["employment_type", "loan_purpose"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "contact_bank",
            "description": (
                "Draft and send a formal loan inquiry to a bank on behalf of the user. "
                "Translates the inquiry to English (for the bank) and back to the user's language. "
                "Returns the bank's simulated response. "
                "Use when the user asks to 'contact a bank', 'apply to a bank', or 'send my details to a bank'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bank_name":       {"type": "string"},
                    "loan_purpose":    {"type": "string"},
                    "loan_amount":     {"type": "number"},
                    "income":          {"type": "number"},
                    "tenure_months":   {"type": "integer"},
                    "employment_type": {"type": "string"},
                    "risk_score":      {"type": "number"},
                },
                "required": ["bank_name", "loan_purpose", "loan_amount", "income"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_to_page",
            "description": "Navigate to a page. ONLY call in 4 cases: (1) after assess_loan_risk completes and user asks to see bank rates, (2) right after generate_roadmap completes, (3) gold loan details are ready for calculator, (4) user EXPLICITLY asks to go to a page. NEVER call for informational questions or general advice.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {"type": "string", "enum": ["loan-risk", "roadmap", "simulate", "reminders", "documents", "bank-rates", "gold-loan"]},
                },
                "required": ["page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_phone",
            "description": "Save user's WhatsApp phone number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {"type": "string"},
                },
                "required": ["phone"],
            },
        },
    },
]


# ── Server-side tool executors ─────────────────────────────────────────────────

def _exec_assess_loan_risk(args: dict) -> str:
    try:
        from app.services.risk_service import predict_risk
        data = LoanRiskInput(
            monthly_income=float(args["income"]),
            monthly_expenses=float(args["expenses"]),
            existing_loans=float(args.get("existing_loans", 0)),
            emi_amount=float(args.get("emi", 0)),
            loan_amount_requested=float(args["loan_amount"]),
            loan_tenure_months=int(args["tenure_months"]),
            employment_type=args.get("employment_type", "salaried"),
            language=SupportedLanguage.ENGLISH,
            cibil_score=int(args["cibil_score"]) if args.get("cibil_score") else None,
            age=int(args["age"]) if args.get("age") else None,
            loan_purpose=args.get("loan_purpose"),
            employment_stability_years=float(args["employment_stability_years"]) if args.get("employment_stability_years") else None,
            co_applicant_income=float(args["co_applicant_income"]) if args.get("co_applicant_income") else None,
            co_applicant_employment_type=args.get("co_applicant_employment_type"),
            co_applicant_cibil_score=int(args["co_applicant_cibil_score"]) if args.get("co_applicant_cibil_score") else None,
        )
        r = predict_risk(data)
        poor = [f.label for f in r.factor_breakdown if f.status == "poor"]
        fair = [f.label for f in r.factor_breakdown if f.status == "fair"]
        return json.dumps({
            "risk_score": r.risk_score,
            "risk_category": r.risk_category,
            "explanation": r.explanation,
            "recommendation": r.recommendation,
            "poor_factors": poor,
            "fair_factors": fair,
            "debt_to_income_pct": round(r.debt_to_income_ratio * 100, 1),
            "emi_to_income_pct": round(r.emi_to_income_ratio * 100, 1),
        })
    except Exception as exc:
        logger.error("assess_loan_risk failed: %s", exc)
        return json.dumps({"error": str(exc)})


def _exec_generate_roadmap(args: dict) -> str:
    try:
        loan = float(args["loan_amount"])
        months = int(args["tenure_months"])
        risk_score = float(args.get("risk_score", 60))
        rate = 0.10 if risk_score < 40 else (0.14 if risk_score < 70 else 0.18)
        r = rate / 12
        emi = loan * r * (1 + r) ** months / ((1 + r) ** months - 1) if r > 0 else loan / months
        emi = round(emi, 2)
        income = float(args.get("income", 0))
        expenses = float(args.get("expenses", 0))
        return json.dumps({
            "monthly_emi": emi,
            "total_repayment": round(emi * months, 2),
            "total_interest": round(emi * months - loan, 2),
            "interest_rate_pa": f"{rate * 100:.0f}%",
            "monthly_savings_after_emi": round(income - expenses - emi, 2),
            "affordable": (income - expenses - emi) >= 0,
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _exec_get_loan_schemes(args: dict) -> str:
    employment = args.get("employment_type", "other").lower()
    purpose = args.get("loan_purpose", "personal").lower()
    schemes: list[dict] = []

    if employment == "farmer" or "agri" in purpose:
        schemes += [
            {"name": "Kisan Credit Card (KCC)", "bank": "All Banks / NABARD",
             "rate": "4–7% p.a.", "max": "₹3 lakh+",
             "note": "Subsidised crop loans. Govt pays part of interest."},
            {"name": "MUDRA Agri Loan", "bank": "RRBs / Cooperative Banks",
             "rate": "7–9% p.a.", "max": "₹10 lakh", "note": "No collateral below ₹1.6 lakh."},
        ]
    if "edu" in purpose:
        schemes += [
            {"name": "Vidya Lakshmi", "bank": "38+ banks via vidyalakshmi.co.in",
             "rate": "8.5–12% p.a.", "max": "₹40 lakh", "note": "Apply to multiple banks in one form."},
        ]
    if "home" in purpose:
        schemes += [
            {"name": "PMAY CLSS", "bank": "All HFCs / Banks",
             "rate": "6.5% effective after subsidy", "max": "₹12 lakh subsidy",
             "note": "Income < ₹18 lakh/yr. Govt gives upfront subsidy on interest."},
        ]
    if employment in ("self_employed",) or "busi" in purpose:
        schemes += [
            {"name": "MUDRA Loan (Tarun)", "bank": "All Banks / MFIs",
             "rate": "8–12% p.a.", "max": "₹10 lakh", "note": "No collateral. For micro/small businesses."},
            {"name": "CGTMSE", "bank": "Scheduled Commercial Banks",
             "rate": "10–13% p.a.", "max": "₹2 crore", "note": "Govt credit guarantee — no personal security needed."},
        ]
    if not schemes:
        schemes += [
            {"name": "SBI Personal Loan", "bank": "State Bank of India",
             "rate": "11–14% p.a.", "max": "₹20 lakh", "note": "Fast 24-hr approval for salaried."},
            {"name": "Jan Samarth Portal", "bank": "Govt of India",
             "rate": "Varies", "max": "Varies", "note": "Check eligibility for 13 govt schemes at jansamarth.in"},
        ]
    return json.dumps({"schemes": schemes[:4], "portal": "jansamarth.in"})


def _exec_contact_bank(args: dict, language: str) -> str:
    """
    Draft a formal loan inquiry for the specified bank,
    get an AI-simulated bank response, return both.
    """
    try:
        bank = args.get("bank_name", "SBI")
        purpose = args.get("loan_purpose", "personal")
        amount = args.get("loan_amount", 0)
        tenure = args.get("tenure_months", 24)
        income = args.get("income", 0)
        expenses = args.get("expenses", 0)
        employment = args.get("employment_type", "salaried")
        risk_score = args.get("risk_score")
        user_lang = args.get("user_language", language)

        # Draft formal inquiry (always in English for bank)
        inquiry = (
            f"Dear {bank} Loan Department,\n\n"
            f"I wish to apply for a {purpose} loan. My details:\n"
            f"- Monthly Income: ₹{income:,.0f}\n"
            f"- Monthly Expenses: ₹{expenses:,.0f}\n"
            f"- Employment: {employment.replace('_', ' ').title()}\n"
            f"- Loan Amount Requested: ₹{amount:,.0f}\n"
            f"- Preferred Tenure: {tenure} months\n"
        )
        if risk_score is not None:
            inquiry += f"- FinEdge Risk Score: {risk_score:.0f}/100\n"
        inquiry += "\nKindly advise on eligibility, interest rate, and next steps.\n\nThank you."

        # Simulate bank's response using AI
        api_key = os.getenv("GROQ_API_KEY", "")
        bank_reply = _simulate_bank_response(bank, purpose, income, amount, tenure, employment, risk_score, api_key)

        return json.dumps({
            "inquiry_sent_to": bank,
            "inquiry_text": inquiry,
            "bank_response": bank_reply,
            "user_language": user_lang,
            "status": "response_received",
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _simulate_bank_response(
    bank: str, purpose: str, income: float, amount: float,
    tenure: int, employment: str, risk_score: Optional[float], api_key: str,
) -> str:
    """Template-based bank response — no extra API call, saves tokens."""
    r = 0.14 / 12
    emi = round(amount * r * (1 + r) ** tenure / ((1 + r) ** tenure - 1), 0) if tenure else round(amount * 0.014, 0)
    emi_ratio = emi / income if income else 1
    rate = "11–12%" if (risk_score or 60) < 40 else ("12–14%" if (risk_score or 60) < 70 else "14–18%")

    if emi_ratio < 0.5:
        return (
            f"{bank} Loan Response: Based on your profile, your {purpose} loan application "
            f"of ₹{amount:,.0f} looks eligible. Estimated interest rate: {rate} p.a., "
            f"monthly EMI: ~₹{emi:,.0f} over {tenure} months. "
            f"Please visit your nearest {bank} branch with: Aadhaar, PAN card, "
            f"3 months' bank statements, salary slips or income proof."
        )
    else:
        return (
            f"{bank} Loan Response: Your loan request of ₹{amount:,.0f} has a high EMI-to-income ratio "
            f"(~₹{emi:,.0f}/month on ₹{income:,.0f} income). We suggest reducing the loan amount or "
            f"extending the tenure to improve eligibility. Indicative rate: {rate} p.a. "
            f"Visit your nearest {bank} branch with Aadhaar, PAN, and 3 months' bank statements."
        )


# ── Fallback parser for Groq's non-standard tool-call format ──────────────────

def _repair_json(raw: str) -> str:
    """Best-effort repair of Groq's malformed JSON (e.g. unclosed strings, trailing junk)."""
    # Remove any trailing non-JSON chars after the last }
    raw = raw.strip()
    last_brace = raw.rfind("}")
    if last_brace != -1:
        raw = raw[: last_brace + 1]
    # Fix unclosed string before closing brace: e.g. "en)} → "en"}
    raw = re.sub(r'"([^"]*)[^"}\]]+}$', r'"\1"}', raw)
    return raw


def _parse_failed_generation(text: str) -> list[dict]:
    results = []
    # Groq uses three formats:
    #   <function=name>{"arg": val}</function>      (standard)
    #   <function=name({"arg": val})</function>     (parens)
    #   <function=name{"arg": val}</function>       (no separator — seen with hyphenated page names)
    for match in re.finditer(
        r"<function=(\w+)>(\{.*?\})</function>"           # standard
        r"|<function=(\w+)\((\{.*?\})\)</function>"       # parens
        r"|<function=([\w-]+)(\{.*?\})</function>"        # no separator (hyphenated names)
        r"|<function=([\w-]+)>(\{.*?)(?:</function>|$)",  # unclosed tag
        text, re.DOTALL
    ):
        name = match.group(1) or match.group(3) or match.group(5) or match.group(7)
        raw  = match.group(2) or match.group(4) or match.group(6) or match.group(8) or ""
        for attempt in (raw, _repair_json(raw)):
            try:
                results.append({
                    "id": f"fallback_{name}_{len(results)}",
                    "name": name,
                    "args": json.loads(attempt),
                })
                break
            except json.JSONDecodeError:
                continue
    return results


def _extract_error_tool_calls(exc: Exception) -> list[dict]:
    try:
        body: dict = {}
        if hasattr(exc, "response") and exc.response is not None:
            body = exc.response.json()
        elif hasattr(exc, "body") and exc.body:
            body = exc.body if isinstance(exc.body, dict) else json.loads(exc.body)
        err = body.get("error", {})
        if err.get("code") == "tool_use_failed":
            return _parse_failed_generation(err.get("failed_generation", ""))
    except Exception:
        pass
    return []


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt(language: str, ctx: "UserFinancialContext | None" = None) -> str:
    lang_name = LANG_NAMES.get(language, "English")
    loan_type = (ctx.loan_purpose or "").lower() if ctx else ""

    # ── Known-values block ─────────────────────────────────────────────────────
    known_lines: list[str] = []
    has_risk = False
    if ctx:
        if ctx.monthly_income:                          known_lines.append(f"  income          = ₹{ctx.monthly_income:,.0f}")
        if ctx.monthly_expenses:                        known_lines.append(f"  expenses        = ₹{ctx.monthly_expenses:,.0f}")
        if ctx.existing_loans is not None:              known_lines.append(f"  existing_loans  = ₹{ctx.existing_loans:,.0f}")
        if ctx.emi_amount is not None:                  known_lines.append(f"  emi             = ₹{ctx.emi_amount:,.0f}")
        if ctx.loan_amount_requested:                   known_lines.append(f"  loan_amount     = ₹{ctx.loan_amount_requested:,.0f}")
        if ctx.loan_tenure_months:                      known_lines.append(f"  tenure          = {ctx.loan_tenure_months} months")
        if ctx.cibil_score:                             known_lines.append(f"  cibil_score     = {ctx.cibil_score}")
        if ctx.age:                                     known_lines.append(f"  age             = {ctx.age} yrs")
        if ctx.loan_purpose:                            known_lines.append(f"  loan_type       = {ctx.loan_purpose}")
        if ctx.employment_type:                         known_lines.append(f"  employment      = {ctx.employment_type}")
        if ctx.employment_stability_years:              known_lines.append(f"  stability       = {ctx.employment_stability_years} yrs")
        if ctx.gold_weight_grams:                       known_lines.append(f"  gold_weight     = {ctx.gold_weight_grams}g")
        if ctx.gold_purity_karats:                      known_lines.append(f"  gold_purity     = {ctx.gold_purity_karats}k")
        if ctx.risk_score is not None:
            known_lines.append(f"  risk_score      = {ctx.risk_score:.0f}/100 ({ctx.risk_category})")
            has_risk = True

    known_block = ""
    if known_lines:
        known_block = "\n⚠️  SAVED USER DATA — DO NOT ASK FOR THESE AGAIN:\n" + "\n".join(known_lines) + "\n"

    header = (
        f"You are FinEdge — a proactive AI loan advisor for India. "
        f"Respond in {lang_name}. Keep replies to 2–3 sentences max. Never repeat yourself.\n"
        f"Use tools immediately — do NOT describe what you will do, just call the tool.\n"
        f"{known_block}\n"
    )
    footer = (
        "\nGENERAL RULES:\n"
        "- Call save_user_data IMMEDIATELY whenever the user gives ANY number or personal detail.\n"
        "- 'yes/ok/sure/do it/proceed/go ahead' after an offer → execute that offer NOW without re-asking.\n"
        "- Never explain what a loan is unless explicitly asked.\n"
        "- For general questions (how to improve CIBIL, what is FOIR, interest rate queries, etc.) → answer directly in text. Do NOT call navigate_to_page.\n"
        "- navigate_to_page is ONLY for completed actions (risk result ready, roadmap generated, gold calculator ready) or explicit user navigation requests.\n"
    )

    # ── GOLD LOAN PATH ─────────────────────────────────────────────────────────
    if loan_type == "gold":
        gold_w = ctx.gold_weight_grams if ctx else None
        gold_p = ctx.gold_purity_karats if ctx else None
        if gold_w and gold_p:
            state = (
                "STATE: Gold loan details collected.\n"
                "- Tell user their gold calculator is ready and offer to open it.\n"
                "- If user says yes/ok/open/show → call navigate_to_page(page='gold-loan').\n"
                "- Answer any general gold loan questions normally without navigating."
            )
        else:
            missing_g = []
            if not gold_w: missing_g.append("total gold weight in grams")
            if not gold_p: missing_g.append("gold purity (18, 22, or 24 karat)")
            if not (ctx and ctx.loan_amount_requested): missing_g.append("how much loan do you need")
            state = (
                f"STATE: GOLD LOAN — ask for ONLY these (no income/CIBIL needed, gold is collateral):\n"
                f"  {', '.join(missing_g)}\n"
                f"- As soon as user gives values → call save_user_data with loan_purpose='gold' and gold details."
            )
        return header + state + footer

    # ── LOAN TYPE UNKNOWN ──────────────────────────────────────────────────────
    if not loan_type:
        state = (
            "STATE: Loan type not yet identified.\n"
            "- Ask the user: 'What type of loan are you looking for?'\n"
            "- Options to mention: Personal / Home Loan / Business / Vehicle (Car) / Education / Gold / Agriculture\n"
            "- As soon as they reply → call save_user_data with loan_purpose immediately.\n"
            "- If the user already mentioned a loan type in their very first message, extract it and call save_user_data right away."
        )
        return header + state + footer

    # ── STANDARD LOAN PATHS ────────────────────────────────────────────────────

    # Core fields required for each loan type (index matches has_* flags below)
    TYPE_CORE = {
        "personal":    ("monthly income", "monthly expenses", "total existing loans", "current EMI", "personal loan amount needed", "repayment period in months"),
        "home":        ("monthly income", "monthly expenses", "total existing loans", "current EMI", "home loan amount needed (typically 80% of property value)", "repayment period in months (max 360 months / 30 years)"),
        "business":    ("monthly business income/profit", "monthly business expenses", "total existing loans", "current EMI", "business loan amount needed", "repayment period in months"),
        "vehicle":     ("monthly income", "monthly expenses", "total existing loans", "current EMI", "vehicle on-road price (loan covers up to 90%)", "repayment period in months (max 84 months)"),
        "education":   ("co-applicant monthly income", "monthly expenses", "total existing loans", "current EMI", "total course fee (this is your loan amount)", "repayment period in months"),
        "agriculture": ("monthly/annual farm income", "monthly expenses", "total existing loans", "current EMI", "agriculture loan amount needed", "repayment period in months"),
    }
    TYPE_PROFILE = {
        "personal":    "CIBIL score (300–900), age, employment type (salaried / self-employed), years in current job",
        "home":        "CIBIL score, age, employment type, years in current job",
        "business":    "CIBIL score (if known), your age, how many years the business has been running, GST registered (yes/no)",
        "vehicle":     "CIBIL score, age, employment type (salaried / self-employed)",
        "education":   "CIBIL score of co-applicant, student's age, institute name and course",
        "agriculture": "acres of land owned, main crop type, do you have a Kisan Credit Card",
    }
    core_labels = TYPE_CORE.get(loan_type, TYPE_CORE["personal"])
    profile_ask = TYPE_PROFILE.get(loan_type, "CIBIL score, age, employment type")

    # Presence flags
    has_income  = bool(ctx and ctx.monthly_income)
    has_exp     = bool(ctx and ctx.monthly_expenses)
    has_loans   = ctx is not None and ctx.existing_loans is not None
    has_emi     = ctx is not None and ctx.emi_amount is not None
    has_amount  = bool(ctx and ctx.loan_amount_requested)
    has_tenure  = bool(ctx and ctx.loan_tenure_months)
    all_core    = has_income and has_exp and has_loans and has_emi and has_amount and has_tenure

    if has_risk:
        state = (
            f"STATE: Risk assessment complete ✓ ({ctx.risk_category} risk, score {ctx.risk_score:.0f}/100).\n"
            f"- All forms (Loan Risk, Roadmap, Bank Rates) are pre-filled with user's data.\n"
            f"- Answer any general financial questions normally without navigating.\n"
            f"- Only navigate when user explicitly requests: 'show bank rates' / 'open roadmap' / 'compare banks now' → then call navigate_to_page.\n"
            f"- If user wants a repayment schedule: call generate_roadmap (then navigate_to_page page='roadmap' automatically happens).\n"
            f"- User says 'contact bank' → call contact_bank.\n"
            f"- Do NOT ask for any data. Do NOT re-run assess_loan_risk unless user explicitly asks to recalculate.\n"
        )
    elif all_core:
        state = (
            f"STATE: All core {loan_type} loan data collected. Run the risk assessment NOW.\n"
            f"- Call assess_loan_risk immediately using the saved data above.\n"
            f"- Also include: cibil_score, age, loan_purpose='{loan_type}', employment_type, employment_stability_years if they are known.\n"
            f"- Do NOT ask for anything more before running the assessment.\n"
        )
    else:
        missing_core = [
            label for flag, label in zip(
                [has_income, has_exp, has_loans, has_emi, has_amount, has_tenure],
                core_labels
            ) if not flag
        ]
        state = (
            f"STATE: Collecting {loan_type.upper()} loan details.\n"
            f"  Core fields still needed: {', '.join(missing_core) if missing_core else '(all done!)'}\n"
            f"  Profile info to also collect: {profile_ask}\n"
            f"\n"
            f"- Ask for ALL missing core fields AND all profile fields in ONE single message.\n"
            f"- As soon as user gives ANY value → call save_user_data immediately.\n"
            f"- Once ALL 6 core values are present → call assess_loan_risk right away without asking for confirmation.\n"
        )

    return header + state + footer


# ── Extract financial values from conversation history ────────────────────────

# Matches a single INR value (with optional unit) and captures its char position
_NUMBER_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(crore|cr|lakh|lac|k)?",
    re.IGNORECASE,
)

_FIELD_KEYWORDS: list[tuple[list[str], str]] = [
    (["income", "earn", "salary", "make", "get paid", "per month"],   "income"),
    (["expense", "spend", "expenditure"],                              "expenses"),
    (["existing loan", "outstanding", "current loan", "total loan",
      "already owe", "debt"],                                          "existing_loans"),
    (["emi", "installment", "equated monthly"],                        "emi"),
    (["loan amount", "need a loan", "want a loan", "loan of",
      "need loan", "want loan", "require", "borrow", "loan for"],      "loan_amount"),
    (["tenure", "repayment period", "repayment", "months",
      "period", "duration", "years"],                                  "tenure_months"),
]


def _inr_value(digits: str, unit: Optional[str]) -> float:
    val = float(digits.replace(",", ""))
    if unit:
        u = unit.lower()
        if u in ("crore", "cr"):  return val * 1_00_00_000
        if u in ("lakh", "lac"):  return val * 1_00_000
        if u == "k":              return val * 1_000
    return val


def _extract_from_history(
    history: list[ChatMessage],
    current_message: str,
    ctx: "UserFinancialContext | None",
) -> "UserFinancialContext":
    """
    Scan all user messages to recover financial values the AI may have missed.
    For each number found, look at the surrounding words (±40 chars) for field keywords.
    Existing ctx values take precedence.
    """
    from app.models.schemas import UserFinancialContext as UFC

    extracted: dict[str, Optional[float]] = {
        "income": None, "expenses": None, "existing_loans": None,
        "emi": None, "loan_amount": None, "tenure_months": None,
    }

    user_texts = [m.content for m in history if m.role == "user"] + [current_message]

    for raw in user_texts:
        text = raw.lower()
        for m in _NUMBER_RE.finditer(text):
            val = _inr_value(m.group(1), m.group(2))
            if val <= 0:
                continue
            # Window of surrounding text to identify the field
            start = max(0, m.start() - 40)
            end   = min(len(text), m.end() + 40)
            window = text[start:end]

            for keywords, field in _FIELD_KEYWORDS:
                if any(kw in window for kw in keywords):
                    if extracted[field] is None:
                        extracted[field] = val
                    break

    merged = UFC(
        monthly_income=ctx.monthly_income if ctx else None,
        monthly_expenses=ctx.monthly_expenses if ctx else None,
        existing_loans=ctx.existing_loans if ctx else None,
        emi_amount=ctx.emi_amount if ctx else None,
        loan_amount_requested=ctx.loan_amount_requested if ctx else None,
        loan_tenure_months=ctx.loan_tenure_months if ctx else None,
        risk_score=ctx.risk_score if ctx else None,
        risk_category=ctx.risk_category if ctx else None,
        cibil_score=ctx.cibil_score if ctx else None,
        age=ctx.age if ctx else None,
        loan_purpose=ctx.loan_purpose if ctx else None,
        employment_type=ctx.employment_type if ctx else None,
        employment_stability_years=ctx.employment_stability_years if ctx else None,
        gold_weight_grams=ctx.gold_weight_grams if ctx else None,
        gold_purity_karats=ctx.gold_purity_karats if ctx else None,
    )
    if not merged.monthly_income        and extracted["income"]:                       merged.monthly_income = extracted["income"]
    if not merged.monthly_expenses      and extracted["expenses"]:                     merged.monthly_expenses = extracted["expenses"]
    if merged.existing_loans is None    and extracted["existing_loans"] is not None:   merged.existing_loans = extracted["existing_loans"]
    if merged.emi_amount is None        and extracted["emi"] is not None:              merged.emi_amount = extracted["emi"]
    if not merged.loan_amount_requested and extracted["loan_amount"]:                  merged.loan_amount_requested = extracted["loan_amount"]
    if not merged.loan_tenure_months    and extracted["tenure_months"]:                merged.loan_tenure_months = int(extracted["tenure_months"])

    return merged


# ── Agent loop ────────────────────────────────────────────────────────────────

def _run_agent(
    message: str,
    history: list[ChatMessage],
    language: str,
    ctx: "UserFinancialContext | None" = None,
) -> tuple[str, list[dict]]:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "I'm your FinEdge loan assistant. Please share your income, expenses, loan amount, and other details so I can assess your eligibility.", []

    # Recover any values the AI forgot to save in previous turns
    ctx = _extract_from_history(history, message, ctx)
    # Ensure ctx is always a mutable UFC so save_user_data can update it mid-loop
    if ctx is None:
        from app.models.schemas import UserFinancialContext as _UFC
        ctx = _UFC()

    try:
        from groq import Groq  # type: ignore
        client = Groq(api_key=api_key)

        messages: list[dict] = [{"role": "system", "content": _build_system_prompt(language, ctx)}]
        for msg in history[-6:]:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": message})

        model = os.getenv("GROQ_MODEL", "llama3-groq-70b-8192-tool-use-preview")
        client_actions: list[dict] = []

        for _iteration in range(MAX_ITER):
            raw_tool_calls: list[dict] = []
            msg_content = ""

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=400,
                    temperature=0.4,
                    timeout=30,
                )
                msg_obj = response.choices[0].message
                msg_content = msg_obj.content or ""

                if not msg_obj.tool_calls:
                    return msg_content.strip(), client_actions

                raw_tool_calls = [
                    {"id": tc.id, "name": tc.function.name,
                     "args": json.loads(tc.function.arguments)}
                    for tc in msg_obj.tool_calls
                ]

            except Exception as api_exc:
                fallback = _extract_error_tool_calls(api_exc)
                if not fallback:
                    raise
                logger.warning("Groq tool_use_failed — fallback parsed %d calls", len(fallback))
                raw_tool_calls = fallback

            # Append assistant message with tool calls to history
            messages.append({
                "role": "assistant",
                "content": msg_content or None,
                "tool_calls": [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                    for tc in raw_tool_calls
                ],
            })

            for tc in raw_tool_calls:
                name, args = tc["name"], tc["args"]

                # ── Server-side ────────────────────────────────────────────
                if name == "assess_loan_risk":
                    result_str = _exec_assess_loan_risk(args)
                    client_actions.append({
                        "prefill_loan_risk": {
                            "income":        str(args.get("income", "")),
                            "expenses":      str(args.get("expenses", "")),
                            "existingLoans": str(args.get("existing_loans", "0")),
                            "emi":           str(args.get("emi", "0")),
                            "loanAmount":    str(args.get("loan_amount", "")),
                            "tenure":        str(args.get("tenure_months", "")),
                        },
                        "navigate": "loan-risk",
                        "submit_loan_risk": True,
                    })

                elif name == "generate_roadmap":
                    result_str = _exec_generate_roadmap(args)
                    client_actions.append({"navigate": "roadmap"})

                elif name == "get_loan_schemes":
                    result_str = _exec_get_loan_schemes(args)

                elif name == "contact_bank":
                    result_str = _exec_contact_bank(args, language)

                # ── Client-side ────────────────────────────────────────────
                elif name == "save_user_data":
                    result_str = json.dumps({"status": "saved"})
                    # Also keep the server-side ctx up-to-date so subsequent iterations see these values
                    if "income"                  in args: ctx.monthly_income             = float(args["income"])
                    if "expenses"                in args: ctx.monthly_expenses            = float(args["expenses"])
                    if "existing_loans"          in args: ctx.existing_loans              = float(args["existing_loans"])
                    if "emi"                     in args: ctx.emi_amount                  = float(args["emi"])
                    if "loan_amount"             in args: ctx.loan_amount_requested       = float(args["loan_amount"])
                    if "tenure_months"           in args: ctx.loan_tenure_months          = int(args["tenure_months"])
                    if "cibil_score"             in args: ctx.cibil_score                 = int(args["cibil_score"])
                    if "age"                     in args: ctx.age                         = int(args["age"])
                    if "loan_purpose"            in args: ctx.loan_purpose                = args["loan_purpose"]
                    if "employment_type"         in args: ctx.employment_type             = args["employment_type"]
                    if "employment_stability_years" in args: ctx.employment_stability_years = float(args["employment_stability_years"])
                    if "gold_weight_grams"       in args: ctx.gold_weight_grams           = float(args["gold_weight_grams"])
                    if "gold_purity_karats"      in args: ctx.gold_purity_karats          = int(args["gold_purity_karats"])
                    client_actions.append({
                        "save_user_data": {
                            "income":           str(args["income"])                    if "income"                    in args else None,
                            "expenses":         str(args["expenses"])                  if "expenses"                  in args else None,
                            "existingLoans":    str(args["existing_loans"])            if "existing_loans"            in args else None,
                            "emi":              str(args["emi"])                       if "emi"                       in args else None,
                            "loanAmount":       str(args["loan_amount"])               if "loan_amount"               in args else None,
                            "tenure":           str(args["tenure_months"])             if "tenure_months"             in args else None,
                            "employmentType":   args.get("employment_type"),
                            "cibilScore":       str(args["cibil_score"])               if "cibil_score"               in args else None,
                            "age":              str(args["age"])                       if "age"                       in args else None,
                            "loanPurpose":      args.get("loan_purpose"),
                            "stabilityYears":   str(args["employment_stability_years"]) if "employment_stability_years" in args else None,
                            "goldWeightGrams":  str(args["gold_weight_grams"])         if "gold_weight_grams"         in args else None,
                            "goldPurityKarats": str(args["gold_purity_karats"])        if "gold_purity_karats"        in args else None,
                        }
                    })

                elif name == "navigate_to_page":
                    result_str = json.dumps({"status": "ok"})
                    client_actions.append({"navigate": args.get("page", "loan-risk")})

                elif name == "save_phone":
                    result_str = json.dumps({"status": "saved"})
                    client_actions.append({"set_phone": args.get("phone", "")})

                else:
                    result_str = json.dumps({"error": f"Unknown tool: {name}"})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

        logger.warning("Agent hit max iterations")
        return "Analysis complete. Please check the results on screen.", client_actions

    except Exception as exc:
        logger.error("Agent loop failed: %s", exc)
        return "Sorry, something went wrong. Please try again.", []


# ── Chat processing ───────────────────────────────────────────────────────────

def _process_chat(
    message: str,
    language: str,
    history: list[ChatMessage],
    return_audio: bool,
    ctx: "UserFinancialContext | None" = None,
) -> ChatOutput:
    reply, actions = _run_agent(message, history, language, ctx)

    updated_history = list(history) + [
        ChatMessage(role="user", content=message),
        ChatMessage(role="assistant", content=reply),
    ]
    updated_history = updated_history[-20:]

    audio_b64: Optional[str] = None
    if return_audio and reply:
        tts_lang = tts_service.language_code_for(language)
        audio_b64 = tts_service.synthesize_speech(reply, language_code=tts_lang)

    return ChatOutput(
        reply=reply,
        detected_language=language,
        translated_reply=None,
        audio_base64=audio_b64,
        conversation_history=updated_history,
        actions=actions,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("", response_model=ChatOutput, summary="Chat with FinEdge AI")
async def chat(data: ChatInput) -> ChatOutput:
    if not data.message or not data.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    return _process_chat(
        message=data.message,
        language=data.language.value,
        history=data.conversation_history,
        return_audio=data.return_audio,
        ctx=data.user_context,
    )


@router.post("/voice", summary="Chat with FinEdge AI (voice)")
async def chat_voice(
    audio: UploadFile = File(...),
    language: str = Form(default="en"),
    return_audio: bool = Form(default=True),
    conversation_history: str = Form(default="[]"),
    user_context: str = Form(default="{}"),
) -> JSONResponse:
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    transcribed = speech_service.transcribe_audio(audio_bytes=audio_bytes, language_code=language)
    if not transcribed:
        raise HTTPException(status_code=422, detail="Could not transcribe audio. Please speak clearly.")
    logger.info("Transcribed [%s]: %s", language, transcribed)

    try:
        history_data = json.loads(conversation_history)
        history = [ChatMessage(role=m["role"], content=m["content"]) for m in history_data]
    except Exception:
        history = []

    ctx: Optional[UserFinancialContext] = None
    try:
        ctx_data = json.loads(user_context)
        if ctx_data:
            ctx = UserFinancialContext(**ctx_data)
    except Exception:
        pass

    result = _process_chat(message=transcribed, language=language, history=history, return_audio=return_audio, ctx=ctx)
    return JSONResponse({"transcribed_text": transcribed, **result.model_dump()})


# ── Negotiation endpoint ───────────────────────────────────────────────────────

@router.post("/negotiate", summary="AI Loan Negotiation Coach")
async def negotiate(data: dict) -> dict:
    """
    Specialised conversational endpoint for loan negotiation coaching.
    Receives messages + borrower profile, returns AI negotiation advice.
    """
    messages: list[dict] = data.get("messages", [])
    profile: dict = data.get("profile", {})
    language: str = data.get("language", "en")
    lang_name = LANG_NAMES.get(language, "English")

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return {"reply": "Negotiation assistant is unavailable (no API key configured)."}

    system_prompt = (
        f"You are an expert loan negotiation coach helping an Indian borrower get better terms from a bank.\n"
        f"You know Indian banking regulations, RBI guidelines, MCLR/RLLR-based pricing, and how relationship managers work.\n\n"
        f"Borrower profile:\n"
        f"- Monthly Income: ₹{profile.get('income') or 'not provided'}\n"
        f"- Existing EMI: ₹{profile.get('existingEmi') or '0'}\n"
        f"- CIBIL Score: {profile.get('cibilScore') or 'not provided'}\n"
        f"- Loan Amount: ₹{profile.get('loanAmount') or 'not provided'}\n"
        f"- Tenure: {profile.get('tenure') or 'not provided'} months\n"
        f"- Purpose: {profile.get('loanPurpose') or 'personal'}\n"
        f"- Employment: {profile.get('employmentType') or 'salaried'}\n"
        f"- Bank's Offered Rate: {(profile.get('offeredRate') or 'not provided') + '%' if profile.get('offeredRate') else 'not provided'}\n"
        f"- Target Bank: {profile.get('targetBank') or 'not specified'}\n"
        f"- Competitor Rate: {(profile.get('competitorRate') or 'not provided') + '%' if profile.get('competitorRate') else 'not provided'}\n\n"
        f"Rules:\n"
        f"1. Give word-for-word scripts the borrower can say to the banker.\n"
        f"2. Quote real RBI policies (MCLR, RLLR, spread cap, prepayment rules) when relevant.\n"
        f"3. Tell them their real leverage — don't be overly optimistic.\n"
        f"4. If they have weak leverage, say so and suggest what to do first.\n"
        f"5. Keep responses concise and practical — max 250 words.\n"
        f"6. Respond in {lang_name}."
    )

    groq_messages = [{"role": "system", "content": system_prompt}]
    for m in messages[-10:]:  # last 10 messages for context
        role = m.get("role", "user")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            groq_messages.append({"role": role, "content": content})

    try:
        from groq import Groq  # type: ignore
        client = Groq(api_key=api_key)
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        response = client.chat.completions.create(
            model=model,
            messages=groq_messages,
            max_tokens=500,
            temperature=0.5,
            timeout=30,
        )
        reply = response.choices[0].message.content or "No response generated."
        return {"reply": reply.strip()}
    except Exception as exc:
        logger.error("Negotiation AI failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Negotiation AI error: {exc}")
