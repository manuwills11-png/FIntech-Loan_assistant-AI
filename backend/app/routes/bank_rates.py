"""
GET /bank-rates  – Bank loan rate comparison based on CIBIL score, loan amount, and purpose.
POST /bank-rates/contact  – Send a formal loan inquiry to a specific bank (returns AI-simulated response).
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/bank-rates", tags=["Bank Rate Comparison"])


# ── Data ──────────────────────────────────────────────────────────────────────

class BankRate(BaseModel):
    bank: str
    bank_short: str
    type: str           # "PSU" | "Private" | "NBFC" | "HFC"
    logo_color: str     # tailwind bg color class
    min_rate: float
    max_rate: float
    rate_for_score: float    # actual rate for the user's CIBIL
    max_loan: str
    max_tenure_years: int
    processing_fee: str
    eligible: bool
    eligibility_note: str
    recommended: bool
    rank: int            # lower = better for this user
    documents: List[str]
    features: List[str]

class BankRatesResponse(BaseModel):
    loan_purpose: str
    cibil_score: int
    loan_amount: float
    monthly_emi: float
    rates: List[BankRate]
    best_pick: str
    summary: str

class ContactBankRequest(BaseModel):
    bank: str
    loan_purpose: str
    loan_amount: float
    tenure_months: int
    income: float
    expenses: float
    existing_loans: float
    employment_type: str
    cibil_score: Optional[int] = None
    age: Optional[int] = None

class ContactBankResponse(BaseModel):
    bank: str
    inquiry_text: str
    bank_response: str
    estimated_rate: float
    estimated_emi: float
    next_steps: List[str]


# ── Rate tables per purpose ───────────────────────────────────────────────────
# Rates sourced from BankBazaar.com — updated April 2026
# Each bank entry: (name, short, type, color, base_rate, spread, max_loan, max_tenure, fee, docs, features)

_BANK_DATA = {
    "home": [
        # Source: bankbazaar.com/home-loan-interest-rate.html (Apr 2026)
        ("LIC Housing Finance",   "LICHFL","HFC",     "bg-green-700",  7.15, 4.85, "₹15 crore",      30, "Nil – ₹10,000",       ["Aadhaar","PAN","Income proof","Property docs"],                             ["Lowest home loan rate","No prepayment charges","Rural scheme"]),
        ("Bank of Baroda",        "BoB",   "PSU",     "bg-orange-700", 7.20, 1.70, "No upper limit",  30, "0.50% (max ₹15,000)", ["Aadhaar","PAN","Income proof","Property docs"],                             ["Min rate 7.20%","Rural branches","Baroda Home Loan app"]),
        ("State Bank of India",   "SBI",   "PSU",     "bg-blue-600",   7.25, 2.50, "No upper limit",  30, "0.35% (min ₹2,000)",  ["Aadhaar","PAN","Income proof","Property docs","Bank statements 6mo"],      ["SBI Regular & Maxgain","No prepayment penalty","Max tenure 30 yrs"]),
        ("Punjab National Bank",  "PNB",   "PSU",     "bg-red-700",    7.75, 7.00, "No upper limit",  30, "0.35%",               ["Aadhaar","PAN","Income proof","Property docs","Bank statements 6mo"],      ["Women's concessional rate","Low fee","Pradhan Mantri Awas integration"]),
        ("HDFC Bank",             "HDFC",  "Private", "bg-red-600",    7.75, 2.00, "No upper limit",  30, "0.50% (min ₹3,000)",  ["Aadhaar","PAN","Salary slips 3mo","Property docs","ITR 2 yrs"],            ["Quick approval 3–5 days","Digital process","Top-up loans available"]),
        ("Kotak Mahindra Bank",   "Kotak", "Private", "bg-red-500",    7.70, 4.30, "No upper limit",  20, "0.50%",               ["Aadhaar","PAN","Salary slips","Property docs"],                            ["ChoiceHome flexible EMI","Online process","Priority banking"]),
        ("ICICI Bank",            "ICICI", "Private", "bg-orange-500", 7.75, 2.25, "No upper limit",  30, "0.50%",               ["Aadhaar","PAN","Salary slips 3mo","Property docs","Bank statements 6mo"],  ["Online tracking","Step-up EMI option","Balance transfer"]),
        ("Axis Bank",             "Axis",  "Private", "bg-purple-600", 8.00, 6.00, "No upper limit",  30, "1% (min ₹10,000)",    ["Aadhaar","PAN","Salary slips","Property docs","ITR"],                      ["Fastest disbursal","Shubh Aarambh (0 EMI 3mo)","Digital-first"]),
    ],
    "personal": [
        # Source: bankbazaar.com/personal-loan-interest-rate.html (Apr 2026)
        ("HDFC Bank",        "HDFC",  "Private", "bg-red-600",    9.99,  14.01, "₹40 lakh",  5, "2.50% (max ₹25,000)", ["Aadhaar","PAN","Salary slips 3mo"],                        ["Pre-approved in seconds","Flexible tenure","No collateral"]),
        ("ICICI Bank",       "ICICI", "Private", "bg-orange-500", 9.99,  6.51,  "₹50 lakh",  6, "2.25%",               ["Aadhaar","PAN","Salary slips 3mo"],                        ["Instant disbursal","No foreclosure charges after 12mo"]),
        ("Axis Bank",        "Axis",  "Private", "bg-purple-600", 9.99,  12.01, "₹40 lakh",  5, "1.50%–2%",            ["Aadhaar","PAN","Salary proof"],                            ["Insta Personal Loan","Same-day disbursal"]),
        ("Bank of Baroda",   "BoB",   "PSU",     "bg-orange-700", 10.15, 5.85,  "₹10 lakh",  7, "2% (min ₹1,000)",     ["Aadhaar","PAN","Salary slips 3mo","Bank statements 3mo"], ["Govt bank","Low spread for good CIBIL","No collateral"]),
        ("Punjab National Bank","PNB","PSU",     "bg-red-700",    10.25, 1.00,  "₹20 lakh",  7, "1% + GST",            ["Aadhaar","PAN","Salary slips 3mo","Bank statements 3mo"], ["Tightest rate band","Govt bank security","Personal Need Loan"]),
        ("State Bank of India","SBI", "PSU",     "bg-blue-600",   10.30, 4.70,  "₹20 lakh",  7, "1% + GST",            ["Aadhaar","PAN","Salary slips 3mo","Bank statements 3mo"], ["Xpress Credit for salaried","No collateral","Lowest PSU rate"]),
        ("Kotak Mahindra Bank","Kotak","Private","bg-red-500",    10.99, 5.01,  "₹40 lakh",  5, "Up to 2.50%",         ["Aadhaar","PAN","Salary slips"],                            ["Pre-approved offers","Quick disbursal"]),
        ("Tata Capital",     "Tata",  "NBFC",    "bg-blue-900",   10.99, 13.01, "₹35 lakh",  6, "2.75%",               ["Aadhaar","PAN","Income proof"],                            ["Doorstep service","Flexible repayment","No prepayment after 6mo"]),
        ("Bajaj Finance",    "Bajaj", "NBFC",    "bg-blue-800",   13.00, 17.00, "₹35 lakh",  8, "Up to 3.93%",         ["Aadhaar","PAN"],                                           ["Minimal docs","Instant approval","Flexi loan option"]),
    ],
    "business": [
        # Source: bankbazaar.com/business-loan-interest-rate.html (Apr 2026)
        ("Bank of Baroda",    "BoB",   "PSU",     "bg-orange-700", 10.85, 2.15, "₹25 lakh",  10, "0.50%",    ["Aadhaar","PAN","ITR 2yrs","Business vintage proof"],                 ["CGTMSE coverage","Rural business focus","Low processing"]),
        ("State Bank of India","SBI",  "PSU",     "bg-blue-600",   10.90, 2.10, "₹50 lakh",   7, "1%",       ["Aadhaar","PAN","ITR 3yrs","Business proof","Bank statements 12mo"],  ["MUDRA Tarun","No collateral <₹10L","Govt-backed"]),
        ("Axis Bank",         "Axis",  "Private", "bg-purple-600", 11.05, 8.95, "₹75 lakh",   5, "2%",       ["Aadhaar","PAN","ITR 2yrs","Business proof","Bank statements 6mo"],   ["Secured term loans","Quick sanction","Digital-first"]),
        ("ICICI Bank",        "ICICI", "Private", "bg-orange-500", 11.50, 4.50, "₹2 crore",   7, "2%",       ["Aadhaar","PAN","ITR","Business proof","Bank statements"],             ["Instant Business Loan","No collateral up to ₹50L"]),
        ("HDFC Bank",         "HDFC",  "Private", "bg-red-600",    11.90, 4.10, "₹50 lakh",   4, "2%",       ["Aadhaar","PAN","ITR 2yrs","GST returns"],                             ["Business SmartXpress","Quick 4-hr disbursal"]),
        ("Bajaj Finance",     "Bajaj", "NBFC",    "bg-blue-800",   14.00, 8.00, "₹80 lakh",   8, "3%",       ["Aadhaar","PAN","Bank statements 6mo"],                               ["Minimal docs","Flexi loan","Line of credit"]),
    ],
    "education": [
        # Source: bankbazaar.com/education-loan-interest-rate.html (Apr 2026)
        ("Punjab National Bank",  "PNB",  "PSU",     "bg-red-700",    4.00,  6.10, "₹1 crore",   15, "Nil",   ["Aadhaar","PAN","Admission letter","Mark sheets","Co-applicant income"], ["Lowest rate 4% (priority sector)","Vidya Lakshmi portal","Tax deduction u/s 80E"]),
        ("Bank of Baroda",        "BoB",  "PSU",     "bg-orange-700", 6.85,  3.60, "₹1.5 crore", 15, "Nil",   ["Aadhaar","PAN","Admission letter","Mark sheets"],                        ["Baroda Scholar for abroad","Lowest PSU base rate","Simple Moratorium"]),
        ("State Bank of India",   "SBI",  "PSU",     "bg-blue-600",   6.90,  3.00, "₹1.5 crore", 15, "Nil",   ["Aadhaar","PAN","Admission letter","Mark sheets","Co-applicant income"], ["SBI Student Loan","Vidya Lakshmi portal","No margin <₹4L"]),
        ("Axis Bank",             "Axis", "Private", "bg-purple-600", 7.45,  5.55, "₹75 lakh",   15, "1.50%", ["Aadhaar","PAN","Admission letter","Co-applicant docs"],                  ["Fast approval","Abroad specialist","Digital process"]),
        ("ICICI Bank",            "ICICI","Private", "bg-orange-500", 8.50,  5.25, "₹1 crore",   10, "1%",    ["Aadhaar","PAN","Admission letter","Income proof","Mark sheets"],          ["Instant Education Loan","Abroad coverage","Partial disbursement"]),
        ("HDFC Credila",          "HDFC", "HFC",     "bg-red-600",    9.50,  4.50, "No limit",   15, "1%",    ["Aadhaar","PAN","Admission letter","Income proof"],                       ["100% tuition coverage","Abroad study specialist","No collateral <₹7.5L"]),
    ],
    "vehicle": [
        # Source: bankbazaar.com/car-loan-interest-rate.html (Apr 2026)
        ("Bank of Baroda",   "BoB",   "PSU",     "bg-orange-700", 7.60, 0.90, "No upper limit", 7, "0.50% (min ₹1,000)",  ["Aadhaar","PAN","Income proof","Vehicle invoice"],      ["Lowest car loan rate","New & used cars","Rural branches"]),
        ("Punjab National Bank","PNB","PSU",     "bg-red-700",    7.60, 2.05, "No upper limit", 7, "0.35% + GST",         ["Aadhaar","PAN","Income proof","Vehicle invoice"],      ["PNB Pride Car Loan","Tractor loans available","Low fee"]),
        ("State Bank of India","SBI", "PSU",     "bg-blue-600",   8.80, 1.20, "No upper limit", 7, "0.51% (min ₹1,000)",  ["Aadhaar","PAN","Income proof","Vehicle invoice"],      ["SBI Car Loan","Up to 90% on-road price","No prepayment charges"]),
        ("Axis Bank",        "Axis",  "Private", "bg-purple-600", 8.95, 1.05, "₹1 crore",       7, "1% (max ₹10,000)",    ["Aadhaar","PAN","Salary proof","Vehicle invoice"],      ["Quick approval","Flexible tenure","Digital-first"]),
        ("HDFC Bank",        "HDFC",  "Private", "bg-red-600",    9.00, 1.50, "No upper limit", 7, "0.50% (max ₹10,000)", ["Aadhaar","PAN","Salary slips","Vehicle invoice"],      ["Guaranteed approval in 30 min","Zero down payment for high CIBIL"]),
        ("ICICI Bank",       "ICICI", "Private", "bg-orange-500", 9.15, 1.00, "No upper limit", 7, "0.50%–1%",            ["Aadhaar","PAN","Income proof","Vehicle invoice"],      ["Instant car loan","Used car loans too","Digital process"]),
        ("Kotak Mahindra Bank","Kotak","Private","bg-red-500",    8.99, 1.51, "No upper limit", 7, "0.50%",               ["Aadhaar","PAN","Income proof"],                        ["Special rates for salaried","Online application"]),
    ],
    "agriculture": [
        # Source: NABARD & bank websites (Apr 2026)
        ("SBI",               "SBI",   "PSU",     "bg-blue-600",   4.00, 5.00, "₹3 lakh+",   5, "Nil",  ["Aadhaar","PAN","Land records","Kisan Credit Card"], ["KCC Scheme – 4% after subsidy","Revolving credit","NABARD-backed"]),
        ("Bank of Baroda",    "BoB",   "PSU",     "bg-orange-700", 7.00, 3.00, "₹10 lakh",   7, "Nil",  ["Aadhaar","PAN","Land records","Farm income proof"],  ["MUDRA Agri","No collateral <₹1.6L","Cooperative bank tie-ups"]),
        ("NABARD",            "NABARD","PSU",     "bg-green-700",  4.00, 4.00, "Varies",      7, "Nil",  ["Aadhaar","Land records","Farmer ID"],                ["Refinance channel","SHG loans","JLG scheme"]),
        ("Punjab National Bank","PNB", "PSU",     "bg-red-700",    7.00, 2.00, "₹25 lakh",   9, "Nil",  ["Aadhaar","PAN","Land records"],                      ["Kisan Tractor Loan","Allied agri activities"]),
    ],
    "gold": [
        # Source: bank websites (Apr 2026) — No CIBIL required, gold is collateral, max LTV 75% (RBI mandate)
        ("State Bank of India","SBI",  "PSU",     "bg-blue-600",   7.50, 2.25, "₹50 lakh",   3, "0.50%", ["Aadhaar","PAN","Gold ornaments/coins/bars"],        ["Lowest gold loan rate","Govt bank security","No prepayment penalty"]),
        ("Muthoot Finance",   "Muthot","NBFC",    "bg-yellow-600", 9.96, 14.04,"Based on gold value",3,"0%", ["Aadhaar","PAN","Gold ornaments/coins/bars"],     ["Fastest disbursal","Doorstep service","No income proof","LTV up to 75%"]),
        ("Manappuram Gold",   "Manppm","NBFC",    "bg-orange-600", 9.90, 14.10,"Based on gold value",1,"0%", ["Aadhaar","PAN","Gold ornaments"],               ["Bullet repayment option","No income proof","Online gold loan","Renew & release"]),
        ("HDFC Bank",         "HDFC",  "Private", "bg-red-600",    9.00,  8.00, "₹50 lakh",  2, "1%",    ["Aadhaar","PAN","Gold ornaments"],                   ["30-min disbursal","Online application","Flexible repayment"]),
        ("ICICI Bank",        "ICICI", "Private", "bg-orange-500", 9.00,  8.00, "₹1 crore",  1, "1%",    ["Aadhaar","PAN","Gold ornaments"],                   ["Instant digital gold loan","No income proof","Flexible repayment"]),
        ("Axis Bank",         "Axis",  "Private", "bg-purple-600", 9.05,  7.95, "₹25 lakh",  3, "0.50%", ["Aadhaar","PAN","Gold ornaments"],                   ["Quick disbursal","Low processing fee","Overdraft facility"]),
    ],
}

# For unknown purpose, fall back to personal
def _get_bank_data(purpose: str):
    return _BANK_DATA.get(purpose.lower(), _BANK_DATA["personal"])


# ── Rate calculator ───────────────────────────────────────────────────────────

def _rate_for_cibil(base: float, spread: float, cibil: int) -> float:
    """
    Give lower rate for better CIBIL.
    750+ → base, 700–749 → base+0.25, 650–699 → base+0.75, <650 → base+spread
    """
    if cibil >= 750:
        return base
    elif cibil >= 700:
        return round(base + 0.25, 2)
    elif cibil >= 650:
        return round(base + 0.75, 2)
    else:
        return round(base + spread, 2)


def _eligible(rate: float, max_rate: float, cibil: int, purpose: str) -> tuple[bool, str]:
    # Hard cutoffs per purpose type (gold = 0: no CIBIL required, gold is the collateral)
    cutoffs = {"home": 650, "education": 600, "agriculture": 500, "vehicle": 650, "gold": 0}
    min_cibil = cutoffs.get(purpose.lower(), 650)
    if cibil < min_cibil:
        return False, f"CIBIL score below {min_cibil} — not eligible for this bank. Improve credit score first."
    if rate > max_rate:
        return False, "Rate would exceed bank's maximum — apply after improving CIBIL."
    return True, "Eligible based on credit score."


def _emi(principal: float, annual_rate: float, months: int) -> float:
    r = annual_rate / 100 / 12
    if r == 0:
        return principal / months
    return round(principal * r * (1 + r) ** months / ((1 + r) ** months - 1), 2)


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.get("", response_model=BankRatesResponse, summary="Compare bank loan rates by CIBIL score")
async def get_bank_rates(
    cibil_score: int = 700,
    loan_amount: float = 500000,
    loan_purpose: str = "personal",
    tenure_months: int = 60,
):
    bank_data = _get_bank_data(loan_purpose)
    rates: list[BankRate] = []

    for name, short, btype, color, base, spread, max_loan, max_tenure_yr, fee, docs, features in bank_data:
        rate = _rate_for_cibil(base, spread, cibil_score)
        max_rate = base + spread
        eligible, note = _eligible(rate, max_rate, cibil_score, loan_purpose)
        emi = _emi(loan_amount, rate, tenure_months) if eligible else 0

        rates.append(BankRate(
            bank=name, bank_short=short, type=btype, logo_color=color,
            min_rate=base, max_rate=round(max_rate, 2),
            rate_for_score=rate,
            max_loan=max_loan, max_tenure_years=max_tenure_yr,
            processing_fee=fee,
            eligible=eligible, eligibility_note=note,
            recommended=False, rank=0,
            documents=docs, features=features,
        ))

    # Rank eligible banks by rate, then by type preference (PSU first for low income)
    eligible_banks = [r for r in rates if r.eligible]
    eligible_banks.sort(key=lambda r: (r.rate_for_score, 0 if r.type == "PSU" else 1))

    for i, b in enumerate(eligible_banks):
        b.rank = i + 1
        if i == 0:
            b.recommended = True

    best = eligible_banks[0].bank if eligible_banks else "No bank found for your profile"

    if cibil_score >= 750:
        summary = f"Excellent CIBIL score! You qualify for the best rates from {len(eligible_banks)} banks. {best} offers the lowest rate of {eligible_banks[0].rate_for_score:.2f}% p.a."
    elif cibil_score >= 700:
        summary = f"Good CIBIL score. {len(eligible_banks)} banks will lend to you. Improving to 750+ could save you ₹{int(_emi(loan_amount, eligible_banks[0].rate_for_score, tenure_months) - _emi(loan_amount, eligible_banks[0].min_rate, tenure_months)):,}/month on EMI."
    elif cibil_score >= 650:
        summary = f"Fair CIBIL score. {len(eligible_banks)} banks available but at higher rates. Consider improving your score before applying to save significantly on interest."
    else:
        summary = f"Low CIBIL score. Only {len(eligible_banks)} lender(s) may consider your application. Focus on clearing existing dues and paying EMIs on time for 6 months before applying."

    best_emi = _emi(loan_amount, eligible_banks[0].rate_for_score, tenure_months) if eligible_banks else 0

    return BankRatesResponse(
        loan_purpose=loan_purpose,
        cibil_score=cibil_score,
        loan_amount=loan_amount,
        monthly_emi=best_emi,
        rates=rates,
        best_pick=best,
        summary=summary,
    )


@router.post("/contact", response_model=ContactBankResponse, summary="Contact a specific bank for loan inquiry")
async def contact_bank(req: ContactBankRequest):
    bank_data = _get_bank_data(req.loan_purpose)
    # Find this bank's data
    entry = next((b for b in bank_data if b[1].lower() == req.bank.lower() or b[0].lower() == req.bank.lower()), None)

    base_rate = entry[4] if entry else 11.0
    spread    = entry[5] if entry else 5.0
    cibil     = req.cibil_score or 700
    rate      = _rate_for_cibil(base_rate, spread, cibil)
    bank_name = entry[0] if entry else req.bank
    docs      = entry[9] if entry else ["Aadhaar","PAN","Income proof","Bank statements"]

    emi = _emi(req.loan_amount, rate, req.tenure_months)

    inquiry = (
        f"Dear {bank_name} Loan Department,\n\n"
        f"I wish to apply for a {req.loan_purpose} loan.\n\n"
        f"My Details:\n"
        f"  • Monthly Income:     ₹{req.income:,.0f}\n"
        f"  • Monthly Expenses:   ₹{req.expenses:,.0f}\n"
        f"  • Existing Loans:     ₹{req.existing_loans:,.0f}\n"
        f"  • Employment:         {req.employment_type.replace('_',' ').title()}\n"
        f"  • Loan Requested:     ₹{req.loan_amount:,.0f}\n"
        f"  • Preferred Tenure:   {req.tenure_months} months\n"
    )
    if req.cibil_score:
        inquiry += f"  • CIBIL Score:        {req.cibil_score}\n"
    if req.age:
        inquiry += f"  • Age:                {req.age} years\n"
    inquiry += "\nKindly advise on eligibility, interest rate, and next steps.\n\nThank you."

    emi_ratio = emi / req.income if req.income else 1

    if emi_ratio < 0.4 and cibil >= 650:
        response = (
            f"Dear Applicant,\n\n"
            f"Thank you for your inquiry. Based on the information provided:\n\n"
            f"• Your application looks ELIGIBLE for a {req.loan_purpose} loan of ₹{req.loan_amount:,.0f}.\n"
            f"• Indicative interest rate: {rate:.2f}% p.a.\n"
            f"• Estimated monthly EMI: ₹{emi:,.0f} over {req.tenure_months} months.\n"
            f"• Your EMI-to-income ratio is {emi_ratio*100:.0f}% — within our acceptable limit of 50%.\n\n"
            f"Please visit your nearest {bank_name} branch with the following documents:\n"
            + "\n".join(f"  • {d}" for d in docs) +
            f"\n\nOur loan officer will guide you through the application process.\n\n"
            f"Warm regards,\n{bank_name} Loan Department"
        )
    else:
        suggestion = "reduce the loan amount or extend the tenure" if emi_ratio >= 0.4 else "improve your CIBIL score above 650"
        response = (
            f"Dear Applicant,\n\n"
            f"Thank you for your interest in {bank_name}.\n\n"
            f"Based on your profile, your application currently has some challenges:\n"
            f"• Estimated EMI: ₹{emi:,.0f}/month ({emi_ratio*100:.0f}% of income — our limit is 50%)\n"
            f"• CIBIL Score: {cibil} (minimum required: 650+)\n\n"
            f"We recommend you {suggestion} before applying.\n\n"
            f"Alternatively, please visit any {bank_name} branch — our officers can suggest "
            f"customised loan products that may suit your profile better.\n\n"
            f"Documents to bring:\n"
            + "\n".join(f"  • {d}" for d in docs) +
            f"\n\nWarm regards,\n{bank_name} Loan Department"
        )

    next_steps = [
        f"Visit nearest {bank_name} branch with above documents",
        "Fill the loan application form (also available online)",
        "Bank will verify documents and run credit check (1–3 days)",
        "Loan sanction letter issued if approved",
        "Disbursement within 3–7 working days after signing agreement",
    ]

    return ContactBankResponse(
        bank=bank_name,
        inquiry_text=inquiry,
        bank_response=response,
        estimated_rate=rate,
        estimated_emi=emi,
        next_steps=next_steps,
    )
