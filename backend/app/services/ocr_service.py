"""
OCR / Document Intelligence service.

Primary:  Gemini Vision API  (handles Hindi/English, complex layouts, tables)
Fallback: Tesseract OCR + regex parsers

Detects and parses:
  - PAN Card          → name, DOB, PAN number, father's name, age
  - Aadhaar Card      → name, DOB, gender, address, age
  - CIBIL Report      → credit score, active loans, total EMI, outstanding balance
  - Salary Slip       → gross/net/basic salary, employer, month, deductions
  - Bank Statement    → income estimate, expenses, transaction list
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from datetime import date
from typing import List, Optional, Tuple

from app.models.schemas import DocumentExtractOutput, Transaction
from app.utils.helpers import parse_indian_currency, parse_plain_numbers

logger = logging.getLogger(__name__)


# ── OCR engines ───────────────────────────────────────────────────────────────

def _check_tesseract() -> bool:
    try:
        import pytesseract  # type: ignore
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _extract_text_from_image(image_bytes: bytes) -> str:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
        image = Image.open(io.BytesIO(image_bytes))
        try:
            return pytesseract.image_to_string(image, lang="eng+hin").strip()
        except Exception:
            return pytesseract.image_to_string(image, lang="eng").strip()
    except Exception as exc:
        logger.error("Tesseract OCR failed: %s", exc)
        return ""


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        import PyPDF2  # type: ignore
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
        if len(text) > 50:
            return text
    except Exception as exc:
        logger.debug("PyPDF2 skipped: %s", exc)
    try:
        from pdf2image import convert_from_bytes  # type: ignore
        images = convert_from_bytes(pdf_bytes, dpi=200, fmt="jpeg")
        parts = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            parts.append(_extract_text_from_image(buf.getvalue()))
        return "\n".join(parts)
    except Exception as exc:
        logger.error("PDF OCR fallback failed: %s", exc)
        return ""


# ── Shared helpers ─────────────────────────────────────────────────────────────

_DOB_RE   = re.compile(r"\b(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b")
_PAN_RE   = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
_SCORE_RE = re.compile(r"\b([3-9]\d{2})\b")   # 300–999
_INR_RE   = re.compile(r"(?:₹|Rs\.?|INR)?\s*([\d,]+(?:\.\d{1,2})?)")


def _age_from_dob(dob_str: str) -> Optional[int]:
    try:
        d, m, y = re.split(r"[\/\-]", dob_str)
        born = date(int(y), int(m), int(d))
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    except Exception:
        return None


def _parse_inr(text: str) -> Optional[float]:
    m = _INR_RE.search(text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def _lines(text: str) -> List[str]:
    return [l.strip() for l in text.split("\n") if l.strip()]


# ── Document type detector ────────────────────────────────────────────────────

def _detect_type(text: str) -> str:
    u = text.upper()
    if any(k in u for k in ("INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER", "GOVT. OF INDIA")):
        return "pan_card"
    if _PAN_RE.search(u) and any(k in u for k in ("FATHER", "DATE OF BIRTH")):
        return "pan_card"
    if any(k in u for k in ("UNIQUE IDENTIFICATION", "AADHAAR", "UIDAI", "आधार")):
        return "aadhaar"
    if any(k in u for k in ("CREDIT SCORE", "CIBIL SCORE", "CREDIT INFORMATION", "EQUIFAX", "EXPERIAN",
                             "CREDIT REPORT", "CRIF HIGH MARK", "ACTIVE ACCOUNTS", "CREDIT SUMMARY")):
        return "cibil_report"
    if any(k in u for k in ("SALARY SLIP", "PAY SLIP", "PAYSLIP", "PAYROLL", "BASIC SALARY",
                             "GROSS SALARY", "NET PAY", "HRA", "PROVIDENT FUND", "PF DEDUCTION")):
        return "salary_slip"
    return "bank_statement"


# ── PAN Card parser ───────────────────────────────────────────────────────────

def _parse_pan(text: str) -> DocumentExtractOutput:
    u = text.upper()
    ls = _lines(text)

    pan_number = (m := _PAN_RE.search(u)) and m.group(1) or None
    dob        = (m := _DOB_RE.search(text)) and m.group(1) or None
    age        = _age_from_dob(dob) if dob else None

    name: Optional[str] = None
    father: Optional[str] = None
    skip_kw = ("INCOME TAX", "GOVT", "GOVERNMENT", "PERMANENT", "DEPARTMENT",
                "ACCOUNT", "NUMBER", "INDIA", "CARD")

    for i, line in enumerate(ls):
        lu = line.upper()
        if any(k in lu for k in skip_kw):
            continue
        if "FATHER" in lu or "पिता" in lu:
            after = re.split(r"[:/]", line, 1)
            val = after[1].strip() if len(after) > 1 and after[1].strip() else (
                ls[i + 1].strip() if i + 1 < len(ls) else None)
            if val:
                father = re.sub(r"[^A-Za-z\s]", "", val).strip().title() or None
            continue
        if _DOB_RE.search(line) or any(k in lu for k in ("DATE", "BIRTH", "DOB")):
            continue
        if pan_number and pan_number in lu:
            continue
        clean = re.sub(r"[^A-Za-z\s]", "", line).strip()
        if clean and len(clean) > 2 and clean.replace(" ", "").isalpha():
            if name is None:
                name = clean.title()
            elif father is None:
                father = clean.title()

    confidence = "high" if pan_number else ("medium" if name else "low")
    return DocumentExtractOutput(
        document_type="pan_card",
        confidence=confidence,
        raw_text_preview=text[:500],
        pan_number=pan_number,
        pan_name=name,
        pan_dob=dob,
        pan_father_name=father,
        age_from_dob=age,
        transactions=[],
    )


# ── Aadhaar parser ────────────────────────────────────────────────────────────

def _parse_aadhaar(text: str) -> DocumentExtractOutput:
    u = text.upper()
    ls = _lines(text)

    dob  = (m := _DOB_RE.search(text)) and m.group(1) or None
    age  = _age_from_dob(dob) if dob else None

    # Year of birth alternate format (some Aadhaar show only year)
    if not dob:
        ym = re.search(r"\bYOB\s*[:\-]?\s*(\d{4})\b", u)
        if ym:
            try:
                age = date.today().year - int(ym.group(1))
            except Exception:
                pass

    gender: Optional[str] = None
    if re.search(r"\bMALE\b", u):
        gender = "Male"
    elif re.search(r"\bFEMALE\b", u):
        gender = "Female"

    # Name: first non-header all-alpha line
    name: Optional[str] = None
    skip_kw = ("AADHAAR", "UIDAI", "UNIQUE", "IDENTIFICATION", "AUTHORITY",
                "INDIA", "GOVERNMENT", "ENROLLMENT")
    for line in ls:
        lu = line.upper()
        if any(k in lu for k in skip_kw):
            continue
        if _DOB_RE.search(line) or re.search(r"\b(MALE|FEMALE)\b", lu):
            continue
        if re.search(r"\d{4}\s\d{4}\s\d{4}", line):  # Aadhaar number
            continue
        clean = re.sub(r"[^A-Za-z\s]", "", line).strip()
        if clean and len(clean) > 2 and clean.replace(" ", "").isalpha():
            name = clean.title()
            break

    # Address: lines after name/gender/dob block
    address_lines = []
    collecting = False
    for line in ls:
        lu = line.upper()
        if any(k in lu for k in ("S/O", "W/O", "D/O", "C/O", "NEAR", "DIST",
                                  "VILLAGE", "HOUSE", "FLAT", "ROAD", "STREET",
                                  "NAGAR", "COLONY", "PIN")):
            collecting = True
        if collecting:
            address_lines.append(line)
        if len(address_lines) > 4:
            break
    address = ", ".join(address_lines) if address_lines else None

    confidence = "high" if name and dob else ("medium" if name else "low")
    return DocumentExtractOutput(
        document_type="aadhaar",
        confidence=confidence,
        raw_text_preview=text[:500],
        aadhaar_name=name,
        aadhaar_dob=dob,
        aadhaar_gender=gender,
        aadhaar_address=address,
        age_from_dob=age,
        transactions=[],
    )


# ── CIBIL / Credit Report parser ──────────────────────────────────────────────

def _parse_cibil(text: str) -> DocumentExtractOutput:
    u = text.upper()

    # Credit score — look for explicit label first, then standalone 3-digit 300–900
    score: Optional[int] = None
    for pattern in (
        r"(?:CIBIL|CREDIT|TRANSUNION)\s*(?:SCORE|RANK)\s*[:\-]?\s*(\d{3})",
        r"YOUR\s+(?:CREDIT\s+)?SCORE\s*(?:IS)?\s*[:\-]?\s*(\d{3})",
        r"SCORE\s*[:\-]\s*(\d{3})",
        r"\b((?:3[0-9]{2}|[4-8]\d{2}|900))\b",  # 300–900
    ):
        m = re.search(pattern, u)
        if m:
            v = int(m.group(1))
            if 300 <= v <= 900:
                score = v
                break

    # Report date
    report_date = (m := re.search(r"(?:REPORT\s+DATE|DATE\s+OF\s+REPORT|GENERATED\s+ON)[^\d]*(\d{2}[\/\-]\d{2}[\/\-]\d{4})", u)) and m.group(1) or None

    # Active loan accounts
    active_count: Optional[int] = None
    for pattern in (
        r"ACTIVE\s+ACCOUNTS?\s*[:\-]?\s*(\d+)",
        r"OPEN\s+ACCOUNTS?\s*[:\-]?\s*(\d+)",
        r"NO\.\s*OF\s+(?:ACTIVE\s+)?ACCOUNTS?\s*[:\-]?\s*(\d+)",
    ):
        m = re.search(pattern, u)
        if m:
            active_count = int(m.group(1))
            break

    # Total outstanding
    outstanding: Optional[float] = None
    for pattern in (
        r"TOTAL\s+(?:OUTSTANDING|BALANCE|AMOUNT\s+OUTSTANDING)[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"OUTSTANDING\s+BALANCE[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"CURRENT\s+BALANCE[^\d]*([\d,]+(?:\.\d{1,2})?)",
    ):
        m = re.search(pattern, u)
        if m:
            try:
                outstanding = float(m.group(1).replace(",", ""))
                break
            except ValueError:
                pass

    # Monthly EMI from report
    total_emi: Optional[float] = None
    emi_vals: List[float] = []
    for pattern in (
        r"(?:EMI|MONTHLY\s+INSTALMENT|MONTHLY\s+PAYMENT)[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"(?:REPAYMENT\s+AMOUNT)[^\d]*([\d,]+(?:\.\d{1,2})?)",
    ):
        for m in re.finditer(pattern, u):
            try:
                v = float(m.group(1).replace(",", ""))
                if v > 0:
                    emi_vals.append(v)
            except ValueError:
                pass
    if emi_vals:
        total_emi = round(sum(emi_vals), 2)

    # Individual credit accounts (loan type, bank, outstanding, EMI)
    accounts: List[dict] = []
    loan_kws = ("HOME LOAN", "PERSONAL LOAN", "CAR LOAN", "AUTO LOAN",
                 "EDUCATION LOAN", "BUSINESS LOAN", "CREDIT CARD",
                 "GOLD LOAN", "MICROFINANCE", "KISAN CREDIT")
    for kw in loan_kws:
        if kw in u:
            # Try to grab the surrounding line for amount
            idx = u.find(kw)
            snippet = text[max(0, idx - 20):idx + 120]
            nums = parse_indian_currency(snippet) or parse_plain_numbers(snippet)
            acc: dict = {"type": kw.title()}
            if nums:
                acc["outstanding"] = max(nums)
            accounts.append(acc)

    confidence = "high" if score else ("medium" if outstanding else "low")
    return DocumentExtractOutput(
        document_type="cibil_report",
        confidence=confidence,
        raw_text_preview=text[:500],
        cibil_score=score,
        cibil_report_date=report_date,
        active_loans_count=active_count,
        total_outstanding=outstanding,
        total_monthly_emi=total_emi,
        credit_accounts=accounts[:8],
        transactions=[],
        estimated_income=None,
    )


# ── Salary Slip parser ────────────────────────────────────────────────────────

def _parse_salary_slip(text: str) -> DocumentExtractOutput:
    u = text.upper()

    def _grab(patterns: List[str]) -> Optional[float]:
        for p in patterns:
            m = re.search(p, u)
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except ValueError:
                    pass
        return None

    gross = _grab([
        r"GROSS\s+(?:SALARY|PAY|EARNINGS?)[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"TOTAL\s+EARNINGS?[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"GROSS\s+AMOUNT[^\d]*([\d,]+(?:\.\d{1,2})?)",
    ])
    net = _grab([
        r"NET\s+(?:SALARY|PAY|TAKE.?HOME)[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"TAKE\s+HOME[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"NET\s+PAYABLE[^\d]*([\d,]+(?:\.\d{1,2})?)",
    ])
    basic = _grab([
        r"BASIC\s+(?:SALARY|PAY)[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"BASIC[^\d]*([\d,]+(?:\.\d{1,2})?)",
    ])
    deductions = _grab([
        r"TOTAL\s+DEDUCTIONS?[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"DEDUCTIONS?\s+TOTAL[^\d]*([\d,]+(?:\.\d{1,2})?)",
    ])

    # Employer name
    employer: Optional[str] = None
    for pattern in (
        r"(?:COMPANY|EMPLOYER|ORGANISATION|ORGANIZATION)[^\n:]*[:\-]\s*([A-Za-z0-9\s&\.]+)",
        r"^([A-Z][A-Za-z\s&\.]{5,50}(?:LTD|LIMITED|PVT|PRIVATE|INC|CORP|TECHNOLOGIES|SERVICES))",
    ):
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            employer = m.group(1).strip()
            break

    # Salary month
    months = ("JANUARY","FEBRUARY","MARCH","APRIL","MAY","JUNE",
              "JULY","AUGUST","SEPTEMBER","OCTOBER","NOVEMBER","DECEMBER")
    salary_month: Optional[str] = None
    for mon in months:
        if mon in u:
            yr_m = re.search(rf"{mon}[\s,\-]*(\d{{4}})", u)
            salary_month = f"{mon.title()} {yr_m.group(1)}" if yr_m else mon.title()
            break

    income = net or gross
    confidence = "high" if (gross and net) else ("medium" if income else "low")
    return DocumentExtractOutput(
        document_type="salary_slip",
        confidence=confidence,
        raw_text_preview=text[:500],
        gross_salary=gross,
        net_salary=net,
        basic_salary=basic,
        total_deductions=deductions,
        employer_name=employer,
        salary_month=salary_month,
        estimated_income=income,
        transactions=[],
    )


# ── Bank Statement parser ─────────────────────────────────────────────────────

def _parse_transactions(text: str) -> List[Transaction]:
    transactions: List[Transaction] = []
    date_re   = re.compile(r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4})\b", re.I)
    amount_re = re.compile(r"(?:₹|Rs\.?)?\s*([\d,]+(?:\.\d{1,2})?)")

    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 10:
            continue
        date_m  = date_re.search(line)
        amounts = parse_indian_currency(line) or parse_plain_numbers(line)
        if not amounts:
            continue
        amount = amounts[-1]
        if amount < 1:
            continue
        ll = line.lower()
        if any(k in ll for k in ("credit", "cr", "deposit", "salary", "neft cr", "upi cr")):
            txn_type = "credit"
        else:
            txn_type = "debit"
        desc = re.sub(date_re.pattern, "", line, flags=re.I)
        desc = re.sub(amount_re.pattern, "", desc).strip()
        desc = re.sub(r"\s+", " ", desc)[:80]
        transactions.append(Transaction(
            date=date_m.group(0) if date_m else None,
            description=desc or "Transaction",
            amount=amount,
            type=txn_type,
        ))
    return transactions[:50]


def _parse_bank_statement(text: str) -> DocumentExtractOutput:
    transactions = _parse_transactions(text)
    credits = [t.amount for t in transactions if t.type == "credit"]
    debits  = [t.amount for t in transactions if t.type == "debit"]
    income   = round(sum(credits), 2) if credits else None
    expenses = round(sum(debits),  2) if debits  else None

    # Try explicit salary line
    m = re.search(r"(?:salary|sal|wages|net pay|take.?home)[^\d]*([\d,]+(?:\.\d{1,2})?)", text, re.I)
    if m:
        try:
            income = float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    confidence = "high" if len(transactions) > 10 else ("medium" if transactions else "low")
    return DocumentExtractOutput(
        document_type="bank_statement",
        confidence=confidence,
        raw_text_preview=text[:500],
        estimated_income=income,
        monthly_expenses=expenses,
        transactions=transactions,
    )


# ── Gemini Vision extractor ───────────────────────────────────────────────────

_GEMINI_PROMPT = """You are a document OCR expert for Indian financial documents.
Carefully read every character visible in this image — including text in Hindi or other Indian languages.
Ignore any watermarks like SPECIMEN, MOCK, or FOR DEMO ONLY — extract the real data.

Identify the document type and return a single JSON object with the exact fields below.
Return ONLY valid JSON. No markdown fences, no explanation, no extra keys.

If document is AADHAAR CARD:
{"document_type":"aadhaar","name":"Rohan Sharma","dob":"15/06/1990","gender":"Male","aadhaar_number":"1234 5678 9012","address":"..."}

If document is PAN CARD:
{"document_type":"pan_card","name":"Rohan Sharma","father_name":"Suresh Sharma","dob":"15/05/1960","pan_number":"ABCPS1234D"}

If document is CIBIL / CREDIT REPORT:
{"document_type":"cibil_report","cibil_score":745,"report_date":"12/10/2023","total_accounts":10,"active_loans_count":3,"total_outstanding":500000,"total_monthly_emi":15000}

If document is SALARY SLIP / PAY SLIP:
{"document_type":"salary_slip","employer_name":"Innovatech Solutions Pvt Ltd","employee_name":"Rohan Sharma","salary_month":"September 2023","gross_salary":85000,"net_salary":75000,"basic_salary":40000,"hra":15000,"total_deductions":10000}

If document is BANK STATEMENT:
{"document_type":"bank_statement","bank_name":"HDFC Bank","account_holder":"Rohan Sharma","period":"Sep 2023","estimated_monthly_income":75000,"monthly_expenses":30000}
"""


def _safe_int(v) -> Optional[int]:
    try:
        return int(float(str(v).replace(",", ""))) if v is not None else None
    except (ValueError, TypeError):
        return None


def _safe_float(v) -> Optional[float]:
    try:
        return float(str(v).replace(",", "")) if v is not None else None
    except (ValueError, TypeError):
        return None


def _gemini_dict_to_output(data: dict) -> DocumentExtractOutput:
    doc_type = data.get("document_type", "financial")
    preview  = json.dumps(data, indent=2, ensure_ascii=False)[:500]

    if doc_type == "pan_card":
        dob = data.get("dob")
        return DocumentExtractOutput(
            document_type="pan_card", confidence="high", raw_text_preview=preview,
            pan_number=data.get("pan_number"), pan_name=data.get("name"),
            pan_dob=dob, pan_father_name=data.get("father_name"),
            age_from_dob=_age_from_dob(dob) if dob else None, transactions=[],
        )
    if doc_type == "aadhaar":
        dob = data.get("dob")
        return DocumentExtractOutput(
            document_type="aadhaar", confidence="high", raw_text_preview=preview,
            aadhaar_name=data.get("name"), aadhaar_dob=dob,
            aadhaar_gender=data.get("gender"), aadhaar_address=data.get("address"),
            age_from_dob=_age_from_dob(dob) if dob else None, transactions=[],
        )
    if doc_type == "cibil_report":
        active = _safe_int(data.get("active_loans_count") or data.get("total_accounts"))
        return DocumentExtractOutput(
            document_type="cibil_report", confidence="high", raw_text_preview=preview,
            cibil_score=_safe_int(data.get("cibil_score")),
            cibil_report_date=data.get("report_date"),
            active_loans_count=active,
            total_outstanding=_safe_float(data.get("total_outstanding")),
            total_monthly_emi=_safe_float(data.get("total_monthly_emi")),
            credit_accounts=[], transactions=[],
        )
    if doc_type == "salary_slip":
        gross = _safe_float(data.get("gross_salary"))
        net   = _safe_float(data.get("net_salary"))
        return DocumentExtractOutput(
            document_type="salary_slip", confidence="high", raw_text_preview=preview,
            gross_salary=gross, net_salary=net,
            basic_salary=_safe_float(data.get("basic_salary")),
            total_deductions=_safe_float(data.get("total_deductions")),
            employer_name=data.get("employer_name"),
            salary_month=data.get("salary_month"),
            estimated_income=net or gross, transactions=[],
        )
    if doc_type == "bank_statement":
        return DocumentExtractOutput(
            document_type="bank_statement", confidence="high", raw_text_preview=preview,
            estimated_income=_safe_float(data.get("estimated_monthly_income")),
            monthly_expenses=_safe_float(data.get("monthly_expenses")),
            transactions=[],
        )
    return DocumentExtractOutput(
        document_type="financial", confidence="medium",
        raw_text_preview=preview, transactions=[],
    )


def _extract_with_gemini(file_bytes: bytes, content_type: str) -> Optional[DocumentExtractOutput]:
    """Use Gemini 1.5 Flash Vision to extract structured document data."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return None
    try:
        import google.generativeai as genai  # type: ignore
        from PIL import Image  # type: ignore

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Convert bytes to PIL image (works for JPEG, PNG, WEBP, TIFF)
        if "pdf" in content_type.lower():
            # Convert first page of PDF to image
            try:
                from pdf2image import convert_from_bytes  # type: ignore
                pages = convert_from_bytes(file_bytes, dpi=200, fmt="jpeg", first_page=1, last_page=1)
                pil_img = pages[0] if pages else None
            except Exception:
                pil_img = None
        else:
            pil_img = Image.open(io.BytesIO(file_bytes))

        if pil_img is None:
            return None

        response = model.generate_content(
            [_GEMINI_PROMPT, pil_img],
            generation_config={"temperature": 0.05, "max_output_tokens": 1024},
        )

        raw = response.text.strip()
        # Strip markdown code fences if model added them
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE).strip()

        data = json.loads(raw)
        logger.info("Gemini Vision extracted document_type=%s", data.get("document_type"))
        return _gemini_dict_to_output(data)

    except Exception as exc:
        logger.warning("Gemini Vision extraction failed: %s", exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def extract_document(file_bytes: bytes, content_type: str) -> DocumentExtractOutput:
    # ── 1. Try Gemini Vision first (far superior for complex Indian docs) ──────
    gemini_result = _extract_with_gemini(file_bytes, content_type)
    if gemini_result and gemini_result.confidence != "low":
        return gemini_result

    # ── 2. Fall back to Tesseract + regex ─────────────────────────────────────
    if not _check_tesseract():
        return DocumentExtractOutput(
            document_type="financial",
            confidence="low",
            raw_text_preview=(
                "Could not extract document data. "
                "Set GEMINI_API_KEY for AI-powered extraction, or install Tesseract OCR."
            ),
            transactions=[],
        )

    if "pdf" in content_type.lower():
        raw_text = _extract_text_from_pdf(file_bytes)
    else:
        raw_text = _extract_text_from_image(file_bytes)

    if not raw_text:
        return DocumentExtractOutput(
            document_type="financial",
            confidence="low",
            raw_text_preview="No text could be extracted from this document.",
            transactions=[],
        )

    doc_type = _detect_type(raw_text)
    logger.info("Tesseract fallback — document detected as: %s", doc_type)

    if doc_type == "pan_card":
        return _parse_pan(raw_text)
    if doc_type == "aadhaar":
        return _parse_aadhaar(raw_text)
    if doc_type == "cibil_report":
        return _parse_cibil(raw_text)
    if doc_type == "salary_slip":
        return _parse_salary_slip(raw_text)
    return _parse_bank_statement(raw_text)
