"""
Shared utility helpers used across services and routes.
"""

from __future__ import annotations

import re
from typing import List


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a float value between lo and hi."""
    return max(lo, min(hi, value))


def score_to_category(score: float) -> str:
    """Map a 0–100 risk score to Low / Medium / High."""
    if score < 40:
        return "Low"
    if score < 70:
        return "Medium"
    return "High"


def parse_indian_currency(text: str) -> List[float]:
    """
    Extract numeric currency amounts from OCR text.
    Handles Indian formats like ₹1,23,456.78 or Rs 12000
    """
    # Remove currency symbols and parse
    pattern = r"(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d{1,2})?)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    amounts = []
    for m in matches:
        try:
            amounts.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return amounts


def parse_plain_numbers(text: str) -> List[float]:
    """Extract standalone numbers (possibly with commas) from text."""
    pattern = r"\b\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?\b"
    matches = re.findall(pattern, text)
    results = []
    for m in matches:
        try:
            results.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return results


def build_system_prompt(language: str = "en") -> str:
    """
    Return a system prompt that instructs the LLM to behave as a
    simple, empathetic financial assistant for rural/non-technical users.
    """
    lang_name_map = {
        "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
        "kn": "Kannada", "mr": "Marathi", "bn": "Bengali", "ml": "Malayalam",
        "gu": "Gujarati", "ur": "Urdu", "pa": "Punjabi", "as": "Assamese",
        "or": "Odia",
    }
    lang_name = lang_name_map.get(language, "the user's language")
    return (
        f"You are FinEdge, a friendly and empathetic AI financial assistant "
        f"helping rural and semi-urban users in India understand their loan "
        f"risk and financial health. "
        f"Always respond in {lang_name}. "
        f"Use very simple words. Avoid jargon. "
        f"Be encouraging, not scary. "
        f"When the user shares financial data, give specific, actionable advice. "
        f"Keep answers short (3–5 sentences unless a plan is requested)."
    )


def format_inr(amount: float) -> str:
    """Format float as Indian Rupee string."""
    return f"₹{amount:,.2f}"


def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Division that returns fallback instead of raising ZeroDivisionError."""
    if denominator == 0:
        return fallback
    return numerator / denominator
