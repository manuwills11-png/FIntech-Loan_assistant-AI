"""
AI / LLM service for conversational responses.

Priority order:
  1. Groq (if GROQ_API_KEY is set)           ← fastest, free tier
  2. Google Gemini (if GEMINI_API_KEY is set)
  3. OpenAI (if OPENAI_API_KEY is set)
  4. Rule-based fallback (always works without any API key)
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from app.models.schemas import ChatMessage
from app.utils.helpers import build_system_prompt

logger = logging.getLogger(__name__)

# ── Groq ─────────────────────────────────────────────────────────────────────

def _call_groq(
    user_message: str,
    history: List[ChatMessage],
    language: str,
) -> str:
    """Call Groq API (OpenAI-compatible). Uses llama-3.3-70b-versatile by default."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")

    from groq import Groq  # type: ignore

    client = Groq(api_key=api_key)
    messages = [{"role": "system", "content": build_system_prompt(language)}]

    for msg in history[-10:]:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": user_message})

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=400,
        timeout=20,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ── Gemini ────────────────────────────────────────────────────────────────────

def _call_gemini(
    user_message: str,
    history: List[ChatMessage],
    language: str,
) -> str:
    """Call Google Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=build_system_prompt(language),
    )

    # Convert history to Gemini format
    gemini_history = []
    for msg in history[-10:]:  # keep last 10 turns
        role = "user" if msg.role == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg.content]})

    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(user_message)
    return response.text.strip()


# ── OpenAI ────────────────────────────────────────────────────────────────────

def _call_openai(
    user_message: str,
    history: List[ChatMessage],
    language: str,
) -> str:
    """Call OpenAI Chat Completions API."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key)
    messages = [{"role": "system", "content": build_system_prompt(language)}]

    for msg in history[-10:]:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=512,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ── Rule-based fallback ───────────────────────────────────────────────────────

_FALLBACK_RESPONSES = {
    "loan": (
        "A loan should be taken only when necessary. "
        "Make sure your EMI is less than 40% of your monthly income. "
        "Pay on time to build a good credit score."
    ),
    "risk": (
        "Your risk score depends on income, existing loans, repayment history, "
        "and how much you want to borrow. A lower score means lower risk."
    ),
    "emi": (
        "EMI stands for Equated Monthly Installment. It is the fixed amount "
        "you pay every month to repay your loan. Try to keep it below 40% of income."
    ),
    "income": (
        "Increasing your income — through additional work or skills — can "
        "significantly improve your loan eligibility and reduce financial risk."
    ),
    "save": (
        "Try to save at least 10–20% of your monthly income. Small savings "
        "built consistently help you handle emergencies and reduce loan dependency."
    ),
    "default": (
        "I'm your FinEdge financial assistant. I can help you understand your loan "
        "risk, plan repayments, and make better financial decisions. "
        "Please ask me anything about loans, savings, or your financial health!"
    ),
}


def _rule_based_fallback(user_message: str) -> str:
    msg_lower = user_message.lower()
    for keyword, response in _FALLBACK_RESPONSES.items():
        if keyword != "default" and keyword in msg_lower:
            return response
    return _FALLBACK_RESPONSES["default"]


# ── Public API ────────────────────────────────────────────────────────────────

def generate_response(
    user_message: str,
    history: Optional[List[ChatMessage]] = None,
    language: str = "en",
) -> str:
    """
    Generate a conversational AI response.

    Tries Groq → Gemini → OpenAI → rule-based fallback in order.

    Args:
        user_message: The user's message (always in English at this point).
        history:      Previous conversation turns.
        language:     Target response language code.

    Returns:
        Assistant's reply string.
    """
    history = history or []

    # 1. Try Groq (fastest, free tier)
    try:
        return _call_groq(user_message, history, language)
    except EnvironmentError:
        pass  # API key not set
    except Exception as exc:
        logger.warning("Groq call failed: %s", exc)

    # 2. Try Gemini
    try:
        return _call_gemini(user_message, history, language)
    except EnvironmentError:
        pass  # API key not set
    except Exception as exc:
        logger.warning("Gemini call failed: %s", exc)

    # 3. Try OpenAI
    try:
        return _call_openai(user_message, history, language)
    except EnvironmentError:
        pass  # API key not set
    except Exception as exc:
        logger.warning("OpenAI call failed: %s", exc)

    # 4. Rule-based fallback
    logger.info("Using rule-based fallback for AI response.")
    return _rule_based_fallback(user_message)


def generate_roadmap_narrative(roadmap_data: dict, language: str = "en") -> str:
    """
    Use LLM to write a plain-language explanation of the financial roadmap.
    """
    prompt = (
        f"Based on this financial roadmap data: {roadmap_data}, "
        "write a simple, encouraging explanation (5–7 sentences) that helps "
        "a rural user understand their repayment plan, key tips to reduce expenses, "
        "and how to improve income. Use very simple words."
    )
    return generate_response(prompt, language=language)
