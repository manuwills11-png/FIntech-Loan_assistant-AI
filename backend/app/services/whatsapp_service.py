"""
WhatsApp messaging via Twilio WhatsApp API.

Setup (one-time for sandbox):
  1. Recipient sends "join <your-sandbox-word>" to +14155238886 on WhatsApp
  2. Add TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM to .env
  3. Messages will now send for real

Production: upgrade to a Twilio WhatsApp approved sender (no join needed).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _get_credentials():
    sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    token = os.getenv("TWILIO_AUTH_TOKEN", "")
    from_number = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    return sid, token, from_number


def send_whatsapp_message(to: str, message: str) -> dict:
    """
    Send a WhatsApp message via Twilio.

    Args:
        to:      Recipient number with country code, e.g. '+919876543210'
        message: Message text
    """
    sid, token, from_number = _get_credentials()

    # Normalise the recipient number to whatsapp:+91... format
    to = to.replace("whatsapp:", "").strip()
    if not to.startswith("+"):
        to = f"+91{to}"
    to_wa = f"whatsapp:{to}"

    if not sid or not token:
        logger.warning("Twilio not configured — simulating send to %s", to)
        logger.warning("Message: %s", message)
        return {"sid": "SIMULATED", "status": "simulated", "to": to}

    try:
        from twilio.rest import Client  # type: ignore

        client = Client(sid, token)
        msg = client.messages.create(
            body=message,
            from_=from_number,
            to=to_wa,
        )
        logger.info("WhatsApp sent to %s, sid=%s, status=%s", to, msg.sid, msg.status)
        return {"sid": msg.sid, "status": msg.status, "to": to}

    except Exception as exc:
        logger.error("Twilio WhatsApp failed: %s", exc)
        raise RuntimeError(f"WhatsApp failed: {exc}") from exc


def build_emi_reminder_message(
    user_name: str,
    emi_amount: float,
    due_date: str,
    bank_name: Optional[str] = None,
) -> str:
    bank_info = f" to {bank_name}" if bank_name else ""
    return (
        f"Hi {user_name}! 👋\n\n"
        f"Reminder: Your EMI of ₹{emi_amount:,.0f}{bank_info} is due on {due_date}.\n\n"
        f"💡 Paying on time improves your credit score.\n\n"
        f"- FinEdge Financial Assistant 🤖"
    )


def build_nudge_message(tip: str, user_name: Optional[str] = None) -> str:
    greeting = f"Hi {user_name}! " if user_name else "Hello! "
    return (
        f"{greeting}💡 FinEdge Tip:\n\n{tip}\n\n- FinEdge AI 🤖"
    )
