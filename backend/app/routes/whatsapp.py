"""
POST /send-whatsapp            – Send an immediate WhatsApp message via Twilio.
GET  /send-whatsapp/join-link  – Return the wa.me link for sandbox activation.
"""

import os

from fastapi import APIRouter, HTTPException

from app.models.schemas import WhatsAppInput, WhatsAppOutput
from app.services.whatsapp_service import send_whatsapp_message

router = APIRouter(prefix="/send-whatsapp", tags=["WhatsApp"])


@router.get("/join-link", summary="Get the Twilio sandbox WhatsApp join link")
async def get_join_link():
    """
    Returns a wa.me deep-link that pre-fills the Twilio sandbox join message.
    The user taps this link, WhatsApp opens with the message ready — they just hit Send.
    This is a one-time step per phone number.
    """
    sandbox_word = os.getenv("TWILIO_SANDBOX_WORD", "")
    sandbox_number = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    # Strip "whatsapp:" prefix and "+" to get digits only for wa.me
    number_digits = sandbox_number.replace("whatsapp:", "").replace("+", "")

    if not sandbox_word or sandbox_word == "your-sandbox-word":
        return {
            "configured": False,
            "link": None,
            "sandbox_number": sandbox_number.replace("whatsapp:", ""),
            "message": "TWILIO_SANDBOX_WORD not set in .env",
        }

    join_text = f"join {sandbox_word}"
    link = f"https://wa.me/{number_digits}?text={join_text.replace(' ', '%20')}"
    return {
        "configured": True,
        "link": link,
        "sandbox_number": sandbox_number.replace("whatsapp:", ""),
        "join_text": join_text,
    }


@router.post("", response_model=WhatsAppOutput, summary="Send a WhatsApp message")
async def send_whatsapp(data: WhatsAppInput) -> WhatsAppOutput:
    """
    Send an immediate WhatsApp message via Twilio.

    The `to` number must be registered on WhatsApp and opted into
    the Twilio sandbox during development.

    Example message:
        "Reminder: Your EMI of ₹2,000 is due tomorrow. Pay on time to protect your credit score."
    """
    if not data.to.startswith("+"):
        raise HTTPException(
            status_code=400,
            detail="Phone number must include country code, e.g. +919876543210",
        )
    if len(data.message.strip()) < 5:
        raise HTTPException(status_code=400, detail="Message is too short.")

    try:
        result = send_whatsapp_message(to=data.to, message=data.message)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return WhatsAppOutput(
        sid=result["sid"],
        status=result["status"],
        to=result["to"],
    )
