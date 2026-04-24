"""
FinEdge Loan – FastAPI Application Entry Point
===============================================

AI-Powered Multilingual Financial Assistant

Features:
  • Loan risk prediction (ML model)
  • Explainable AI decisions
  • Conversational AI (Gemini / OpenAI / fallback)
  • Voice input (Google STT) + voice output (Google TTS)
  • Multilingual support (EN/HI/TA/TE/KN/MR/BN)
  • Financial roadmap generation
  • Credit simulator
  • Document OCR (Google Vision)
  • WhatsApp reminders (Twilio)
  • APScheduler for background tasks
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("finedge")

# ── Routes ────────────────────────────────────────────────────────────────────
from app.routes import predict, chat, simulate, roadmap, reminder, whatsapp, document, demo, bank_rates, translate
from app.scheduler.reminder_scheduler import get_scheduler, shutdown_scheduler


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting FinEdge Financial Assistant API...")
    get_scheduler()  # initialise APScheduler
    logger.info("APScheduler initialised.")
    yield
    # Shutdown
    logger.info("Shutting down FinEdge API...")
    shutdown_scheduler()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinEdge – AI Financial Assistant API",
    description=(
        "An AI-powered multilingual financial assistant that predicts loan risk, "
        "explains decisions in simple language, analyzes user documents, and "
        "guides users through voice and WhatsApp — making financial intelligence "
        "accessible to everyone."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
_allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
_allow_all = _allowed_origins_raw.strip() == "*"
_allowed_origins = (
    ["*"] if _allow_all
    else [o.strip() for o in _allowed_origins_raw.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=not _allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(predict.router)
app.include_router(chat.router)
app.include_router(simulate.router)
app.include_router(roadmap.router)
app.include_router(reminder.router)
app.include_router(whatsapp.router)
app.include_router(document.router)
app.include_router(demo.router)
app.include_router(bank_rates.router)
app.include_router(translate.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "FinEdge AI Financial Assistant",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check including optional service availability."""
    services = {}

    # Check ML model
    model_path = os.getenv("MODEL_PATH", "./ml/loan_risk_model.pkl")
    services["ml_model"] = "available" if os.path.exists(model_path) else "not_found"

    # Check LLM config
    if os.getenv("GEMINI_API_KEY"):
        services["llm"] = "gemini"
    elif os.getenv("OPENAI_API_KEY"):
        services["llm"] = "openai"
    else:
        services["llm"] = "rule_based_fallback"

    # Check Twilio
    services["whatsapp"] = (
        "configured"
        if os.getenv("TWILIO_ACCOUNT_SID")
        else "simulated"
    )

    # Check Google Cloud
    services["google_cloud"] = (
        "configured"
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        else "not_configured"
    )

    return {
        "status": "healthy",
        "services": services,
    }


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )
