"""
Reminder scheduler using APScheduler.

Schedules one-time and recurring WhatsApp reminders.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from app.services.whatsapp_service import send_whatsapp_message

logger = logging.getLogger(__name__)

_scheduler: Optional[BackgroundScheduler] = None
_TIMEZONE = "Asia/Kolkata"


def get_scheduler() -> BackgroundScheduler:
    """Return the global scheduler instance (initialised on first call)."""
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler(timezone=_TIMEZONE)
        _scheduler.start()
        logger.info("APScheduler started (timezone: %s)", _TIMEZONE)
    return _scheduler


def _send_reminder_job(phone_number: str, message: str) -> None:
    """Callable executed by APScheduler when a reminder fires."""
    try:
        result = send_whatsapp_message(to=phone_number, message=message)
        logger.info("Reminder sent to %s – SID: %s", phone_number, result.get("sid"))
    except Exception as exc:
        logger.error("Failed to send reminder to %s: %s", phone_number, exc)


def schedule_one_time_reminder(
    phone_number: str,
    message: str,
    run_at: datetime,
    job_id: Optional[str] = None,
) -> str:
    """
    Schedule a single WhatsApp reminder at a specific datetime.

    Args:
        phone_number: Recipient WhatsApp number.
        message:      Reminder message text.
        run_at:       When to send the reminder (timezone-aware preferred).
        job_id:       Optional custom job ID; auto-generated if not provided.

    Returns:
        The job_id string.
    """
    scheduler = get_scheduler()

    # Ensure datetime is timezone-aware
    if run_at.tzinfo is None:
        run_at = run_at.replace(tzinfo=ZoneInfo(_TIMEZONE))

    _id = job_id or f"reminder_{phone_number}_{run_at.timestamp():.0f}"
    scheduler.add_job(
        _send_reminder_job,
        trigger=DateTrigger(run_date=run_at),
        args=[phone_number, message],
        id=_id,
        replace_existing=True,
    )
    logger.info("Scheduled one-time reminder '%s' for %s at %s", _id, phone_number, run_at)
    return _id


def schedule_recurring_reminder(
    phone_number: str,
    message: str,
    cron_expression: str,
    job_id: Optional[str] = None,
) -> str:
    """
    Schedule a recurring WhatsApp reminder using a cron expression.

    Args:
        phone_number:     Recipient WhatsApp number.
        message:          Reminder message text.
        cron_expression:  Standard cron expression e.g. '0 9 * * *' (9 AM daily).
        job_id:           Optional custom job ID.

    Returns:
        The job_id string.
    """
    scheduler = get_scheduler()
    _id = job_id or f"recur_{phone_number}_{cron_expression.replace(' ', '_')}"

    # Parse cron expression (min hour dom mon dow)
    parts = cron_expression.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid cron expression: '{cron_expression}'. Expected 5 fields.")

    minute, hour, day, month, day_of_week = parts
    scheduler.add_job(
        _send_reminder_job,
        trigger=CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
            timezone=_TIMEZONE,
        ),
        args=[phone_number, message],
        id=_id,
        replace_existing=True,
    )
    logger.info("Scheduled recurring reminder '%s' for %s (cron: %s)", _id, phone_number, cron_expression)
    return _id


def cancel_reminder(job_id: str) -> bool:
    """
    Cancel a scheduled reminder by job ID.

    Returns:
        True if cancelled, False if not found.
    """
    scheduler = get_scheduler()
    job = scheduler.get_job(job_id)
    if job:
        scheduler.remove_job(job_id)
        logger.info("Cancelled reminder: %s", job_id)
        return True
    logger.warning("Reminder not found: %s", job_id)
    return False


def list_reminders() -> list[dict]:
    """Return a list of all currently scheduled reminder jobs."""
    scheduler = get_scheduler()
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "job_id": job.id,
            "next_run_time": str(job.next_run_time) if job.next_run_time else None,
            "trigger": str(job.trigger),
        })
    return jobs


def shutdown_scheduler() -> None:
    """Gracefully shut down the scheduler (called on app shutdown)."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("APScheduler shut down.")
        _scheduler = None
