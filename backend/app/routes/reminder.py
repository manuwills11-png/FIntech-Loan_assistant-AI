"""
POST /set-reminder  – Schedule a WhatsApp reminder.
GET  /set-reminder  – List all scheduled reminders.
DELETE /set-reminder/{job_id} – Cancel a reminder.
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.models.schemas import ReminderInput, ReminderOutput
from app.scheduler.reminder_scheduler import (
    cancel_reminder,
    list_reminders,
    schedule_one_time_reminder,
    schedule_recurring_reminder,
)

router = APIRouter(prefix="/set-reminder", tags=["Reminders"])


@router.post("", response_model=ReminderOutput, summary="Schedule a WhatsApp reminder")
async def set_reminder(data: ReminderInput) -> ReminderOutput:
    """
    Schedule a WhatsApp reminder.

    - Provide `remind_at` (ISO-8601) for a one-time reminder.
    - Provide `repeat` (cron expression) for recurring reminders.
    - Both can be provided together (one-time + recurring).
    """
    try:
        run_at = datetime.fromisoformat(data.remind_at)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid remind_at format. Use ISO-8601, e.g. '2024-06-15T09:00:00'",
        )

    try:
        if data.repeat:
            job_id = schedule_recurring_reminder(
                phone_number=data.phone_number,
                message=data.message,
                cron_expression=data.repeat,
            )
        else:
            job_id = schedule_one_time_reminder(
                phone_number=data.phone_number,
                message=data.message,
                run_at=run_at,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scheduler error: {exc}")

    return ReminderOutput(
        job_id=job_id,
        status="scheduled",
        remind_at=data.remind_at,
        message=data.message,
    )


@router.get("", summary="List all scheduled reminders")
async def get_reminders() -> list[dict]:
    """Return all currently scheduled reminders."""
    return list_reminders()


@router.delete("/{job_id}", summary="Cancel a scheduled reminder")
async def delete_reminder(job_id: str) -> dict:
    """Cancel a reminder by its job ID."""
    cancelled = cancel_reminder(job_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail=f"Reminder '{job_id}' not found.")
    return {"status": "cancelled", "job_id": job_id}
