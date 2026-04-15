"""
POST /upload-document  – Document OCR and financial data extraction.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import DocumentExtractOutput
from app.services.ocr_service import extract_document

router = APIRouter(prefix="/upload-document", tags=["Document Intelligence"])

_ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/tiff",
    "application/pdf",
}
_MAX_FILE_SIZE_MB = 10


@router.post(
    "",
    response_model=DocumentExtractOutput,
    summary="Extract financial data from bank statement or salary slip",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF or image of a bank statement or salary slip"),
) -> DocumentExtractOutput:
    """
    Upload a financial document (PDF or image) for automated analysis.

    Uses Google Cloud Vision API for OCR to extract:
    - Estimated monthly income
    - Monthly expense total
    - Individual transactions (with date, description, amount, type)

    Supported formats: JPEG, PNG, WEBP, TIFF, PDF
    """
    content_type = file.content_type or ""

    if content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. "
                   f"Allowed: {', '.join(_ALLOWED_TYPES)}",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > _MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed: {_MAX_FILE_SIZE_MB} MB.",
        )

    try:
        result = extract_document(file_bytes=file_bytes, content_type=content_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {exc}")

    return result
