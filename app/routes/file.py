
import logging
import tempfile

from pathlib import Path
from fastapi import APIRouter, UploadFile, HTTPException

from app.ai.rag.pipeline import rag_pipeline


router = APIRouter(tags=["Documents"])

logger = logging.getLogger("uvicorn.error")

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

MAX_UPLOAD_SIZE = 10 * 1024 * 1024


@router.post("/upload")
async def upload_files(file: UploadFile):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )

    original_filename = Path(file.filename).name
    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT and MD files are supported."
        )

    try:
        file_bytes = await file.read(MAX_UPLOAD_SIZE + 1)

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        if len(file_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Maximum upload size is 10 MB."
            )

        with tempfile.TemporaryDirectory(
            prefix="cloud_llm_upload_"
        ) as temp_dir:

            temp_path = Path(temp_dir) / original_filename

            temp_path.write_bytes(file_bytes)

            result = await rag_pipeline.ingest_file(temp_path)

        return {
            "filename": original_filename,
            **result
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Document upload and ingestion failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Document processing failed. Check server logs."
        )

    finally:
        await file.close()