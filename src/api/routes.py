from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import shutil
import uuid
from pathlib import Path

from src.api.app import pipeline, UPLOAD_DIR, OUTPUT_DIR
from src.utils.logger import get_logger
from src.core.registry import registry

logger = get_logger(__name__)
router = APIRouter()


@router.get("/status")
async def get_status():
    """Health check endpoint."""
    return {"status": "ok", "message": "Image Restoration API is running."}


@router.get("/models")
async def get_models():
    """Returns loaded models."""
    loaded_models = {name: model.is_loaded for name, model in registry._models.items()}
    return {"models": loaded_models}


@router.post("/restore")
async def restore_image(file: UploadFile = File(...)):
    """
    Upload an image and return the restored image along with metadata.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    if not ext:
        ext = ".jpg"

    input_path = UPLOAD_DIR / f"{file_id}{ext}"
    output_path = OUTPUT_DIR / f"{file_id}_restored{ext}"

    # Save uploaded file
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save upload: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    # Process image
    try:
        results = pipeline.process_image(input_path, output_path)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # In a real API, you might want to return the image directly or a URL to download it.
    # Here, we return metadata and the client can fetch the image from another endpoint or we return it directly.
    # We will return the metadata and the path.
    return {
        "id": file_id,
        "filename": file.filename,
        "restored_url": f"/download/{file_id}_restored{ext}",
        "metadata": results,
    }


@router.get("/download/{filename}")
async def download_file(filename: str):
    """Serve restored images."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path))
