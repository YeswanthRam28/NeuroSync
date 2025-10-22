<<<<<<< HEAD
from fastapi import APIRouter, UploadFile, File
from deepface import DeepFace
import tempfile
import aiofiles
import os

router = APIRouter()

def convert(obj):
    """Convert numpy types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    return obj

@router.post("/analyze")
async def analyze_emotion(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    
    # Save uploaded file asynchronously
    async with aiofiles.tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        await tmp.write(content)

    try:
        # DeepFace analyze is blocking, keep it synchronous
        result = DeepFace.analyze(tmp_path, actions=['emotion'])
        return {"emotion_result": convert(result)}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
=======
from fastapi import APIRouter, UploadFile, File
from deepface import DeepFace
import tempfile
import aiofiles
import os

router = APIRouter()

def convert(obj):
    """Convert numpy types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    return obj

@router.post("/analyze")
async def analyze_emotion(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    
    # Save uploaded file asynchronously
    async with aiofiles.tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        await tmp.write(content)

    try:
        # DeepFace analyze is blocking, keep it synchronous
        result = DeepFace.analyze(tmp_path, actions=['emotion'])
        return {"emotion_result": convert(result)}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
>>>>>>> ccdce86647545da532a9c5a3730488c2e88c7dd7
