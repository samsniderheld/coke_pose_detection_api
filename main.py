import io

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from pose_detectors.media_pipe_detector import MediaPipeDetector

app = FastAPI()

pose_Detector = MediaPipeDetector()

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), return_type: str = "json"):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        pose_data = pose_Detector.detect_poses(image)
        
        # If return_type is JSON, return metadata
        return JSONResponse({"landmarks": pose_data.landmakrs, "world_landmarks": pose_data.world_landmarks})
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
