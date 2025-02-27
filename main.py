import io

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from pose_detectors.metrabs_detector import MetrabsDetector

app = FastAPI()

pose_detector = MetrabsDetector()

@app.post("/generate_pose/")
async def upload_image(file: UploadFile = File(...), return_type: str = "json"):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        pose_data = pose_detector.detect_poses(image)
        
        # If return_type is JSON, return metadata
        return JSONResponse(pose_detector.create_json_response(pose_data))
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
