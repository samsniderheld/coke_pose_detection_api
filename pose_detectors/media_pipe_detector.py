import mediapipe as mp
import numpy as np
import os
import requests
from PIL import Image
from pose_detectors import PoseDetector


from mediapipe.tasks.vision import poseLandmarkerResult

class MediaPipeDetector(PoseDetector):
    def __init__(self):
        self.load_model()
        self.model_path = "pose_landmarker_heavy.task"

    def check_and_download_model(self):
        if not os.path.exists(self.model_path):
            print(f"Model file not found at {self.model_path}. Downloading...")
            url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
            response = requests.get(url)
            with open(self.model_path, 'wb') as f:
                f.write(response.content)
            print(f"Model downloaded to {self.model_path}")

    def load_model(self):
        self.model_path = 'pose_landmarker_heavy.task'
        self.pose = mp.solutions.pose.Pose()
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
        running_mode=self.VisionRunningMode.IMAGE)

    def load_image_path(self, image_path):
        # Load the input image from an image file.
        mp_image = mp.Image.create_from_file(image_path)
        return mp_image
    
    def load_image_np_array(self, numpy_image):
        # Load the input image from a numpy array.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        return mp_image

    def detect_poses(self, image: Image) -> poseLandmarkerResult:
        image = np.array(image)
        mp_image = self.load_image_np_array(image)

        with self.PoseLandmarker.create_from_options(self.options) as landmarker:
        # The landmarker is initialized. Use it here.
            pose_landmarker_result = landmarker.detect(mp_image)
        
        return pose_landmarker_result
    