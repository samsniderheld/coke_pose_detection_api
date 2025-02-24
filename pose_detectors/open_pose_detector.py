import numpy as np
from PIL import Image
from pose_detectors import PoseDetector

class OpenPoseDetector(PoseDetector):
    def detect_poses(self, image: Image) -> dict:
        """Dummy implementation of OpenPose detection."""
        keypoints = np.random.rand(17, 2).tolist()  # Simulating 17 keypoints (x, y)
        return {"method": "OpenPose", "keypoints": keypoints}
