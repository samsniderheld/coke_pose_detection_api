from abc import ABC, abstractmethod
import os
from PIL import Image

class PoseDetector(ABC):
    """Abstract base class for pose detection methods."""

    def __init__(self):
        self.load_model()
        self.model_path = "path/to/your/model/file"  # Placeholder path

    def check_and_download_model(self):
        if not os.path.exists(self.model_path):
            pass
        pass

    def load_model(self):
        pass

    def load_image_path(self, image_path):
        pass
    
    def load_image_np_array(self, numpy_image):
        pass

    @abstractmethod
    def detect_poses(self, image: Image) -> dict:
        """Detect poses in the given image and return keypoints."""
        pass
