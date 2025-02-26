import io
import mediapipe as mp
import numpy as np
import os
import requests
from PIL import Image
from PIL import Image as PILImage
from .pose_detector import PoseDetector

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from mediapipe.python._framework_bindings.image import Image
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

connections = mp.solutions.pose.POSE_CONNECTIONS

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5

class MediaPipeDetector(PoseDetector):
    def __init__(self):
        self.model_path = "pose_landmarker_heavy.task"
        self.check_and_download_model()
        self.load_model()
        

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

    def load_image_path(self, image_path) -> Image:
        # Load the input image from an image file.
        mp_image = mp.Image.create_from_file(image_path)
        return mp_image
    
    def load_image_np_array(self, numpy_image) -> Image:
        # Load the input image from a numpy array.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        return mp_image
    
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        np_image = rgb_image.numpy_view()
        annotated_image = np.copy(np_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image
    
    def figure_to_numpy(self, fig):
        # Save the figure to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        # Read the buffer into a PIL image
        img = PILImage.open(buf)
        
        # Convert the PIL image to a NumPy array
        img_array = np.array(img)
        
        return img_array
    
    def plot_landmarks(self, poses):
        fig = plt.figure()
        ax = plt.axes(111,projection='3d')
        ax.view_init(elev=10, azim=10)
        plotted_landmarks = {}
        for idx, landmark in enumerate(poses.pose_world_landmarks[0]):

            if (landmark.visibility < _VISIBILITY_THRESHOLD):
                continue

            ax.scatter3D(
                xs=[-landmark.z],
                ys=[landmark.x],
                zs=[-landmark.y]
            )
            plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)

        num_landmarks = len(poses.pose_world_landmarks[0])
        
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx], plotted_landmarks[end_idx]
                ]
                ax.plot3D(
                    xs=[landmark_pair[0][0], landmark_pair[1][0]],
                    ys=[landmark_pair[0][1], landmark_pair[1][1]],
                    zs=[landmark_pair[0][2], landmark_pair[1][2]],
                )
        # Convert the figure to a NumPy array
        img_array = self.figure_to_numpy(fig)
        
        plt.close(fig)  # Close the figure to free memory
        
        return img_array


    def detect_poses(self, image_or_path) -> list[PoseLandmarkerResult, Image, np.ndarray]:
        if isinstance(image_or_path, Image):
            image = np.array(image_or_path)
            mp_image = self.load_image_np_array(image)
        elif isinstance(image_or_path, str):
            mp_image = self.load_image_path(image_or_path)
        else:
            raise ValueError("Input must be a PIL Image or a file path")

        with self.PoseLandmarker.create_from_options(self.options) as landmarker:
            # The landmarker is initialized. Use it here.
            pose_landmarker_result = landmarker.detect(mp_image)

        rendered_image = self.draw_landmarks_on_image(mp_image, pose_landmarker_result)
        plotted_image = self.plot_landmarks(pose_landmarker_result)
        
        return [pose_landmarker_result, rendered_image,plotted_image]
    
    
    