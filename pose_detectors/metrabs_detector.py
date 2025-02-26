import io
import cv2
import numpy as np
import os
from PIL import Image
from .pose_detector import PoseDetector

import tensorflow as tf
import os

import matplotlib.pyplot as plt



class MetrabsDetector(PoseDetector):
    def __init__(self):
        self.model_path = "metrabs_mob3l_y4t"
        self.check_and_download_model()
        self.load_model()
        

    def check_and_download_model(self):
        if not os.path.exists(self.model_path):
            print(f"Model file not found at {self.model_path}. Downloading...")
            server_prefix = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs'
            model_zippath = tf.keras.utils.get_file(
                origin=f'{server_prefix}/{self.model_path}_20211019.zip',
                extract=True, cache_subdir='models')
            model_path = os.path.join(os.path.dirname(model_zippath), self.model_path)
            print(f"Model downloaded to {self.model_path}")


    def load_model(self):
       self.model = tf.saved_model.load(self.model_path) # or metrabs_eff2l_y4 for the big model


    def load_image_path(self, image_path) -> Image:
        # Load the input image from an image file.
        tf_image = tf.image.decode_jpeg(tf.io.read_file(image_path))
        return tf_image
    
    def draw_landmarks_on_image(self, tf_image, detection_result):
        pose_landmarks_list = detection_result['poses2d'].numpy()
        edges = self.model.per_skeleton_joint_edges['smpl_24'].numpy()
        np_image = tf_image.numpy()
        annotated_image = np.copy(np_image)

        # Loop through the detected poses to visualize.
        for pose_landmarks in pose_landmarks_list:
            for landmark in pose_landmarks:
                x, y = int(landmark[0]), int(landmark[1])
                cv2.circle(annotated_image, (x, y), 3, (0, 255, 0), -1)

            for edge in edges:
                start_idx, end_idx = edge
                start_point = pose_landmarks[start_idx]
                end_point = pose_landmarks[end_idx]
                start_x, start_y = int(start_point[0]), int(start_point[1])
                end_x, end_y = int(end_point[0]), int(end_point[1])
                cv2.line(annotated_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
           
        return annotated_image
    
    
    
    def figure_to_numpy(self, fig):
        # Save the figure to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        # Read the buffer into a PIL image
        img = Image.open(buf)
        
        # Convert the PIL image to a NumPy array
        img_array = np.array(img)
        
        return img_array
    
    def plot_landmarks(self, detection_result):
        edges = self.model.per_skeleton_joint_edges['smpl_24'].numpy()
        poses3d = detection_result['poses3d'].numpy()
        fig = plt.figure(figsize=(10, 5.2))
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.view_init(5, -85)
        ax.set_xlim3d(-1500, 1500)
        ax.set_zlim3d(-1500, 1500)
        ax.set_ylim3d(0, 3000)

        # Matplotlib plots the Z axis as vertical, but our poses have Y as the vertical axis.
        # Therefore, we do a 90° rotation around the X axis:
        poses3d[..., 1], poses3d[..., 2] = poses3d[..., 2], -poses3d[..., 1]
        for pose3d in poses3d:
            for i_start, i_end in edges:
                ax.plot(*zip(pose3d[i_start], pose3d[i_end]), marker='o', markersize=2)
            ax.scatter(*pose3d.T, s=2)

        fig.tight_layout()
        plt.show()
        # Convert the figure to a NumPy array
        img_array = self.figure_to_numpy(fig)
        
        plt.close(fig)  # Close the figure to free memory
        
        return img_array


    def detect_poses(self, image_or_path) -> list[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(image_or_path, Image):
            raise ValueError("Input must be a file path")
        elif isinstance(image_or_path, str):
            image = self.load_image_path(image_or_path)
        else:
            raise ValueError("Input must be a file path")

        detection_results = self.odel.detect_poses(image, skeleton='smpl_24')

        rendered_image = self.draw_landmarks_on_image(image, detection_results)
        plotted_image = self.plot_landmarks(detection_results)
        
        return [detection_results['poses3d'].numpy(), rendered_image,plotted_image]
    
    
    