import cv2
import io
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as tfhub

from PIL import Image
from .pose_detector import PoseDetector

"""
full list of skeletons:
smpl_24, kinectv2_25, h36m_17, h36m_25, mpi_inf_3dhp_17, mpi_inf_3dhp_28, coco_19, smplx_42, ghum_35, 
lsp_14, sailvos_26, gpa_34, aspset_17, bml_movi_87, mads_19, berkeley_mhad_43, total_capture_21, jta_22, 
ikea_asm_17, human4d_32, 3dpeople_29, umpm_15, smpl+head_30

"""

class MetrabsDetector(PoseDetector):
    def __init__(self,model='large'):
        self.load_model(model)

    def load_model(self,model):
        if model == 'large':
            self.model = tfhub.load('https://bit.ly/metrabs_l')
        elif model == 'small':
            self.model = tfhub.load('https://bit.ly/metrabs_s')
        else:
            raise ValueError("Invalid model size. Choose either 'large' or 'small'.")
        

    def load_image_path(self, image_path) -> Image:
        # Load the input image from an image file.
        tf_image = tf.image.decode_jpeg(tf.io.read_file(image_path))
        return tf_image
    
    def draw_landmarks_on_image(self, tf_image, detection_result,skeleton)-> np.ndarray:
        pose_landmarks_list = detection_result['poses2d'].numpy()
        edges = self.model.per_skeleton_joint_edges[skeleton].numpy()
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
    
    def figure_to_numpy(self, fig)-> np.ndarray:
        # Save the figure to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        # Read the buffer into a PIL image
        img = Image.open(buf)
        
        # Convert the PIL image to a NumPy array
        img_array = np.array(img)
        
        return img_array
    
    def plot_landmarks(self, detection_result, skeleton)-> np.ndarray:
        edges = self.model.per_skeleton_joint_edges[skeleton].numpy()
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
        # Convert the figure to a NumPy array
        img_array = self.figure_to_numpy(fig)
        
        plt.close(fig)  # Close the figure to free memory
        
        return img_array


    def detect_poses(self, image_or_path,skeleton="smpl_24") -> list[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(image_or_path, Image.Image):
            raise ValueError("Input must be a file path")
        elif isinstance(image_or_path, str):
            image = self.load_image_path(image_or_path)
        else:
            raise ValueError("Input must be a file path")

        detection_results = self.model.detect_poses(image, skeleton=skeleton)

        rendered_image = self.draw_landmarks_on_image(image, detection_results, skeleton)
        plotted_image = self.plot_landmarks(detection_results,skeleton)
        
        return [detection_results, rendered_image,plotted_image]
    
    def create_json_response(self, detection_results)-> list:
        # Convert the detection results to a JSON-compatible format
        poses3d = detection_results['poses3d'].numpy().tolist()
        joint_names = self.model.per_skeleton_joint_names[skeleton].numpy().tolist()
        
        # Create a dictionary with joint names as keys and corresponding 3D poses as values
        joint_pose_dict = {str(joint_names[i]): poses3d[0][i] for i in range(len(joint_names))}
        json = {"keypoints": joint_pose_dict}
        return json

    def create_json_file(self, detection_results, output_path):
        json_response = self.create_json_response(detection_results)
        with open(output_path, 'w') as f:
            json.dump(json_response, f)
    
    
    