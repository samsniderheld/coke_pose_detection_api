import pytest
import numpy as np
import tensorflow as tf
from pose_detectors.metrabs_detector import MetrabsDetector

@pytest.fixture(scope="session")
def metrabs_detector():
    return MetrabsDetector(model="small")

def test_load_model(metrabs_detector):
    assert metrabs_detector.model is not None

def test_load_image_path(metrabs_detector):
    image_path = "test_img.png"
    image = metrabs_detector.load_image_path(image_path)
    assert isinstance(image, tf.Tensor)

def test_detect_poses(metrabs_detector):
    image_path = "test_img.png"
    detection_results, rendered_image, plotted_image = metrabs_detector.detect_poses(image_path)
    assert isinstance(detection_results, dict)
    assert isinstance(rendered_image, np.ndarray)
    assert isinstance(plotted_image, np.ndarray)

def test_create_json_response(metrabs_detector):
    image_path = "test_img.png"
    detection_results, _, _ = metrabs_detector.detect_poses(image_path)
    json_response = metrabs_detector.create_json_response(detection_results)
    assert "keypoints" in json_response
    assert isinstance(json_response["keypoints"], dict)

def test_draw_landmarks_on_image(metrabs_detector):
    image_path = "test_img.png"
    detection_results, _, _ = metrabs_detector.detect_poses(image_path)
    tf_image = metrabs_detector.load_image_path(image_path)
    annotated_image = metrabs_detector.draw_landmarks_on_image(tf_image, detection_results)
    assert isinstance(annotated_image, np.ndarray)

def test_plot_landmarks(metrabs_detector):
    image_path = "test_img.png"
    detection_results, _, _ = metrabs_detector.detect_poses(image_path)
    plotted_image = metrabs_detector.plot_landmarks(detection_results)
    assert isinstance(plotted_image, np.ndarray)