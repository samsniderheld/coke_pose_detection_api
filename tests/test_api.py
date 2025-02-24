import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_detect_pose_openpose():
    with open("test_image.png", "rb") as img:
        files = {"file": ("test_image.png", img, "image/png")}
        response = client.post("/detect_pose/?method=openpose", files=files)
        assert response.status_code == 200
        assert "method" in response.json()
        assert "keypoints" in response.json()

def test_detect_pose_mediapipe():
    with open("test_image.png", "rb") as img:
        files = {"file": ("test_image.png", img, "image/png")}
        response = client.post("/detect_pose/?method=mediapipe", files=files)
        assert response.status_code == 200
        assert "method" in response.json()
        assert "keypoints" in response.json()

def test_invalid_method():
    response = client.post("/detect_pose/?method=invalid", files={"file": ("test_image.png", b"image", "image/png")})
    assert response.status_code == 400
    assert "error" in response.json()
