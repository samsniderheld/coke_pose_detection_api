import pytest
from flask import Flask
from flask.testing import FlaskClient
from main import app

@pytest.fixture
def client() -> FlaskClient:
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_generate_pose_json(client):
    with open("test_image.png", "rb") as img:
        data = {"file": (img, "test_image.png")}
        response = client.post("/generate_pose/", content_type='multipart/form-data', data=data)
        assert response.status_code == 200
        json_response = response.get_json()
        assert "keypoints" in json_response
        assert isinstance(json_response["keypoints"], dict)

def test_generate_pose_invalid_file(client):
    with open("invalid_file.txt", "rb") as img:
        data = {"file": (img, "invalid_file.txt")}
        response = client.post("/generate_pose/", content_type='multipart/form-data', data=data)
        assert response.status_code == 400
        json_response = response.get_json()
        assert "error" in json_response

def test_generate_pose_missing_file(client):
    response = client.post("/generate_pose/")
    assert response.status_code == 400  # Bad Request, since no file was provided
    json_response = response.get_json()
    assert "error" in json_response