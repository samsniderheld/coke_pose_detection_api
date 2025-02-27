import pytest
from flask import Flask
from flask.testing import FlaskClient
from main import app

@pytest.fixture(scope="session")
def client() -> FlaskClient:
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_generate_pose_json(client):
    with open("test_img.png", "rb") as img:
        data = {"file": (img, "test_img.png")}
        response = client.post("/generate_pose/", content_type='multipart/form-data', data=data)
        assert response.status_code == 200
        json_response = response.get_json()
        assert "keypoints" in json_response
        assert isinstance(json_response["keypoints"], dict)
