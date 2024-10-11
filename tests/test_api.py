from fastapi.testclient import TestClient
from src.main import app
from src.config import get_settings

SETTINGS = get_settings()

client = TestClient(app)


def test_get_model_info():
    data = {
        "model_name": "Pedestrian detection",
        "model_arquitecture": SETTINGS.yolo_version,
        "classes": ["person"],
        "input_type": "image",
    }
    response = client.get("/info")
    assert response.status_code == 200
    assert response.json() == data
