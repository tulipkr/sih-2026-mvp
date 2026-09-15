import pytest
from fastapi.testclient import TestClient
import os,sys 

# 1. ALWAYS modify sys.path BEFORE importing custom local modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",  ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from backend.src.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_invalid_scene_id_returns_400():
    response = client.post("/pipeline/run", json={"scene_id": "nonexistent_scene"})
    assert response.status_code == 400
    assert "Unknown scene_id" in response.json()["detail"]


def test_nonexistent_run_id_returns_404():
    response = client.get("/results/run_99999999_000000")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]