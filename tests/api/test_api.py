import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from src.api.app import app
import io
from PIL import Image

client = TestClient(app)

def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Image Restoration API is running."}

def test_models():
    response = client.get("/models")
    assert response.status_code == 200
    assert "models" in response.json()

def test_restore_endpoint():
    # Create a dummy image in memory
    img = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}
    
    response = client.post("/restore", files=files)
    assert response.status_code == 200
    json_resp = response.json()
    assert "id" in json_resp
    assert "restored_url" in json_resp
    assert "metadata" in json_resp
