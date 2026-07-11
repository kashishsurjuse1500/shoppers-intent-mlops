# This file contains pytest tests for FastAPI endpoints

import pytest  # pytest testing framework
from fastapi.testclient import TestClient  # test client for fastapi
import sys  # system path manipulation
import os  # access env variables

# add root and ml directory to path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ml')))

from api.main import app  # import fastapi app

client = TestClient(app)  # create test client

# sample valid input for prediction
sample_input = {
    "Administrative": 0,
    "Administrative_Duration": 0.0,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 1,
    "ProductRelated_Duration": 0.0,
    "BounceRates": 0.2,
    "ExitRates": 0.2,
    "PageValues": 0.0,
    "SpecialDay": 0.0,
    "Month": 2,
    "OperatingSystems": 1,
    "Browser": 1,
    "Region": 1,
    "TrafficType": 1,
    "VisitorType": 2,
    "Weekend": 0
}

def test_root():
    # test root endpoint returns 200 with HTML response
    response = client.get("/")  # call root endpoint
    assert response.status_code == 200  # check status code
    assert "text/html" in response.headers["content-type"]  # check html response

def test_health():
    # test health endpoint returns healthy status
    response = client.get("/health")  # call health endpoint
    assert response.status_code == 200  # check status code
    assert response.json()["status"] == "healthy"  # check response

def test_predict_valid_input():
    # test predict endpoint with valid input
    response = client.post("/predict", json=sample_input)  # call predict endpoint
    assert response.status_code == 200  # check status code
    data = response.json()  # parse response
    assert "prediction" in data           # check prediction key exists
    assert "probability" in data          # check probability key exists
    assert "message" in data              # check message key exists
    assert data["prediction"] in [0, 1]   # prediction must be 0 or 1
    assert 0.0 <= data["probability"] <= 1.0  # probability must be between 0 and 1

def test_predict_invalid_input():
    # test predict endpoint with missing fields
    response = client.post("/predict", json={"Administrative": 0})  # incomplete input
    assert response.status_code == 422  # 422 = validation error from pydantic

def test_predict_purchase_case():
    # test predict endpoint with high purchase probability input
    purchase_input = {
        "Administrative": 2,
        "Administrative_Duration": 80.0,
        "Informational": 1,
        "Informational_Duration": 25.0,
        "ProductRelated": 15,
        "ProductRelated_Duration": 450.0,
        "BounceRates": 0.02,
        "ExitRates": 0.05,
        "PageValues": 25.5,
        "SpecialDay": 0.0,
        "Month": 5,
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 2,
        "VisitorType": 2,
        "Weekend": 0
    }
    response = client.post("/predict", json=purchase_input)  # call predict endpoint
    assert response.status_code == 200  # check status code
    data = response.json()  # parse response
    assert data["prediction"] == 1  # high page value should predict purchase