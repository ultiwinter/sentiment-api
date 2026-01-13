from fastapi.testclient import TestClient
from src.app_textblob import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    assert "labels" in j

def test_predict_single():
    r = client.post("/predict", json={"text": "I love it"})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data and len(data["results"]) == 1
    assert data["results"][0]["label"] in ["negative", "neutral", "positive"]
    assert data["results"][0]["label"] == "positive"
    assert "X-Process-Time-ms" in r.headers

def test_predict_batch_with_labels():
    texts = ["bad", "meh", "great"]
    expected = ["negative", "neutral", "positive"]

    r = client.post("/predict", json={"texts": texts})
    assert r.status_code == 200
    data = r.json()
    results = data["results"]

    assert len(results) == len(expected)

    for i, (res, exp) in enumerate(zip(results, expected)):
        assert res["label"] in ["negative", "neutral", "positive"]
        assert res["label"] == exp, f"For '{texts[i]}', expected {exp} but got {res['label']}"