from fastapi.testclient import TestClient

import web_app


def test_home_page_loads():
    with TestClient(web_app.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Vehicle Knowledge Assistant" in response.text


def test_health_endpoint():
    # Entering TestClient as a context manager runs FastAPI startup.
    with TestClient(web_app.app) as client:
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["local_rag_ready"] is True
        assert data["local_records"] == 8


def test_vehicle_query_parser():
    result = web_app.parse_vehicle_query("What recalls affect a 2020 Toyota Camry?")
    assert result == (2020, "Toyota", "Camry")


def test_campaign_deduplication():
    records = [
        {"id": "20V682000", "text": "Fuel pump recall details"},
        {"id": "20V682000", "text": "Duplicate fuel pump recall details"},
        {"id": "23V865000", "text": "Airbag sensor recall details"},
    ]

    result = web_app.deduplicate(records)
    campaigns = [web_app.get_campaign(doc) for doc in result]

    assert len(result) == 2
    assert campaigns == ["20V682000", "23V865000"]


def test_empty_question_is_rejected():
    with TestClient(web_app.app) as client:
        response = client.post("/api/ask", json={"question": "   "})
        assert response.status_code == 400
