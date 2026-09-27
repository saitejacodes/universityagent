from fastapi.testclient import TestClient

from uniagent import api
from uniagent.config import Settings
from uniagent.store.db import Store


def test_api_serves_facts_with_evidence(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path)
    store = Store(settings.db_path)
    store.add_run("r1", "2026-01-01T00:00:00", "m", {})
    store.add_run("r2", "2026-02-01T00:00:00", "m", {})
    row = {
        "university": "mit",
        "field": "founding_year",
        "status": "accepted",
        "confidence": 0.9,
        "quote": "1861: Founded in Boston",
        "url": "https://facts.mit.edu",
        "snapshot_sha": "abc",
        "page_type": "facts",
        "note": None,
    }
    store.add_facts("r1", [{**row, "value": 1860, "confidence": 0.5}])
    store.add_facts("r2", [{**row, "value": 1861}])
    store.close()
    monkeypatch.setattr(api, "get_settings", lambda: settings)

    client = TestClient(api.app)
    body = client.get("/universities/mit").json()
    fy = body["fields"]["founding_year"]
    assert fy["value"] == 1861 and fy["evidence"]["quote"].startswith("1861")
    assert client.get("/universities/mit", params={"min_confidence": 0.95}).json()["fields"] == {}
    hist = client.get("/universities/mit/history/founding_year").json()
    assert [h["value"] for h in hist] == [1860, 1861]
    assert client.get("/universities/nope").status_code == 404
