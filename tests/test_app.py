import app


def test_fallback_reply_is_usable():
    reply = app.fallback_reply("hello there")
    assert isinstance(reply, str)
    assert len(reply) > 10
    assert "cebri" in reply.lower() or "hello" in reply.lower()


def test_ask_cebri_uses_memory_when_available():
    app.remember_exchange("unit-test-user", "hello", "hi there")
    reply = app.ask_cebri("unit-test-user", "what did I just say?")
    assert isinstance(reply, str)
    assert len(reply) > 10

import types


def test_browser_action_requires_confirmation(monkeypatch):
    monkeypatch.setattr(app.webbrowser, "open", lambda *args, **kwargs: True)
    with app.app.test_client() as client:
        response = client.post("/browser-action", json={"url": "https://example.com"})
        assert response.status_code == 400
        data = response.get_json()
        assert data["requires_confirmation"] is True
        assert "confirm" in (data.get("error") or "").lower()


def test_browser_action_confirmed_opens_url(monkeypatch):
    opened = []
    monkeypatch.setattr(app.webbrowser, "open", lambda url, new=2: opened.append(url) or True)
    with app.app.test_client() as client:
        response = client.post("/browser-action", json={"url": "https://example.com", "confirmed": True})
        assert response.status_code == 200
        data = response.get_json()
        assert data["opened"] is True
        assert opened == ["https://example.com"]


def test_browser_action_pending_confirmation_flow(monkeypatch):
    opened = []
    monkeypatch.setattr(app.webbrowser, "open", lambda url, new=2: opened.append(url) or True)
    with app.app.test_client() as client:
        response = client.post("/browser-action", json={"url": "https://example.com", "request_confirmation": True})
        assert response.status_code == 202
        data = response.get_json()
        assert data["requires_confirmation"] is True
        action_id = data["action_id"]
        confirm = client.post("/browser-action/confirm", json={"action_id": action_id, "confirmed": True})
        assert confirm.status_code == 200
        assert confirm.get_json()["opened"] is True
        assert opened == ["https://example.com"]


def test_browser_action_rejects_invalid_url():
    with app.app.test_client() as client:
        response = client.post("/browser-action", json={"url": "not-a-url", "confirmed": True})
        assert response.status_code == 400
        assert response.get_json()["opened"] is False
