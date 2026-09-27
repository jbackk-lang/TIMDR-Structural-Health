"""Testy dashboardu: serwer w watku, zapytania do API (tylko 127.0.0.1)."""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

import dashboard as D


@pytest.fixture(scope="module")
def base():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), D.Handler); th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(u):
    with urllib.request.urlopen(u) as r:
        return r.status, r.read()


def post(u, body):
    req = urllib.request.Request(u, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_and_validation(base):
    s, b = get(base + "/"); assert s == 200 and b"TIMDR Structural Health" in b
    s, b = get(base + "/api/validation"); v = json.loads(b); assert any(r["werdykt"] == "NOT SUPPORTED" for r in v)


def test_demos(base):
    most = json.loads(get(base + "/api/demo?name=most")[1])
    assert most["po"][1] - most["przed"][1] == pytest.approx(0.07, abs=0.02)
    sruba = json.loads(get(base + "/api/demo?name=sruba")[1]); H = [r["H"] for r in sruba["poziomy"]]
    assert all(b > a for a, b in zip(H, H[1:]))
    ud = json.loads(get(base + "/api/demo?name=uderzenia")[1])
    assert [r["rezim"] for r in ud["przypadki"]] == ["czasteczka", "pakiet", "fala"]


def test_analyze_csv_modes_and_errors(base):
    fs = 50.0; t = np.arange(int(120 * fs)) / fs; rng = np.random.default_rng(0)
    X = np.stack([np.sin(2 * np.pi * 2.95 * t) + 0.3 * rng.standard_normal(len(t)) for _ in range(2)], 1)
    csv = "\n".join(",".join(f"{v:.6f}" for v in row) for row in X)
    s, r = post(base + "/api/analyze", {"csv": csv, "fs": fs, "analysis": "modes", "anchors": "2.92"})
    assert s == 200 and r["czestotliwosci"][0] == pytest.approx(2.95, abs=0.01)
    s, r = post(base + "/api/analyze", {"csv": "a,b\n1,x", "fs": 1, "analysis": "modes", "anchors": "1"})
    assert s == 400 and "blad" in r
