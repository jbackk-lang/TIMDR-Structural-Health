"""dashboard.py -- lokalny dashboard TIMDR-Structural-Health (tylko biblioteka standardowa + numpy/scipy).

  GET  /                    -> static/dashboard.html
  GET  /api/validation      -> tabela walidacji (tez wyniki negatywne)
  GET  /api/demo?name=...   -> demo syntetyczne: most | sruba | uderzenia
  POST /api/analyze         -> analiza wlasnego pliku CSV (kolumny = kanaly), JSON:
                               {"csv": "...", "analysis": "modes|lines|lines2|regime", "fs": 100, ...}
Serwer slucha tylko na 127.0.0.1 (nic nie wychodzi poza komputer).
Uruchomienie: python dashboard.py  (albo run.bat) -> http://127.0.0.1:8765
"""
from __future__ import annotations

import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from scipy.signal import welch

import timdr_shm as T

HOST, PORT = "127.0.0.1", 8765
STATIC = Path(__file__).resolve().parent / "static"
MAX_BYTES = 60 * 1024 * 1024

VALIDATION = [
    {"obiekt": "Most kolejowy KW51 (Leuven), przed/po wzmocnieniu", "timdr": "kotwica z 2 kanałów: AUC 1,00; zgodność z OMA ≤ 1% (4/5 modów)",
     "odniesienie": "SSI autorów 1,00 (remis); AR 0,67", "werdykt": "SUPPORTED"},
    {"obiekt": "Rama 3-kondygnacyjna LANL, zderzak", "timdr": "linie przy dwóch modach: ciężkość ρ +0,98",
     "odniesienie": "klasyczne SHM 0,97", "werdykt": "SUPPORTED"},
    {"obiekt": "Połączenie śrubowe ORION-AE, luzowanie 60 → 5 cNm", "timdr": "linie przy kotwicy: ρ −0,85 / −0,64; reżim cząsteczki przy średnim luzie",
     "odniesienie": "dokładny poziom: TIMDR 0,18, klasyka 0,04 (obie losowo)", "werdykt": "SUPPORTED (trend)"},
    {"obiekt": "Most Hell Bridge Test Arena, 8 lokalnych uszkodzeń", "timdr": "kotwica + kształt modu: AUC 0,65",
     "odniesienie": "AR 0,79", "werdykt": "NOT SUPPORTED"},
]


def spectrum(X: np.ndarray, fs: float, fmax: float | None = None, n_out: int = 800):
    X = np.atleast_2d(X)
    f, P = welch(X - X.mean(1, keepdims=True), fs, nperseg=min(4096, X.shape[1]))
    S = (P / np.median(P, 1, keepdims=True)).mean(0)
    m = f <= (fmax or fs / 2); f, S = f[m], S[m]
    if len(f) > n_out:
        idx = np.linspace(0, len(f) - 1, n_out).astype(int); f, S = f[idx], S[idx]
    return {"f": np.round(f, 5).tolist(), "S": np.round(np.log10(S + 1e-12), 4).tolist()}


# ---------------- dema syntetyczne (oznaczone jako syntetyczne) ----------------
def demo_most(seed=1):
    rng = np.random.default_rng(seed); fs = 50.0; n = int(300 * fs); t = np.arange(n) / fs
    model = [2.58, 2.92, 5.32, 6.31]; after = [2.57, 2.99, 5.44, 6.41]   # przesuniecia jak po wzmocnieniu KW51
    def rec(freqs):
        X = 0.3 * rng.standard_normal((2, n))
        for fm in freqs:
            for c in range(2):
                X[c] += (0.6 + 0.4 * c) * np.sin(2 * np.pi * fm * t + rng.uniform(0, 6)) * (1 + 0.2 * rng.standard_normal(n)).clip(0)
        return X
    A, B = rec(model), rec(after)
    fa, fb = T.anchor_frequencies(A, fs, model), T.anchor_frequencies(B, fs, model)
    return {"typ": "most", "opis": "Syntetyczny most: 2 kanały pionowe, 300 s, drgania otoczenia. Wzmocnienie przesuwa mody (jak w KW51).",
            "kotwice_modelu": model, "przed": fa, "po": fb,
            "widmo_przed": spectrum(A, fs, 8.0), "widmo_po": spectrum(B, fs, 8.0)}


def demo_sruba(seed=2):
    rng = np.random.default_rng(seed); fs = 5000.0; t = np.arange(int(2 * fs)) / fs
    levels = [60, 50, 40, 30, 20, 10, 5]; rows = []
    for i, lv in enumerate(levels):
        c = 0.004 * (i + 1) ** 1.6                       # luz rosnie -> nieliniowosc kontaktu
        v = np.sin(2 * np.pi * 118.4 * t); v = v + c * v ** 2 + 0.5 * c * v ** 3 + 0.002 * rng.standard_normal(len(t))
        r = T.lines_known_excitation(v, fs); rows.append({"moment_cNm": lv, "H": r["H"], "f0": r["f0"]})
    return {"typ": "sruba", "opis": "Syntetyczna śruba na wibratorze 118,4 Hz: luzowanie 60 → 5 cNm zwiększa nieliniowość kontaktu.", "poziomy": rows}


def demo_uderzenia(seed=3):
    rng = np.random.default_rng(seed); fs = 20000.0; n = int(1.0 * fs); rows = []
    ring = np.exp(-np.arange(80) / 12) * np.sin(2 * np.pi * 4000 * np.arange(80) / fs)
    for rate, name in ((10, "rzadkie"), (150, "średnie"), (2000, "częste")):
        imp = np.zeros(n); imp[(np.arange(0, 1.0, 1 / rate) * fs).astype(int)] = 5.0
        x = 0.05 * rng.standard_normal(n) + np.convolve(imp, ring, "same")
        r = T.regime(x, fs, 3000, 5000); rows.append({"uderzenia_na_s": rate, "nazwa": name, **r})
    return {"typ": "uderzenia", "opis": "Syntetyczne uderzenia (zderzak, mikropoślizg): rzadkie są cząsteczkami, częste zlewają się w falę.", "przypadki": rows}


DEMOS = {"most": demo_most, "sruba": demo_sruba, "uderzenia": demo_uderzenia}


def analyze_csv(req: dict) -> dict:
    X = np.atleast_2d(np.loadtxt(io.StringIO(req["csv"]), delimiter=",", ndmin=2).T)
    fs = float(req["fs"]); kind = req.get("analysis", "modes")
    out = {"kanaly": int(X.shape[0]), "probki": int(X.shape[1]), "czas_s": round(X.shape[1] / fs, 2), "analiza": kind}
    if kind == "modes":
        anch = [float(v) for v in str(req["anchors"]).split(",") if v.strip()]
        f = T.anchor_frequencies(X, fs, anch)
        out.update({"kotwice_modelu": anch, "czestotliwosci": f, "widmo": spectrum(X, fs, 1.3 * max(anch))})
        out["wykonalnosc_N_cyk"] = [round(out["czas_s"] * a) for a in anch]
    elif kind == "lines":
        out.update(T.lines_known_excitation(X[0], fs)); out["widmo"] = spectrum(X[:1], fs, 4 * out["f0"])
    elif kind == "lines2":
        ba = [float(v) for v in str(req["band_a"]).split(",")]; bb = [float(v) for v in str(req["band_b"]).split(",")]
        out.update(T.lines_two_modes(X[0], fs, ba, bb)); out["widmo"] = spectrum(X[:1], fs, 2.3 * max(bb))
    elif kind == "regime":
        lo, hi = [float(v) for v in str(req["band"]).split(",")]
        out.update(T.regime(X[0], fs, lo, hi))
    else:
        raise ValueError("nieznana analiza")
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, (STATIC / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/validation":
            return self._send(200, VALIDATION)
        if u.path == "/api/demo":
            name = parse_qs(u.query).get("name", ["most"])[0]
            if name not in DEMOS:
                return self._send(404, {"blad": "nieznane demo"})
            return self._send(200, DEMOS[name]())
        return self._send(404, {"blad": "nie znaleziono"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/analyze":
            return self._send(404, {"blad": "nie znaleziono"})
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_BYTES:
            return self._send(413, {"blad": "plik za duży (limit 60 MB)"})
        try:
            return self._send(200, analyze_csv(json.loads(self.rfile.read(n))))
        except Exception as e:                      # blad danych uzytkownika -> czytelny komunikat
            return self._send(400, {"blad": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    print(f"TIMDR-Structural-Health: http://{HOST}:{PORT}  (Ctrl+C konczy)")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
