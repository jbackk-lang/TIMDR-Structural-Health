# timdr_shm.py -- monitorowanie konstrukcji (budynki, mosty, polaczenia) wg drogowskazow TIMDR.
#
# Wszystkie funkcje przeniesione 1:1 (bez importu, repo niezalezne) z wersji walidowanych w GIA-TIMDR; zgodnosc liczbowa
# pilnowana testami wzorcowymi (test_timdr_shm.py). Walidacja pre-rejestrowana -- patrz README (takze wyniki negatywne).
#
#   kotwica modalna (galaz K)      -- core/kw51_modal_anchor.py      (most KW51: AUC 1,00, zgodnosc z OMA <= 1%)
#   linie przy kotwicy (nielin.)   -- core/real_lanl_modal_anchor.py (rama LANL: ciezkosc rho +0,98)
#                                     core/orion_bolt.py             (sruba ORION-AE: rho -0,85 / -0,64)
#   rezim: czasteczkowosc P, D     -- core/transition_params.py, core/orion_bolt.py
#   detektor nowosci kNN           -- core/real_lanl_3story_bridge.py
#
# Uzycie (CLI):
#   python timdr_shm.py modes    plik.npz --fs 825.8 --anchors 2.58,2.92,5.32 [--decimate 32]
#   python timdr_shm.py lines    plik.npz --fs 5000 --excitation auto            (znane wymuszenie)
#   python timdr_shm.py lines2   plik.npz --fs 322.58 --band-a 45,62 --band-b 64,80   (wymuszenie losowe, dwa mody)
#   python timdr_shm.py regime   plik.npz --fs 5e6 --band 100e3,900e3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.signal import decimate, welch

TOL = 0.04        # samokorekta kotwicy: +-4% wokol czestotliwosci modelu modalnego


# ---------------- kotwica modalna (galaz K) ----------------
def _decim(x: np.ndarray, q: int) -> np.ndarray:
    y = np.asarray(x, float)
    while q > 1:
        s = 10 if q % 10 == 0 else (8 if q % 8 == 0 else (4 if q % 4 == 0 else (2 if q % 2 == 0 else q)))
        y = decimate(y, s, ftype="fir", zero_phase=True); q //= s
    return y


def anchor_frequencies(X: np.ndarray, fs: float, anchors: Sequence[float], q: int = 1, nperseg: int = 1024) -> List[float]:
    """X: (kanaly, probki). Widmo usrednione po kanalach (kazdy znormalizowany mediana); w +-4% wokol kazdej kotwicy
    szczyt z interpolacja paraboliczna. Zwraca czestotliwosci [Hz]."""
    Y = np.stack([_decim(c, q) for c in np.atleast_2d(X)]); fd = fs / q
    f, P = welch(Y - Y.mean(1, keepdims=True), fd, nperseg=min(nperseg, Y.shape[1]))
    S = (P / np.median(P, 1, keepdims=True)).mean(0)
    out = []
    for a in anchors:
        m = np.where(np.abs(f - a) <= TOL * a)[0]; j = m[np.argmax(S[m])]
        if 0 < j < len(S) - 1:
            y0, y1, y2 = np.log(S[j - 1:j + 2]); d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
        else:
            d = 0.0
        out.append(float(f[j] + d * (f[1] - f[0])))
    return out


def novelty(train: np.ndarray, test: np.ndarray, k: int = 3, loo: bool = False) -> np.ndarray:
    """Srednia odleglosc do k najblizszych wektorow uczacych po standaryzacji mediana/MAD zbioru uczacego."""
    med = np.median(train, 0); mad = 1.4826 * np.median(np.abs(train - med), 0); mad[mad < 1e-12] = 1.0
    A = (train - med) / mad; B = (test - med) / mad
    D = np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1))
    if loo:
        np.fill_diagonal(D, np.inf)
    return np.sort(D, 1)[:, :k].mean(1)


# ---------------- linie przy kotwicy: nieliniowosc kontaktu (luz, zderzak, sruba) ----------------
def lines_known_excitation(v: np.ndarray, fs: float, fmin: float = 20.0, fmax: float = 500.0, nperseg: int = 1024) -> Dict[str, float]:
    """Wymuszenie harmoniczne: kotwica f0 = szczyt widma w [fmin, fmax]; H = log((P(2f0) + P(3f0)) / P(f0))."""
    f, P = welch(v - np.mean(v), fs, nperseg=min(nperseg, len(v)))
    m = (f > fmin) & (f < fmax); f0 = f[m][np.argmax(P[m])]
    pw = lambda c: P[np.abs(f - c) <= 3 * (f[1] - f[0])].max()
    return {"f0": float(f0), "H": float(np.log((pw(2 * f0) + pw(3 * f0)) / pw(f0)))}


def lines_two_modes(x: np.ndarray, fs: float, band_a: Sequence[float], band_b: Sequence[float], nperseg: int = 2048) -> Dict[str, float]:
    """Wymuszenie losowe: dwa mody fa, fb (szczyty w pasmach); H = log(moc przy {2fa, 2fb, fa+fb} / moc przy {fa, fb})."""
    f, P = welch(x - np.mean(x), fs, nperseg=nperseg)
    def pk(lo, hi):
        m = (f >= lo) & (f <= hi); return f[m][np.argmax(P[m])]
    def pw(c):
        m = np.abs(f - c) <= 1.5; return P[m].max() if m.any() else np.nan
    fa, fb = pk(*band_a), pk(*band_b)
    return {"fa": float(fa), "fb": float(fb),
            "H": float(np.log(np.nansum([pw(2 * fa), pw(2 * fb), pw(fa + fb)]) / (pw(fa) + pw(fb))))}


# ---------------- rezim: czasteczka / pakiet / fala ----------------
def band_envelope(x, fs, lo, hi):
    x = np.asarray(x, float) - np.mean(x); X = np.fft.fft(x); f = np.fft.fftfreq(len(x), 1 / fs)
    Z = np.zeros_like(X); m = (f >= lo) & (f < hi); Z[m] = 2 * X[m]
    return np.abs(np.fft.ifft(Z))


def regime(x: np.ndarray, fs: float, lo: float, hi: float) -> Dict[str, object]:
    """P = udzial 5% najsilniejszych probek energii obwiedni; D = eta/(1-eta), eta = (E e)^2 / E e^2.
    Czasteczka: rzadkie ostre zdarzenia (mikroposlizg, uderzenia); przy pelnym luzie/czestych uderzeniach P wraca do tla."""
    env = band_envelope(x, fs, lo, hi); e2 = env ** 2; k = max(1, int(0.05 * len(e2)))
    eta = float(env.mean() ** 2 / (np.mean(e2) + 1e-30)); D = eta / max(1 - eta, 1e-9)
    return {"P": float(np.sort(e2)[-k:].sum() / e2.sum()), "D": D,
            "rezim": "czasteczka" if D < 0.3 else ("fala" if D > 3.0 else "pakiet")}


def _load(path: str, channels: List[str] | None) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npz":
        d = np.load(p); keys = channels or d.files; return np.stack([np.asarray(d[k], float).ravel() for k in keys])
    if p.suffix == ".npy":
        return np.atleast_2d(np.load(p))
    return np.atleast_2d(np.loadtxt(p, delimiter=",").T)


def main(argv=None):
    ap = argparse.ArgumentParser(description="TIMDR: monitorowanie konstrukcji")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("modes", "lines", "lines2", "regime"):
        s = sub.add_parser(n); s.add_argument("file"); s.add_argument("--fs", type=float, required=True)
        s.add_argument("--channels", default=None)
    sub.choices["modes"].add_argument("--anchors", required=True); sub.choices["modes"].add_argument("--decimate", type=int, default=1)
    sub.choices["lines2"].add_argument("--band-a", required=True); sub.choices["lines2"].add_argument("--band-b", required=True)
    sub.choices["regime"].add_argument("--band", required=True)
    a = ap.parse_args(argv)
    X = _load(a.file, a.channels.split(",") if a.channels else None)
    if a.cmd == "modes":
        out = {"f_hz": anchor_frequencies(X, a.fs, [float(v) for v in a.anchors.split(",")], a.decimate)}
    elif a.cmd == "lines":
        out = lines_known_excitation(X[0], a.fs)
    elif a.cmd == "lines2":
        out = lines_two_modes(X[0], a.fs, [float(v) for v in a.band_a.split(",")], [float(v) for v in a.band_b.split(",")])
    else:
        lo, hi = [float(v) for v in a.band.split(",")]; out = regime(X[0], a.fs, lo, hi)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
