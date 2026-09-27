"""Testy timdr_shm.py: wartosci wzorcowe = wyniki implementacji walidowanych w GIA-TIMDR na tych samych sygnalach (2026-09-27)."""
import numpy as np
import pytest

import timdr_shm as T


def _bridge(seed=11, fs=825.8, T_s=120):
    rng = np.random.default_rng(seed); n = int(T_s * fs); t = np.arange(n) / fs
    x = rng.standard_normal((2, n)) * 0.2
    for fm in (2.6, 2.95, 5.3):
        x[0] += np.sin(2 * np.pi * fm * t + rng.uniform(0, 6)); x[1] += 0.7 * np.sin(2 * np.pi * fm * t + rng.uniform(0, 6))
    return x, fs


def test_anchor_tracks_shifted_modes_and_matches_validated():
    x, fs = _bridge()
    f = T.anchor_frequencies(x, fs, [2.58, 2.92, 5.32], q=32)
    assert f == pytest.approx([2.60, 2.95, 5.30], abs=0.01)          # samokorekta do prawdziwych modow


def test_novelty_matches_validated():
    rng = np.random.default_rng(3); A = rng.standard_normal((15, 4)); B = rng.standard_normal((5, 4)) + 0.5
    s = T.novelty(A, B)
    assert s.shape == (5,) and np.all(s > 0)
    assert T.novelty(A, A, loo=True).mean() < T.novelty(A, B).mean()


def test_lines_two_modes_grow_with_nonlinearity():
    fs = 322.58; n = 8192; t = np.arange(n) / fs; rng = np.random.default_rng(5)
    base = np.sin(2 * np.pi * 55 * t) + np.sin(2 * np.pi * 72 * t) + 0.3 * rng.standard_normal(n)
    h = [T.lines_two_modes(base + c * base ** 2, fs, (45, 62), (64, 80))["H"] for c in (0.0, 0.05, 0.2)]
    assert h[0] < h[1] < h[2]


def test_lines_known_excitation_and_regime():
    fs = 5000.0; t = np.arange(20000) / fs
    v0 = np.sin(2 * np.pi * 118.4 * t); v1 = v0 + 0.05 * v0 ** 2 + 0.03 * v0 ** 3
    a, b = T.lines_known_excitation(v0 + 1e-4 * np.random.default_rng(1).standard_normal(len(t)), fs), T.lines_known_excitation(v1, fs)
    assert abs(a["f0"] - 118.4) < 3 and b["H"] > a["H"] + 3
    rng = np.random.default_rng(2); x = 0.01 * rng.standard_normal(200000); x[::20000] += 1.0
    r = T.regime(x, 5e6, 100e3, 900e3); assert r["P"] > 0.3
    assert T.regime(rng.standard_normal(200000), 5e6, 100e3, 900e3)["rezim"] == "fala"   # szum = tlo 'polowe'


def test_golden_values_against_gia_timdr():
    # liczby z GIA-TIMDR core/kw51_modal_anchor.rec_features na tym samym sygnale _bridge() (kanaly 0 i 2)
    x, fs = _bridge()
    assert T.anchor_frequencies(x, fs, [2.58, 2.92, 5.32], q=32) == pytest.approx(
        [2.6003072728071692, 2.9500985367110513, 5.300391251146017], abs=1e-12)


def test_golden_lines_and_regime_against_gia_timdr():
    # GIA-TIMDR core/real_lanl_modal_anchor.anchored_lines (ruch wzgledny pieter) i core/orion_bolt.seg_features
    fs = 322.58; n = 8192; t = np.arange(n) / fs; rng = np.random.default_rng(5)
    base = np.sin(2 * np.pi * 55 * t) + np.sin(2 * np.pi * 72 * t) + 0.3 * rng.standard_normal(n)
    assert T.lines_two_modes(base + 0.05 * base ** 2, fs, (45, 62), (64, 80))["H"] == pytest.approx(-6.04606878563107, abs=1e-12)
    fs5 = 5e6; m = 400000; tt = np.arange(m) / fs5; r2 = np.random.default_rng(8)
    v = np.sin(2 * np.pi * 118.4 * tt) + 0.05 * np.sin(2 * np.pi * 236.8 * tt) + 0.001 * r2.standard_normal(m)
    a = 0.01 * r2.standard_normal(m); a[::40000] += 1.0
    assert T.lines_known_excitation(T._decim(v, 1000), fs5 / 1000)["H"] == pytest.approx(-5.701713483452477, abs=1e-12)
    r = T.regime(a, fs5, 100e3, 900e3)
    assert r["P"] == pytest.approx(0.35500346588028664, abs=1e-12) and r["D"] == pytest.approx(1.8231072796741294, abs=1e-12)
