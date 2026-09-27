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


# ---------------- przerwa ciaglosci (K -> G -> M/S) ----------------
ANCH_HBTA = [6.86, 7.42, 17.26, 24.12, 30.08, 32.32]


def _grid_bridge(seed=21, damage=0.0):
    """40 kanalow = 4 linie x 10 punktow (rozstaw 3,5 m), wymuszenie szumem; damage -> zalamanie ksztaltu w punkcie 4 linii 1."""
    rng = np.random.default_rng(seed); fs = 100.0; n = 6000; a = rng.standard_normal(n)
    xpos = np.arange(10) * 3.5; X = np.zeros((40, n))
    for k, f0 in enumerate(ANCH_HBTA):
        mode = np.sin((k % 3 + 1) * np.pi * (xpos + 1.75) / 35.0)
        drive = np.convolve(a, np.exp(-np.arange(200) / 60) * np.sin(2 * np.pi * f0 * np.arange(200) / fs), "same")
        for line in range(4):
            m = mode * (1 + 0.1 * line)
            if line == 1 and damage:
                m = m.copy(); m[4] += damage
            X[line * 10:(line + 1) * 10] += np.outer(m, drive)
    return X + 0.05 * rng.standard_normal(X.shape), a, fs


def test_mode_shapes_golden_against_gia_timdr():
    # GIA-TIMDR core/hbta_modal_curvature.window_shapes na tym samym sygnale (anchors jak tam: 8 kotwic v0.1)
    X, a, fs = _grid_bridge()
    m = T.mode_shapes(X, a, fs, [6.86, 7.42, 9.40, 12.79, 17.26, 24.12, 30.08, 32.32])
    assert m[0][:4] == pytest.approx([-0.04318594037, -0.121734126991, -0.178025679198, -0.203782262091], abs=1e-10)
    A1, C1 = T.discontinuity_maps(m[1], np.arange(40).reshape(4, 10))
    assert float(A1.sum()) == pytest.approx(-0.003783981314716984, abs=1e-12)
    assert float(C1.sum()) == pytest.approx(-0.04165072278566179, abs=1e-12)


def test_discontinuity_flags_local_kink():
    grid = np.arange(40).reshape(4, 10)
    train = [T.mode_shapes(*_grid_bridge(seed=s)[:2], 100.0, ANCH_HBTA) for s in range(30, 42)]
    B = T.DiscontinuityBaseline(train, grid)
    healthy = [T.mode_shapes(*_grid_bridge(seed=s)[:2], 100.0, ANCH_HBTA) for s in range(50, 54)]
    broken = [T.mode_shapes(*_grid_bridge(seed=s, damage=0.4)[:2], 100.0, ANCH_HBTA) for s in range(50, 54)]
    Ftr, Fh, Fb = B.features(train), B.features(healthy), B.features(broken)
    assert T.novelty(Ftr, Fb).min() > T.novelty(Ftr, Fh).max()          # zalamanie ksztaltu wykryte
    za, zc = B.z_maps(broken)
    line, pt = np.unravel_index(int(np.argmax(zc.mean((0, 1)))), zc.shape[2:])
    assert line in (0, 1) and pt == 4                                    # i wskazane miejsce: linia 1, punkt 4
