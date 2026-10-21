"""Self-test for the rPPG DSP pipeline.

Run:  python test_dsp.py      (or: python -m pytest test_dsp.py)

The main check feeds a synthetic 72 BPM pulsatile RGB trace corrupted by
head-motion luminance spikes, sensor noise and webcam frame-timing jitter
through the streaming engine and requires POS to recover 72.0 +/- 1.0 BPM.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from rppg_dsp import ALGORITHMS, FS, RPPGEngine, estimate_spectrum, extract_bvp, resample_uniform
from rppg_synthetic import SyntheticSkinTrace

SKIN = np.array([178.0, 122.0, 98.0])
PULSE_SIGNATURE = np.array([0.33, 0.77, 0.53])


def webcam_timestamps(duration: float, rng: np.random.Generator, fps: float = 30.0) -> np.ndarray:
    """Frame times at `fps` with +/-3 ms jitter and ~3 % dropped frames."""
    t, out = 0.0, []
    while t < duration:
        if rng.random() > 0.03:
            out.append(t)
        t += 1.0 / fps + rng.normal(0.0, 0.003)
    return np.array(out)


def motion_spike_trace(duration: float = 20.0, bpm: float = 72.0, seed: int = 7, fps: float = 30.0):
    """Pulse + luminance spikes, built independently of rppg_synthetic."""
    rng = np.random.default_rng(seed)
    t = webcam_timestamps(duration, rng, fps)
    f = bpm / 60.0
    pulse = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t + 0.8)

    lum = np.zeros_like(t)
    for start in rng.uniform(0.5, duration - 1.0, size=10):     # head-motion spikes
        dur = rng.uniform(0.2, 0.5)
        amp = rng.choice([-1, 1]) * rng.uniform(0.02, 0.05)       # 2-5 %, ~10x the pulse
        u = (t - start) / dur
        inside = (u >= 0) & (u < 1)
        lum[inside] += amp * 0.5 * (1 - np.cos(2 * np.pi * u[inside]))
    lum[t > duration * 0.6] += 0.03                               # head turn: lasting step

    rgb = SKIN * (1.0 - 0.005 * PULSE_SIGNATURE * pulse[:, None]) * (1.0 + lum[:, None])
    rgb += rng.normal(0.0, 0.05, rgb.shape)
    return t, rgb


def run_engine(t, rgb, algorithm: str, fs: float = FS) -> RPPGEngine:
    """Stream samples through the engine as the live server does."""
    eng = RPPGEngine(algorithm=algorithm, fs=fs)
    for ti, (r, g, b) in zip(t, rgb):
        eng.add_sample(float(ti), float(r), float(g), float(b))
        eng.process(float(ti))
    return eng


# -- tests -------------------------------------------------------------------

def test_resampling_is_uniform_and_exact_for_linear_signals():
    rng = np.random.default_rng(0)
    t = webcam_timestamps(10.0, rng)
    rgb = np.stack([2 * t + 1, -t + 5, 0.5 * t], axis=1)
    grid, out = resample_uniform(t, rgb)
    assert np.allclose(np.diff(grid), 1.0 / FS)
    assert grid[-1] == t[-1]
    assert np.allclose(out, np.stack([2 * grid + 1, -grid + 5, 0.5 * grid], axis=1), atol=1e-9)


def test_pos_recovers_72_bpm_with_motion_spikes():
    t, rgb = motion_spike_trace(bpm=72.0)
    eng = run_engine(t, rgb, "POS")
    print(f"  POS   : raw {eng.bpm_raw:6.2f}  EMA {eng.bpm:6.2f} BPM  SQI {eng.sqi:5.1f}%  "
          f"SNR {eng.snr_db:+5.1f} dB  status {eng.status()}")
    assert abs(eng.bpm_raw - 72.0) <= 1.0
    assert abs(eng.bpm - 72.0) <= 1.0


def test_pos_at_60_fps():
    t, rgb = motion_spike_trace(bpm=72.0, fps=60.0)
    eng = run_engine(t, rgb, "POS", fs=60.0)
    print(f"  POS @60 Hz: raw {eng.bpm_raw:6.2f}  EMA {eng.bpm:6.2f} BPM  SQI {eng.sqi:5.1f}%  "
          f"({len(t)} frames)")
    assert abs(eng.bpm - 72.0) <= 1.0


def test_strong_second_harmonic_is_not_reported_as_heart_rate():
    """Real rPPG often puts more power at 2x the heart rate than at the heart
    rate itself; the estimate must still be the fundamental."""
    rng = np.random.default_rng(4)
    t = webcam_timestamps(15.0, rng)
    f = 75 / 60.0
    pulse = 0.6 * np.sin(2 * np.pi * f * t) + 1.0 * np.sin(4 * np.pi * f * t + 0.5)
    rgb = SKIN * (1.0 - 0.005 * PULSE_SIGNATURE * pulse[:, None]) + rng.normal(0.0, 0.05, (len(t), 3))
    eng = run_engine(t, rgb, "POS")
    print(f"  2nd harmonic 1.7x the fundamental -> raw {eng.bpm_raw:6.2f}  readout {eng.bpm:6.2f} BPM")
    assert abs(eng.bpm - 75.0) <= 1.5


def test_patch_fusion_ignores_a_bad_patch():
    """Forehead + two cheeks: two patches carry a weak 70 bpm pulse, one is
    swamped by a pulse-coloured artefact at 110 bpm (e.g. hair moving over it).
    Averaging all pixels into one colour gets misled; per-patch weighting does not."""
    rng = np.random.default_rng(8)
    t = webcam_timestamps(15.0, rng)
    pulse = np.sin(2 * np.pi * 70 / 60 * t)
    junk = 5.0 * np.sin(2 * np.pi * 110 / 60 * t + 1.0)
    good = [SKIN * (1.0 - 0.004 * PULSE_SIGNATURE * pulse[:, None]) + rng.normal(0.0, 0.08, (len(t), 3))
            for _ in range(2)]
    bad = SKIN * (1.0 - 0.004 * PULSE_SIGNATURE * (pulse + junk)[:, None]) + rng.normal(0.0, 0.08, (len(t), 3))
    patches = np.stack(good + [bad], axis=1)                 # (n, 3 patches, rgb)

    pooled = run_engine(t, patches.mean(axis=1), "POS")
    fused = RPPGEngine("POS")
    for ti, rgb in zip(t, patches):
        fused.add_patches(float(ti), rgb)
        fused.process(float(ti))
    print(f"  one pooled colour: raw {pooled.bpm_raw:6.2f} BPM | per-patch fusion: raw {fused.bpm_raw:6.2f} BPM, "
          f"weights {[round(w, 2) for w in fused.patch_weights]}")
    assert abs(fused.bpm_raw - 70.0) <= 1.5


def test_readout_ignores_a_burst_of_motion():
    """A 3 s burst of rhythmic, blood-coloured motion (2.5x the pulse, at 138 bpm)
    must not drag the displayed reading away. POS cannot cancel this kind: it is
    what the ROI sliding over differently perfused skin looks like. Before the
    continuity / median / jump-hold logic the readout swung ~55 bpm here."""
    rng = np.random.default_rng(21)
    t = webcam_timestamps(30.0, rng)
    f = 70 / 60.0
    pulse = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t + 0.8)
    burst = ((t > 20.0) & (t < 23.0)) * 2.5 * np.sin(2 * np.pi * 2.3 * t)
    rgb = SKIN * (1.0 - 0.005 * PULSE_SIGNATURE * (pulse + burst)[:, None])
    rgb += rng.normal(0.0, 0.05, rgb.shape)
    eng = RPPGEngine("POS")
    worst = 0.0
    for ti, (r, g, b) in zip(t, rgb):
        eng.add_sample(float(ti), float(r), float(g), float(b))
        eng.process(float(ti))
        if ti > 12.0 and eng.bpm is not None:
            worst = max(worst, abs(eng.bpm - 70.0))
    print(f"  worst readout error during and after the burst: {worst:.2f} BPM")
    assert worst <= 3.0


def test_pos_beats_raw_green_under_motion():
    t, rgb = motion_spike_trace(bpm=72.0)
    sqi = {}
    for algo in ALGORITHMS:
        eng = run_engine(t, rgb, algo)
        sqi[algo] = eng.sqi
        print(f"  {algo:<6}: raw {eng.bpm_raw:6.2f}  readout {eng.bpm or float('nan'):6.2f} BPM  SQI {eng.sqi:5.1f}%")
    assert sqi["POS"] > sqi["GREEN"] + 15.0


def test_pos_tracks_full_range_on_synthetic_generator():
    worst = 0.0
    for bpm in (60, 75, 90, 110, 130, 160):
        gen = SyntheticSkinTrace(bpm=bpm, motion=0.6, seed=bpm)
        t = webcam_timestamps(20.0, np.random.default_rng(bpm))
        rgb = np.array([gen.sample(float(ti)) for ti in t])
        bvp = extract_bvp(resample_uniform(t, rgb)[1], "POS", FS)
        est = estimate_spectrum(bvp).peak_hz * 60.0
        err = abs(est - bpm)
        worst = max(worst, err)
        print(f"  synthetic {bpm:3d} BPM (motion 60%) -> POS {est:6.2f} BPM  (error {err:4.2f})")
    assert worst <= 2.5


def test_moderate_but_steady_signal_still_shows_a_reading():
    """Dim light: SQI around 40-50 %, never a solid lock, but the estimate holds
    steady. That must still produce a number (and the right one)."""
    import rppg_dsp
    rng = np.random.default_rng(12)
    t = webcam_timestamps(20.0, rng)
    pulse = np.sin(2 * np.pi * 74 / 60 * t)
    rgb = SKIN * (1.0 - 0.0013 * PULSE_SIGNATURE * pulse[:, None]) + rng.normal(0.0, 0.12, (len(t), 3))
    saved = rppg_dsp.SQI_LOCKED
    rppg_dsp.SQI_LOCKED = 101.0                    # a solid lock is impossible: only "steady" can qualify
    try:
        eng = run_engine(t, rgb, "POS")
    finally:
        rppg_dsp.SQI_LOCKED = saved
    print(f"  SQI {eng.sqi:5.1f}% (never a solid lock)  readout {eng.bpm and round(eng.bpm, 2)}")
    assert eng.bpm is not None and abs(eng.bpm - 74.0) <= 1.5


def test_noise_only_is_not_locked():
    rng = np.random.default_rng(3)
    t = webcam_timestamps(15.0, rng)
    rgb = SKIN + rng.normal(0.0, 0.3, (len(t), 3))
    eng = run_engine(t, rgb, "POS")
    print(f"  noise only: SQI {eng.sqi:5.1f}%  status {eng.status()}")
    assert eng.status() != "LOCKED"


def test_hrv_rmssd_is_plausible():
    gen = SyntheticSkinTrace(bpm=70, motion=0.0, seed=11)
    t = webcam_timestamps(15.0, np.random.default_rng(11))
    rgb = np.array([gen.sample(float(ti)) for ti in t])
    eng = run_engine(t, rgb, "POS")
    truth = gen.truth["rmssd"]
    print(f"  RMSSD estimate {eng.rmssd} ms vs injected {truth} ms")
    assert eng.rmssd is not None and 5.0 < eng.rmssd < 150.0


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        print(f"{name}")
        try:
            fn()
            print("  PASS")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
