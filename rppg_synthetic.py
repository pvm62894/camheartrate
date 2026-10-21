"""Synthetic skin-reflectance generator for demos and self-tests.

Simplified dichromatic reflection model of a face ROI seen by a camera:

    C_k(t) = d_k * (1 - A * p_k * b(t)) * (1 + (m(t) + drift(t)) * (1 + chi_k))
             + s(t) + n_k(t)

    d     diffuse skin colour (R, G, B) in 0-255 camera units
    b(t)  normalised blood-volume pulse waveform (systolic peak + dicrotic wave)
    p     relative pulsatile strength per channel, [0.33, 0.77, 0.53]
          (de Haan & Jeanne 2013 standardised skin-tone PPG signature)
    A     pulsatile amplitude; green ends up at ~0.4 % AC/DC, as in real webcams
    m(t)  head-motion luminance changes: slow sway, bursts of nodding, and
          random jerks; chi_k adds a slight colour tilt (ROI drifting over skin)
    s(t)  white specular reflection that grows with motion
    n_k   sensor noise left after spatial averaging (per-pixel sigma / sqrt(N))
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

PULSE_SIGNATURE = np.array([0.33, 0.77, 0.53])
SKIN_RGB = np.array([178.0, 122.0, 98.0])
MOTION_TILT = np.array([0.03, 0.0, -0.03])
PIXEL_NOISE_SIGMA = 6.0


def pulse_waveform(phase: float) -> float:
    """One cardiac cycle (phase in [0, 1)) with systolic peak and dicrotic wave."""
    v = 0.0
    for k in (-1.0, 0.0, 1.0):               # periodic wrap so cycles join smoothly
        x = phase + k
        v += math.exp(-0.5 * ((x - 0.18) / 0.075) ** 2)
        v += 0.45 * math.exp(-0.5 * ((x - 0.48) / 0.11) ** 2)
    return v - 0.312                          # zero mean over a cycle


class SyntheticSkinTrace:
    """Stateful generator: call `sample(t)` with increasing timestamps (seconds)."""

    def __init__(self, bpm: float = 72.0, motion: float = 0.3, seed: int | None = None,
                 skin_rgb=SKIN_RGB, pulse_amplitude: float = 0.006, roi_pixels: int = 15000):
        self.rng = np.random.default_rng(seed)
        self.skin = np.asarray(skin_rgb, dtype=float)
        self.amplitude = pulse_amplitude
        self.sensor_sigma = PIXEL_NOISE_SIGMA / math.sqrt(roi_pixels)
        self.bpm = float(bpm)
        self.motion = float(motion)
        self._t_last: float | None = None
        self._t0: float | None = None
        self._phase = 0.0
        self._hr_walk = 0.0
        self._jerks: list[tuple[float, float, float]] = []   # (start, duration, amplitude)
        self._beats: deque[float] = deque(maxlen=40)
        self._phases = self.rng.uniform(0, 2 * math.pi, size=4)
        self._pick_nod_frequency()
        self.last_motion = (0.0, 0.0, 0.0)   # (sway, nod, jerk) for rendering

    def set_params(self, bpm: float | None = None, motion: float | None = None) -> None:
        if bpm is not None:
            self.bpm = float(min(max(bpm, 40.0), 200.0))
            self._pick_nod_frequency()
        if motion is not None:
            self.motion = float(min(max(motion, 0.0), 1.0))

    def _pick_nod_frequency(self) -> None:
        # Nodding / talking rhythm inside the cardiac band but away from the true HR,
        # so a motion-sensitive method locks onto the wrong peak.
        hr_hz = self.bpm / 60.0
        candidates = [f for f in np.arange(1.0, 3.2, 0.05) if abs(f - hr_hz) > 0.45]
        self._nod_hz = float(self.rng.choice(candidates))

    # -- generation ----------------------------------------------------------

    def _motion(self, t: float, dt: float) -> tuple[float, float, float]:
        level = self.motion
        sway = 0.008 * math.sin(2 * math.pi * 0.21 * t + self._phases[0]) \
             + 0.006 * math.sin(2 * math.pi * 0.37 * t + self._phases[1])
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 0.07 * t + self._phases[2])
        nod = 0.018 * envelope * math.sin(2 * math.pi * self._nod_hz * t + self._phases[3])

        if level > 0 and self.rng.random() < 0.6 * level * dt:
            self._jerks.append((t, float(self.rng.uniform(0.25, 0.8)),
                                float(self.rng.choice([-1, 1]) * self.rng.uniform(0.02, 0.05))))
        jerk = 0.0
        alive = []
        for start, dur, amp in self._jerks:
            u = (t - start) / dur
            if u < 1.0:
                jerk += amp * 0.5 * (1 - math.cos(2 * math.pi * u))
                alive.append((start, dur, amp))
        self._jerks = alive
        return level * sway, level * nod, jerk

    def sample(self, t: float) -> np.ndarray:
        """Spatially averaged (R, G, B) of the skin ROI at time t."""
        if self._t_last is None:
            self._t_last = self._t0 = t
        dt = max(0.0, t - self._t_last)
        self._t_last = t

        # Heart: respiratory sinus arrhythmia (0.24 Hz) plus a slow random walk.
        self._hr_walk += -0.2 * self._hr_walk * dt + 0.006 * math.sqrt(dt) * self.rng.standard_normal()
        rsa = 0.035 * math.sin(2 * math.pi * 0.24 * (t - self._t0))
        f_hr = self.bpm / 60.0 * (1.0 + rsa + self._hr_walk)
        prev = self._phase
        self._phase += f_hr * dt
        if math.floor(self._phase) > math.floor(prev):
            frac = (math.floor(self._phase) - prev) / (self._phase - prev)
            self._beats.append(t - dt + frac * dt)
        bvp = pulse_waveform(self._phase % 1.0)

        sway, nod, jerk = self._motion(t - self._t0, dt)
        self.last_motion = (sway, nod, jerk)
        drift = 0.004 * math.sin(2 * math.pi * 0.03 * (t - self._t0))
        lum = sway + nod + jerk + drift

        diffuse = self.skin * (1.0 - self.amplitude * PULSE_SIGNATURE * bvp)
        rgb = diffuse * (1.0 + lum * (1.0 + MOTION_TILT))
        rgb += 30.0 * (nod + jerk)                         # specular, white
        rgb += self.rng.normal(0.0, self.sensor_sigma, 3)
        return rgb

    @property
    def truth(self) -> dict:
        """Ground truth for the dashboard.

        `bpm` is the slider setpoint; `bpm_actual` is the mean rate of the beats
        generated over the last ~10 s, which wanders around the setpoint because
        of the simulated sinus arrhythmia.
        """
        rmssd = actual = None
        if len(self._beats) >= 6:
            beats = np.array(self._beats)
            ibi = np.diff(beats[beats > beats[-1] - 10.0])
            if len(ibi) >= 3:
                actual = float(60.0 / ibi.mean())
            ibi20 = np.diff(beats[-20:])
            rmssd = float(np.sqrt(np.mean(np.diff(ibi20) ** 2)) * 1000.0)
        return {"bpm": round(self.bpm, 1), "motion": round(self.motion, 2),
                "bpm_actual": None if actual is None else round(actual, 1),
                "rmssd": None if rmssd is None else round(rmssd, 1)}


# --------------------------------------------------------------------------
# Rendering for the dashboard's video panel
# --------------------------------------------------------------------------

def _roi_shapes(cx: int, cy: int) -> dict[str, np.ndarray]:
    forehead = np.array([[-65, -135], [-30, -150], [30, -150], [65, -135],
                         [72, -95], [25, -85], [-25, -85], [-72, -95]])
    left = np.array([[-95, -20], [-55, -30], [-40, 0], [-50, 40], [-85, 45], [-105, 15]])
    right = left * np.array([-1, 1])
    off = np.array([cx, cy])
    return {name: (pts + off).astype(np.int32).reshape(-1, 1, 2)
            for name, pts in (("forehead", forehead), ("left_malar", left), ("right_malar", right))}


def roi_pixel_count(width: int = 640, height: int = 480) -> int:
    mask = np.zeros((height, width), np.uint8)
    for poly in _roi_shapes(width // 2, height // 2 + 20).values():
        cv2.fillConvexPoly(mask, poly, 255)
    return int(cv2.countNonZero(mask))


class SyntheticRenderer:
    """Draws a plain stand-in face whose skin is filled with the synthetic RGB mean."""

    def __init__(self, width: int = 640, height: int = 480):
        self.w, self.h = width, height
        ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        top, bottom = np.array([62, 58, 60], np.float32), np.array([34, 31, 33], np.float32)
        self.background = np.broadcast_to(top + (bottom - top) * ramp, (height, width, 3)).astype(np.uint8)

    def render(self, rgb: np.ndarray, motion: tuple[float, float, float]):
        """Frame plus the ROI outline in 0-1 coordinates, for the browser to draw."""
        frame = self.background.copy()
        sway, nod, jerk = motion
        cx = int(self.w / 2 + 2500 * sway + 1500 * jerk)
        cy = int(self.h / 2 + 20 + 1200 * nod)
        skin = tuple(int(np.clip(v, 0, 255)) for v in rgb[::-1])   # RGB -> BGR

        cv2.ellipse(frame, (cx, cy), (130, 175), 0, 0, 360, skin, -1, cv2.LINE_AA)
        for ex in (-50, 50):
            cv2.ellipse(frame, (cx + ex, cy - 50), (20, 8), 0, 0, 360, (52, 44, 46), -1, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy + 95), (38, 10), 0, 0, 180, (70, 62, 120), 3, cv2.LINE_AA)

        outline = [np.round(p.reshape(-1, 2) / (self.w, self.h), 4).tolist()
                   for p in _roi_shapes(cx, cy).values()]
        return frame, outline
