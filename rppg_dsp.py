"""Remote photoplethysmography (rPPG) signal processing engine.

Turns a stream of spatially averaged skin colour samples (t, R, G, B) into a
blood volume pulse (BVP) waveform, a heart-rate estimate, a spectral signal
quality index (SQI) and a short-term HRV estimate (RMSSD).

Per update:
    irregular (t, R, G, B) samples
      -> linear resampling onto a uniform grid at the camera rate (30 or 60 Hz)
      -> pulse extraction (POS / CHROM / GREEN), per skin patch
      -> linear detrend + zero-phase Butterworth bandpass, 0.75-3.5 Hz
      -> Welch PSD per patch -> quality-weighted fusion
      -> dominant peak (continuity + harmonic guard) -> BPM, SNR, SQI
      -> median + outlier-gated, quality-weighted smoothing -> displayed BPM
      -> peak detection on the BVP -> inter-beat intervals -> RMSSD
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

FS = 30.0                    # default uniform resampling rate (Hz)
BAND_HZ = (0.75, 3.5)        # 45-210 BPM
BUFFER_SECONDS = 15.0
MIN_SECONDS = 5.0            # data needed before the first HR estimate
UPDATE_INTERVAL = 0.25       # seconds between spectral HR estimates
POS_WINDOW_S = 1.6           # POS / CHROM window (48 samples at 30 Hz)
SQI_HALF_WIDTH_HZ = 0.15
BIN_HZ = 0.0075              # target PSD bin width (0.45 BPM) via zero-padding
MAX_GAP_S = 1.0              # a longer gap (skin lost) restarts the buffer
DISPLAY_SECONDS = 6.0
ALGORITHMS = ("POS", "CHROM", "GREEN")

# Status thresholds on SQI (%). Pure in-band noise measures ~20-37 %.
SQI_LOCKED = 50.0
SQI_WEAK = 40.0
SQI_HYSTERESIS = 3.0         # a status change needs the SQI this far past the threshold

# Readout stabilisation
CONTINUITY_HZ = 0.2          # prefer a peak within +/-12 BPM of the current reading...
CONTINUITY_RATIO = 0.5       # ...if it has at least half the power of the tallest peak
MEDIAN_WINDOW_S = 2.0        # median over the last 2 s of good spectral estimates
SUBHARMONIC_RATIO = 0.35     # a peak at f/2 this strong means f is a harmonic
STEADY_S = 4.0               # first reading also allowed once estimates hold steady this long...
STEADY_BPM = 2.5             # ...within this many BPM of each other (at SQI >= SQI_WEAK)
JUMP_BPM = 12.0              # a bigger change must persist...
JUMP_HOLD_S = 2.5            # ...this long before the readout follows it


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def resample_uniform(t, rgb, fs: float = FS):
    """Linearly interpolate irregular samples onto a uniform grid that ends at t[-1]."""
    t = np.asarray(t, dtype=float)
    rgb = np.asarray(rgb, dtype=float)
    n = int(math.floor((t[-1] - t[0]) * fs)) + 1
    grid = t[-1] - np.arange(n - 1, -1, -1) / fs
    f = interp1d(t, rgb, axis=0, kind="linear", assume_sorted=True,
                 bounds_error=False, fill_value="extrapolate")
    return grid, f(grid)


@lru_cache(maxsize=8)
def _butter(fs: float):
    return signal.butter(4, BAND_HZ, btype="bandpass", fs=fs)


def bandpass(x: np.ndarray, fs: float = FS) -> np.ndarray:
    """Zero-phase 4th-order Butterworth bandpass (0.75-3.5 Hz)."""
    b, a = _butter(float(fs))
    if len(x) <= 3 * max(len(a), len(b)):
        return x - np.mean(x)
    return signal.filtfilt(b, a, x)


def window_length(fs: float) -> int:
    return max(4, round(POS_WINDOW_S * fs))


def pos_bvp(rgb: np.ndarray, window: int) -> np.ndarray:
    """Plane-Orthogonal-to-Skin (Wang et al., 2017) with sliding overlap-add.

    Each window is divided by its own mean so the skin's DC colour becomes
    [1, 1, 1]. Pure intensity changes (motion, exposure) also lie along
    [1, 1, 1], so the projection P = [[0, 1, -1], [-2, 1, 1]] removes them.
    """
    n = rgb.shape[0]
    L = min(window, n)
    win = sliding_window_view(rgb, L, axis=0)                  # (n-L+1, 3, L)
    cn = win / (win.mean(axis=2, keepdims=True) + 1e-12)
    r, g, b = cn[:, 0], cn[:, 1], cn[:, 2]
    s1 = g - b
    s2 = g + b - 2.0 * r
    alpha = s1.std(axis=1, keepdims=True) / (s2.std(axis=1, keepdims=True) + 1e-8)
    h = s1 + alpha * s2
    h -= h.mean(axis=1, keepdims=True)

    m = n - L + 1
    out = np.zeros(n)
    counts = np.zeros(n)
    for k in range(L):
        out[k:k + m] += h[:, k]
        counts[k:k + m] += 1.0
    # Averaging instead of summing keeps the newest samples (covered by fewer
    # windows) on the same scale as the rest. Blood absorbs light, so POS output
    # falls as blood volume rises; negate so systolic peaks point up.
    return -out / counts


def chrom_bvp(rgb: np.ndarray, window: int, fs: float) -> np.ndarray:
    """Chrominance-based rPPG (de Haan & Jeanne, 2013).

    Normalises by a 1.6 s moving mean, builds X = 3R - 2G and
    Y = 1.5R + G - 1.5B, bandpasses both, and combines S = Xf - alpha * Yf
    with alpha = std(Xf) / std(Yf) estimated per half-overlapping window.
    """
    n = rgb.shape[0]
    L = min(window, n)
    cn = rgb / (uniform_filter1d(rgb, size=L, axis=0, mode="nearest") + 1e-12)
    xs = 3.0 * cn[:, 0] - 2.0 * cn[:, 1]
    ys = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
    xf = bandpass(xs - xs.mean(), fs)
    yf = bandpass(ys - ys.mean(), fs)

    hop = max(1, L // 2)
    starts = list(range(0, n - L + 1, hop))
    if starts[-1] != n - L:
        starts.append(n - L)
    centres = [s + (L - 1) / 2.0 for s in starts]
    alphas = [xf[s:s + L].std() / (yf[s:s + L].std() + 1e-8) for s in starts]
    alpha = np.interp(np.arange(n), centres, alphas)
    return xf - alpha * yf


def green_bvp(rgb: np.ndarray) -> np.ndarray:
    """Raw green channel (Verkruysse et al., 2008), inverted so peaks = systole."""
    g = rgb[:, 1]
    return -(g / g.mean() - 1.0)


def extract_bvp(rgb: np.ndarray, algorithm: str = "POS", fs: float = FS) -> np.ndarray:
    """Pulse extraction followed by linear detrend and bandpass filtering."""
    if algorithm == "POS":
        raw = pos_bvp(rgb, window_length(fs))
    elif algorithm == "CHROM":
        raw = chrom_bvp(rgb, window_length(fs), fs)
    else:
        raw = green_bvp(rgb)
    return bandpass(signal.detrend(raw, type="linear"), fs)


@dataclass
class Spectrum:
    freqs: np.ndarray
    power: np.ndarray
    peak_hz: float
    snr_db: float
    sqi: float


def _nfft(fs: float, nper: int) -> int:
    return max(nper, 2 ** math.ceil(math.log2(fs / BIN_HZ)))


def welch_psd(bvp: np.ndarray, fs: float = FS):
    """Welch PSD: Hann, 10 s segments, 50 % overlap, zero-padded to ~0.45 BPM bins."""
    nper = min(len(bvp), int(fs * 10))
    return signal.welch(bvp, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2,
                        nfft=_nfft(fs, nper), detrend="constant")


def estimate_spectrum(bvp: np.ndarray, fs: float = FS, prefer_hz: float | None = None) -> Spectrum:
    f, p = welch_psd(bvp, fs)
    return analyse_spectrum(f, p, prefer_hz)


def analyse_spectrum(f: np.ndarray, p: np.ndarray, prefer_hz: float | None = None) -> Spectrum:
    """Dominant in-band peak, and spectral SNR / SQI.

    With `prefer_hz` (the current reading), a peak near it wins over a taller
    one elsewhere unless that one has more than twice its power. This stops
    single noise bursts from yanking the estimate across the band.

    SNR = power within +/-0.15 Hz of f_max and of 2 f_max, divided by all other
    power in the 0.75-3.5 Hz band. SQI expresses the same quantity as the
    fraction of in-band power that is cardiac: 100 * S / (S + N).
    """
    band = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])
    idx = np.flatnonzero(band)
    k = int(idx[np.argmax(p[idx])])
    local = [int(idx[j]) for j in signal.find_peaks(p[idx])[0]]
    if prefer_hz is not None:
        near = [j for j in local if abs(f[j] - prefer_hz) <= CONTINUITY_HZ]
        if near:
            k_near = max(near, key=lambda j: p[j])
            if p[k_near] >= CONTINUITY_RATIO * p[k]:
                k = k_near
    # Harmonic guard: a sharp systolic upstroke and the dicrotic notch can give
    # the 2nd harmonic more power than the fundamental. A real heart rate has
    # nothing at half its frequency, so a clear peak there is the heart rate.
    half = [j for j in local if abs(f[j] - f[k] / 2.0) <= 0.1]
    if half:
        k_half = max(half, key=lambda j: p[j])
        if p[k_half] >= SUBHARMONIC_RATIO * p[k]:
            k = k_half
    peak = float(f[k])
    if 0 < k < len(p) - 1:
        y0, y1, y2 = p[k - 1], p[k], p[k + 1]
        den = y0 - 2.0 * y1 + y2
        if den != 0:
            peak += 0.5 * (y0 - y2) / den * (f[1] - f[0])

    near = (np.abs(f - peak) <= SQI_HALF_WIDTH_HZ) | (np.abs(f - 2.0 * peak) <= SQI_HALF_WIDTH_HZ)
    s = float(p[band & near].sum())
    noise = float(p[band & ~near].sum())
    snr_db = 10.0 * math.log10(s / noise) if s > 0 and noise > 0 else 0.0
    sqi = 100.0 * s / (s + noise) if (s + noise) > 0 else 0.0
    return Spectrum(f, p, peak, snr_db, sqi)


def fuse_patches(bvps: list[np.ndarray], fs: float, prefer_hz: float | None = None):
    """Agreement-weighted fusion of several skin patches (e.g. forehead + cheeks).

    1. Each patch's spectrum is normalised to unit in-band power and the
       patches are averaged, weighted by their own SQI squared, to find the
       rate most of them agree on.
    2. Each patch is then re-weighted by the share of its power at that rate
       (and its harmonic), squared. A shadowed cheek (noise everywhere) or a
       patch with hair moving over it (a clean rhythm at the wrong rate) ends
       up with almost no say.

    Spectra are fused rather than waveforms because the pulse reaches each
    patch at a slightly different moment; power spectra ignore that phase.
    Returns (fused BVP, fused Spectrum, weights).
    """
    if len(bvps) == 1:
        return bvps[0], estimate_spectrum(bvps[0], fs, prefer_hz), [1.0]
    psds = [welch_psd(b, fs) for b in bvps]
    f = psds[0][0]
    band = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])
    norm = [p / (p[band].sum() + 1e-20) for _, p in psds]

    def combine(weights):
        if sum(weights) <= 0:
            weights = [1.0] * len(norm)
        total = sum(weights)
        return sum(w * n for w, n in zip(weights, norm)) / total, [w / total for w in weights]

    consensus = analyse_spectrum(f, combine([(analyse_spectrum(f, n).sqi / 100.0) ** 2 for n in norm])[0],
                                 prefer_hz).peak_hz
    at = band & ((np.abs(f - consensus) <= SQI_HALF_WIDTH_HZ) | (np.abs(f - 2 * consensus) <= SQI_HALF_WIDTH_HZ))
    fused, weights = combine([float(n[at].sum()) ** 2 for n in norm])
    bvp = sum(w * b / (np.std(b) + 1e-12) for w, b in zip(weights, bvps))
    return bvp, analyse_spectrum(f, fused, prefer_hz), weights


def _refine_peaks(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Sub-sample peak positions by parabolic interpolation."""
    pos = peaks.astype(float)
    inner = (peaks > 0) & (peaks < len(x) - 1)
    k = peaks[inner]
    y0, y1, y2 = x[k - 1], x[k], x[k + 1]
    den = y0 - 2.0 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(den != 0, 0.5 * (y0 - y2) / den, 0.0)
    pos[inner] += np.clip(delta, -0.5, 0.5)
    return pos


def find_beats(bvp: np.ndarray, fs: float, bpm: float | None):
    """Systolic peaks and diastolic troughs of a filtered BVP waveform."""
    period = 60.0 / bpm if bpm else 0.6
    distance = max(1, int(0.6 * period * fs))
    prominence = 0.3 * float(np.std(bvp))
    peaks, _ = signal.find_peaks(bvp, distance=distance, prominence=prominence)
    troughs, _ = signal.find_peaks(-bvp, distance=distance, prominence=prominence)
    return peaks, troughs


def rmssd_ms(bvp: np.ndarray, fs: float, bpm: float) -> float | None:
    """Root mean square of successive inter-beat-interval differences, in ms."""
    peaks, _ = find_beats(bvp, fs, bpm)
    if len(peaks) < 6:
        return None
    beat_t = _refine_peaks(bvp, peaks) / fs
    ibi = np.diff(beat_t)
    med = np.median(ibi)
    ok = np.abs(ibi - med) < 0.25 * med       # drop missed / extra beats
    succ = np.diff(ibi)
    valid = ok[1:] & ok[:-1]
    if valid.sum() < 4:
        return None
    return float(np.sqrt(np.mean(succ[valid] ** 2)) * 1000.0)


def _r(x, nd: int = 4):
    return np.round(np.asarray(x, dtype=float), nd).tolist()


# --------------------------------------------------------------------------
# Streaming engine
# --------------------------------------------------------------------------

class RPPGEngine:
    """Rolling-buffer rPPG estimator. Not thread-safe; callers hold a lock.

    `add_sample` is O(1) and can run at the full camera rate. The spectral
    estimate runs at most every `update_interval` seconds inside `update`.
    """

    def __init__(self, algorithm: str = "POS", fs: float = FS,
                 buffer_seconds: float = BUFFER_SECONDS, min_seconds: float = MIN_SECONDS,
                 update_interval: float = UPDATE_INTERVAL, display_seconds: float = DISPLAY_SECONDS):
        self.fs = float(fs)
        self.buffer_seconds = buffer_seconds
        self.min_seconds = min_seconds
        self.update_interval = update_interval
        self.display_seconds = display_seconds
        self.algorithm = "POS"
        self.set_algorithm(algorithm)
        self.reset()

    # -- state -------------------------------------------------------------

    def reset(self) -> None:
        self._t: deque[float] = deque()
        self._rgb: deque[np.ndarray] = deque()          # (patches, 3) per sample
        self._clear_estimates()

    def _clear_estimates(self) -> None:
        self._last_update = -math.inf
        self._display: dict = {}
        self._raw_hist: deque[tuple[float, float]] = deque()      # (time, bpm) with decent SQI
        self._steady: deque[tuple[float, float]] = deque()        # same, over STEADY_S
        self._jump_since: float | None = None
        self.spectrum: Spectrum | None = None
        self.bpm: float | None = None
        self.bpm_raw: float | None = None
        self.sqi = 0.0
        self.snr_db: float | None = None
        self.rmssd: float | None = None
        self._level = 0                 # 0 noise, 1 weak, 2 locked (with hysteresis)
        self.patch_weights: list[float] = []

    def set_algorithm(self, name: str) -> None:
        name = name.upper()
        if name not in ALGORITHMS:
            raise ValueError(f"unknown algorithm {name!r}; choose from {ALGORITHMS}")
        if name != self.algorithm:
            self.algorithm = name
            self._clear_estimates()     # raw RGB buffer is kept: switching is instant

    def add_sample(self, t: float, r: float, g: float, b: float) -> None:
        """One skin region's mean colour."""
        self.add_patches(t, np.array([[r, g, b]], dtype=float))

    def add_patches(self, t: float, rgb: np.ndarray) -> None:
        """Mean colour of each skin patch, shape (patches, 3), measured on one frame."""
        if self._t:
            if t <= self._t[-1]:
                return
            if t - self._t[-1] > MAX_GAP_S or rgb.shape != self._rgb[-1].shape:
                self.reset()
        self._t.append(t)
        self._rgb.append(rgb)
        while self._t[-1] - self._t[0] > self.buffer_seconds:
            self._t.popleft()
            self._rgb.popleft()

    @property
    def buffered_seconds(self) -> float:
        return self._t[-1] - self._t[0] if len(self._t) > 1 else 0.0

    @property
    def last_sample_time(self) -> float | None:
        return self._t[-1] if self._t else None

    @property
    def effective_fps(self) -> float:
        if len(self._t) < 2:
            return 0.0
        t_end = self._t[-1]
        n = 0
        for t in reversed(self._t):
            if t_end - t > 2.0:
                break
            n += 1
        span = t_end - self._t[-n]
        return (n - 1) / span if span > 0 else 0.0

    def status(self) -> str:
        if self.buffered_seconds < self.min_seconds or self.spectrum is None:
            return "CALIBRATING"
        return ("HIGH MOTION NOISE", "WEAK SIGNAL", "LOCKED")[self._level]

    def _update_level(self, sqi: float) -> None:
        thresholds = {1: SQI_WEAK, 2: SQI_LOCKED}
        new = 2 if sqi >= SQI_LOCKED else 1 if sqi >= SQI_WEAK else 0
        if new > self._level and sqi < thresholds[new] + SQI_HYSTERESIS:
            new = self._level
        elif new < self._level and sqi > thresholds[self._level] - SQI_HYSTERESIS:
            new = self._level
        self._level = new

    # -- processing --------------------------------------------------------

    def _update_bpm(self, value: float, sqi: float, now: float) -> None:
        """Quality-gated smoothing of the displayed reading.

        Noisy estimates never move it; a large change has to persist before
        the readout follows; otherwise an EMA whose gain grows with SQI.
        """
        if sqi < SQI_WEAK:
            return
        if self.bpm is None:
            # First number: a solid lock, or a moderate one that has held steady.
            if sqi >= SQI_LOCKED or self._held_steady(now):
                self.bpm = value
            return
        if abs(value - self.bpm) > JUMP_BPM:
            if self._jump_since is None:
                self._jump_since = now
            if now - self._jump_since < JUMP_HOLD_S:
                return
        else:
            self._jump_since = None
        conf = min(sqi / 100.0, 1.0)
        self.bpm += (0.03 + 0.37 * conf ** 2) * (value - self.bpm)

    def _held_steady(self, now: float) -> bool:
        """Moderate-quality estimates that agree for STEADY_S are a real reading."""
        if len(self._steady) < STEADY_S / self.update_interval * 0.8:
            return False
        if now - self._steady[0][0] < STEADY_S - 2 * self.update_interval:
            return False
        vals = [v for _, v in self._steady]
        return max(vals) - min(vals) <= 2 * STEADY_BPM

    def _estimate(self, bvps: list[np.ndarray], now: float) -> np.ndarray:
        prefer = self.bpm / 60.0 if self.bpm is not None else None
        bvp, spec, self.patch_weights = fuse_patches(bvps, self.fs, prefer)
        self.spectrum = spec
        self.bpm_raw = spec.peak_hz * 60.0
        self.sqi = spec.sqi
        self.snr_db = spec.snr_db
        self._update_level(spec.sqi)
        if spec.sqi >= SQI_WEAK:
            self._raw_hist.append((now, self.bpm_raw))
            self._steady.append((now, self.bpm_raw))
        else:
            self._steady.clear()
        while self._raw_hist and now - self._raw_hist[0][0] > MEDIAN_WINDOW_S:
            self._raw_hist.popleft()
        while self._steady and now - self._steady[0][0] > STEADY_S:
            self._steady.popleft()
        if self._raw_hist:
            self._update_bpm(float(np.median([v for _, v in self._raw_hist])), spec.sqi, now)
        self.rmssd = rmssd_ms(bvp, self.fs, self.bpm_raw) if spec.sqi > SQI_LOCKED else None
        return bvp

    def update(self, now: float | None = None) -> bool:
        """Run the spectral estimate if one is due. Returns True if it ran."""
        if len(self._t) < 2 or self.buffered_seconds < 2.0:
            return False
        now = self._t[-1] if now is None else now
        if now - self._last_update < self.update_interval:
            return False
        self._last_update = now
        grid, rgb = resample_uniform(self._t, np.stack(self._rgb), self.fs)     # (n, patches, 3)
        bvps = [extract_bvp(rgb[:, j], self.algorithm, self.fs) for j in range(rgb.shape[1])]
        if self.buffered_seconds >= self.min_seconds:
            bvp = self._estimate(bvps, now)
        else:
            bvp = sum(b / (np.std(b) + 1e-12) for b in bvps) / len(bvps)
        self._display = self._display_fields(grid, rgb.mean(axis=1), bvp)
        return True

    def telemetry(self, waveforms: bool = True) -> dict:
        tele = {
            "algorithm": self.algorithm,
            "buffer_s": round(self.buffered_seconds, 2),
            "min_buffer_s": self.min_seconds,
            "fps": round(self.effective_fps, 1),
            "fs": self.fs,
            "status": self.status(),
            "bpm": None if self.bpm is None else round(float(self.bpm), 1),
            "bpm_raw": None if self.bpm_raw is None else round(float(self.bpm_raw), 1),
            "sqi": round(self.sqi, 1),
            "snr_db": None if self.snr_db is None else round(self.snr_db, 2),
            "rmssd": None if self.rmssd is None else round(self.rmssd, 1),
            "patch_weights": [round(w, 2) for w in self.patch_weights],
        }
        if waveforms and self.buffered_seconds >= 2.0:
            tele.update(self._display)
        return tele

    def process(self, now: float | None = None) -> dict:
        """update() + telemetry(): add a sample, then call this."""
        self.update(now)
        return self.telemetry()

    def _display_fields(self, grid: np.ndarray, rgb: np.ndarray, bvp: np.ndarray) -> dict:
        n = min(len(grid), int(self.display_seconds * self.fs))
        step = max(1, round(self.fs / 30.0))           # charts never need more than 30 points/s
        t_rel = (grid[-n:] - grid[-1])[::step]
        seg = bvp[-n:][::step]
        seg = seg / (np.std(seg) + 1e-12)
        peaks, troughs = find_beats(seg, self.fs / step, self.bpm_raw)

        rgb_seg = rgb[-n:]
        rgb_pct = (signal.detrend(rgb_seg / rgb_seg.mean(axis=0) - 1.0, axis=0) * 100.0)[::step]

        out = {
            "bvp": {"t": _r(t_rel, 3), "y": _r(seg, 3),
                    "peaks": peaks.tolist(), "troughs": troughs.tolist()},
            "rgb": {"t": _r(t_rel, 3), "r": _r(rgb_pct[:, 0]),
                    "g": _r(rgb_pct[:, 1]), "b": _r(rgb_pct[:, 2])},
        }
        if self.spectrum is not None:
            f, p = self.spectrum.freqs, self.spectrum.power
            band = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])
            pb = p[band]
            out["psd"] = {"bpm": _r(f[band] * 60.0, 2), "power": _r(pb / (pb.max() + 1e-20)),
                          "peak_bpm": round(self.spectrum.peak_hz * 60.0, 1)}
        return out
