"""MediaPipe face/hand landmarks, landmark smoothing and skin segmentation.

Three detectors feed the region selector (rppg_regions.py):
  * FaceMeshTracker  468-point face mesh -> forehead and cheek ROIs
  * HandTracker      21-point hand skeleton -> palm ROI
  * SkinSegmenter    colour-based skin mask -> any other exposed skin (arm, ...)
"""

from __future__ import annotations

import logging
import math
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")        # quieten MediaPipe / TFLite logs
import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402

log = logging.getLogger("rppg.vision")

MODELS_DIR = Path(__file__).resolve().parent / "models"
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")

DETECT_WIDTH = 640               # landmark and skin detection run on a downscaled copy


# --------------------------------------------------------------------------
# Landmark trackers
# --------------------------------------------------------------------------

def ensure_model(url: str) -> Path:
    """Fetch a Google MediaPipe model into models/ on first use."""
    path = MODELS_DIR / url.rsplit("/", 1)[-1]
    if not path.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        log.info("downloading %s", url)
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def to_mp_image(frame_bgr: np.ndarray) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


class LandmarkSmoother:
    """One Euro filter (Casiez et al., 2012) over a whole landmark array.

    Landmarks jitter by a pixel or two between frames even when you hold
    still. Every jitter moves the ROI edge over different pixels and shows up
    as noise in the colour average. The filter smooths hard while still and
    lets go while moving, so the outline neither shakes nor lags.
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.015, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x = self._dx = None
        self._t = 0.0

    @staticmethod
    def _alpha(cutoff, dt: float):
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))

    def __call__(self, x: np.ndarray | None, t: float) -> np.ndarray | None:
        if x is None:
            return None
        dt = t - self._t
        if self._x is None or x.shape != self._x.shape or not 0.0 < dt < 0.5:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x
        dx = (x - self._x) / dt
        self._dx = self._dx + self._alpha(self.d_cutoff, dt) * (dx - self._dx)
        a = self._alpha(self.min_cutoff + self.beta * np.abs(self._dx), dt)
        self._x = self._x + a * (x - self._x)
        self._t = t
        return self._x


class _VideoLandmarker:
    """Shared timestamp handling for MediaPipe VIDEO-mode landmarkers."""

    def __init__(self):
        self._last_ts = -1

    def _ts(self, t: float) -> int:
        ts = max(int(t * 1000), self._last_ts + 1)          # must strictly increase
        self._last_ts = ts
        return ts

    @staticmethod
    def _to_px(points, w: int, h: int) -> np.ndarray:
        return np.array([(p.x * w, p.y * h) for p in points], dtype=np.float32)


class FaceMeshTracker(_VideoLandmarker):
    """MediaPipe FaceLandmarker: the 468-point face mesh (+10 iris points)."""

    def __init__(self):
        super().__init__()
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(FACE_MODEL_URL))),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self._lm = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image: mp.Image, t: float, w: int, h: int) -> np.ndarray | None:
        """Landmarks in pixel coordinates of a (w, h) frame, or None."""
        res = self._lm.detect_for_video(image, self._ts(t))
        return self._to_px(res.face_landmarks[0], w, h) if res.face_landmarks else None

    def close(self) -> None:
        self._lm.close()


class HandTracker(_VideoLandmarker):
    """MediaPipe HandLandmarker: 21 hand keypoints."""

    def __init__(self):
        super().__init__()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(HAND_MODEL_URL))),
            running_mode=vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self._lm = vision.HandLandmarker.create_from_options(options)

    def detect(self, image: mp.Image, t: float, w: int, h: int) -> np.ndarray | None:
        res = self._lm.detect_for_video(image, self._ts(t))
        return self._to_px(res.hand_landmarks[0], w, h) if res.hand_landmarks else None

    def close(self) -> None:
        self._lm.close()


# --------------------------------------------------------------------------
# Skin segmentation
# --------------------------------------------------------------------------

class SkinSegmenter:
    """Finds skin by colour in YCrCb space.

    Skin of every tone clusters tightly in the chroma plane (Cr, Cb) because
    melanin and haemoglobin mostly change brightness, not hue. Until it has
    seen your skin it uses the classic fixed box (Cr 135-175, Cb 85-130).
    Once a face or palm has been tracked it switches to a Gaussian fitted to
    those pixels, which rejects wood, walls and other skin-coloured things far
    better.
    """

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self._open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    @property
    def calibrated(self) -> bool:
        return self.mean is not None

    def learn(self, frame_bgr: np.ndarray, mask: np.ndarray, rate: float = 0.1) -> None:
        """Update the skin colour model from pixels known to be skin."""
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        px = ycrcb[mask > 0][:, 1:3].astype(np.float64)
        if len(px) < 200:
            return
        px = px[:: max(1, len(px) // 4000)]
        mean = px.mean(axis=0)
        cov = np.cov(px.T) + np.eye(2) * 4.0        # floor keeps the gate from collapsing
        if self.mean is None:
            self.mean, self.cov = mean, cov
        else:
            self.mean = (1 - rate) * self.mean + rate * mean
            self.cov = (1 - rate) * self.cov + rate * cov

    def segment(self, small_bgr: np.ndarray) -> np.ndarray:
        """Binary skin mask (uint8, 0/255) at the input's resolution."""
        ycrcb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2YCrCb)
        if self.mean is None:
            mask = cv2.inRange(ycrcb, (40, 135, 85), (240, 175, 130))
        else:
            d = ycrcb[..., 1:3].astype(np.float32) - self.mean.astype(np.float32)
            icov = np.linalg.inv(self.cov).astype(np.float32)
            d2 = (d[..., 0] ** 2 * icov[0, 0] + 2 * d[..., 0] * d[..., 1] * icov[0, 1]
                  + d[..., 1] ** 2 * icov[1, 1])
            y = ycrcb[..., 0]
            mask = ((d2 < 9.0) & (y > 35) & (y < 245)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close)
