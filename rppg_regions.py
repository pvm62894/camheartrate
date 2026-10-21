"""Choosing which patch of skin to read.

Candidates each frame:
  face  forehead + both cheeks (face mesh). Thin skin, dense capillaries,
        little muscle movement.
  palm  centre of the palm (hand landmarks). Very well perfused.
  skin  any other skin-coloured blob, e.g. a forearm.

Automatic mode prefers a palm held up to the camera, then the face, then the
largest other skin blob. Tapping the video locks onto whatever was tapped.
Changing region restarts the measurement, because every patch of skin has its
own colour and pulse amplitude and the two traces cannot be spliced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

FOREHEAD = (10, 338, 297, 332, 299, 337, 151, 108, 69, 103, 67, 109)
LEFT_MALAR = (117, 118, 101, 36, 205, 187, 123)
RIGHT_MALAR = (346, 347, 330, 266, 425, 411, 352)
FACE_ROIS = (FOREHEAD, LEFT_MALAR, RIGHT_MALAR)
PALM = (0, 1, 5, 9, 13, 17)            # wrist, thumb base, the four knuckles

PRIORITY = ("palm", "face", "skin")
SWITCH_FRAMES = 30          # ~1 s an automatic switch must persist (tracking runs at 30 Hz)
HOLD_S = 0.6                # ride through short detection dropouts
LOCK_TIMEOUT_S = 2.0        # a tapped region that disappears this long is released
SKIN_MATCH = 0.12           # max blob-centre jump, as a fraction of the frame diagonal
MIN_BLOB_FRACTION = 0.012   # smallest skin blob worth reading, as a fraction of the frame
MAX_BLOB_FRACTION = 0.40    # bigger "skin" is a wall or a colour cast, not an arm


@dataclass
class Region:
    kind: str                       # "face" | "palm" | "skin"
    outline: list[np.ndarray]       # int32 contours in frame pixels; what the UI draws
    hit: np.ndarray                 # contour used for tap hit-testing
    centre: tuple[float, float]
    frame_w: int = 1280

    def mask(self, shape, scale: float = 1.0, offset=(0, 0)) -> np.ndarray:
        """Pixels to average, for a frame of `shape` that is `scale` times smaller
        than the frame the outline was found in. Skin blobs lose a border so edge
        pixels, which flicker between skin and background as you move, are left out."""
        m = np.zeros(shape[:2], np.uint8)
        polys = [((p.reshape(-1, 2) - offset) / scale).astype(np.int32) for p in self.outline]
        if self.kind == "skin":
            cv2.drawContours(m, polys, -1, 255, cv2.FILLED)
            k = max(3, int(0.012 * self.frame_w / scale) | 1)
            m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        else:
            for poly in polys:
                cv2.fillConvexPoly(m, poly, 255)
        return m

    def crop_masks(self, shape) -> tuple[int, int, int, int, list[np.ndarray]]:
        """(x0, y0, x1, y1, masks) covering just the region's bounding box, so
        averaging a 1080p frame only touches the pixels that matter. One mask
        per patch (forehead, each cheek), so each patch is measured separately."""
        x, y, w, h = cv2.boundingRect(np.concatenate([p.reshape(-1, 2) for p in self.outline]))
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(shape[1], x + w + 1), min(shape[0], y + h + 1)
        size = (y1 - y0, x1 - x0)
        if self.kind == "skin":
            return x0, y0, x1, y1, [self.mask(size, offset=(x0, y0))]
        masks = []
        for poly in self.outline:
            m = np.zeros(size, np.uint8)
            cv2.fillConvexPoly(m, (poly.reshape(-1, 2) - (x0, y0)).astype(np.int32), 255)
            masks.append(m)
        return x0, y0, x1, y1, masks

    def contains(self, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(self.hit, (float(x), float(y)), False) >= 0


def face_region(landmarks: np.ndarray, frame_w: int) -> Region:
    outline = [cv2.convexHull(landmarks[list(idx)].astype(np.int32)) for idx in FACE_ROIS]
    mesh = landmarks[:468]
    return Region("face", outline, cv2.convexHull(mesh.astype(np.int32)), tuple(mesh.mean(axis=0)), frame_w)


def palm_region(landmarks: np.ndarray, frame_w: int) -> Region:
    pts = landmarks[list(PALM)]
    c = pts.mean(axis=0)
    shrunk = c + 0.85 * (pts - c)                     # keep clear of the palm's edges
    return Region("palm", [cv2.convexHull(shrunk.astype(np.int32))],
                  cv2.convexHull(landmarks.astype(np.int32)), tuple(c), frame_w)


def skin_regions(mask_small: np.ndarray, scale: float, exclude: list[np.ndarray]) -> list[Region]:
    """Skin blobs (largest first), excluding face and hand, in full-frame pixels."""
    mask = mask_small.copy()
    for hull in exclude:                               # grow 15 % so the neck/edges go too
        c = hull.reshape(-1, 2).mean(axis=0)
        grown = c + 1.15 * (hull.reshape(-1, 2) - c)
        cv2.fillConvexPoly(mask, (grown / scale).astype(np.int32), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = mask.shape[0] * mask.shape[1]
    out = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area > MAX_BLOB_FRACTION * frame_area:
            continue
        if area < MIN_BLOB_FRACTION * frame_area:
            break
        full = (cv2.approxPolyDP(cnt, 1.5, True).astype(np.float32) * scale).astype(np.int32)
        m = cv2.moments(full)
        if m["m00"] <= 0:
            continue
        out.append(Region("skin", [full], full, (m["m10"] / m["m00"], m["m01"] / m["m00"]),
                          round(mask.shape[1] * scale)))
    return out


class RegionSelector:
    """Picks one region per frame, with hysteresis so detection flicker does
    not keep restarting the measurement."""

    def __init__(self):
        self.current: Region | None = None
        self.locked = False
        self.notice: str | None = None
        self._last_seen = -math.inf
        self._pending: str | None = None
        self._pending_n = 0
        self._tap: tuple[float, float] | None = None

    # -- requests from the UI --------------------------------------------------

    def tap(self, x: float, y: float) -> None:
        """Select the skin under a point given in 0-1 frame coordinates."""
        self._tap = (min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0))

    def release(self) -> None:
        if self.locked:
            self.locked = False
            self.notice = "Choosing skin automatically"

    def needs_skin_blobs(self, has_face: bool, has_palm: bool) -> bool:
        if self._tap is not None or (self.current is not None and self.current.kind == "skin"):
            return True
        return not self.locked and not has_face and not has_palm

    # -- per frame -------------------------------------------------------------

    def update(self, cands: dict[str, list[Region]], t: float, size: tuple[int, int]):
        """Returns (region or None, changed). None means: skip this frame."""
        diag = math.hypot(*size)
        changed = False
        if self._tap is not None:
            changed = self._resolve_tap(cands, t, size)
            self._tap = None

        if self.locked:
            match = self._match(cands, self.current, diag)
            if match is not None:
                self.current, self._last_seen = match, t
                return match, changed
            if t - self._last_seen <= LOCK_TIMEOUT_S:
                return None, changed
            self.locked = False
            self.notice = "Lost the skin you picked. Choosing automatically"

        best = None
        for kind in PRIORITY:
            if cands.get(kind):
                if self.current is not None and self.current.kind == kind:
                    best = self._match(cands, self.current, diag)
                best = best or cands[kind][0]
                break

        if best is None:
            if self.current is not None and t - self._last_seen > HOLD_S:
                self.current = None
                changed = True
            return None, changed

        if self.current is None or self._same(best, self.current, diag):
            changed = changed or self.current is None
            self.current, self._last_seen = best, t
            self._pending, self._pending_n = None, 0
            return best, changed

        # A different region wins this frame: switch only if it persists.
        if self._pending == best.kind:
            self._pending_n += 1
        else:
            self._pending, self._pending_n = best.kind, 1
        still_there = self._match(cands, self.current, diag)
        if self._pending_n >= SWITCH_FRAMES or (still_there is None and t - self._last_seen > HOLD_S):
            self.current, self._last_seen = best, t
            self._pending, self._pending_n = None, 0
            return best, True
        if still_there is not None:
            self.current, self._last_seen = still_there, t
            return still_there, changed
        return None, changed

    # -- helpers -----------------------------------------------------------------

    @staticmethod
    def _same(a: Region, b: Region, diag: float) -> bool:
        if a.kind != b.kind:
            return False
        if a.kind != "skin":
            return True
        return math.dist(a.centre, b.centre) <= SKIN_MATCH * diag

    def _match(self, cands: dict[str, list[Region]], ref: Region, diag: float) -> Region | None:
        """The candidate that continues `ref` (same face/palm, or the skin blob it moved to)."""
        options = cands.get(ref.kind, [])
        if not options:
            return None
        if ref.kind != "skin":
            return options[0]
        for r in options:
            if r.contains(*ref.centre):
                return r
        near = min(options, key=lambda r: math.dist(r.centre, ref.centre))
        return near if math.dist(near.centre, ref.centre) <= SKIN_MATCH * diag else None

    def _resolve_tap(self, cands, t: float, size) -> bool:
        x, y = self._tap[0] * size[0], self._tap[1] * size[1]
        for kind in ("palm", "face", "skin"):
            for r in cands.get(kind, []):
                if r.contains(x, y):
                    changed = self.current is None or not self._same(r, self.current, math.hypot(*size))
                    self.current, self._last_seen, self.locked = r, t, True
                    self.notice = {"palm": "Reading your palm", "face": "Reading your face",
                                   "skin": "Reading the skin you tapped"}[kind]
                    return changed
        self.locked = False
        self.notice = "No skin found there. Choosing automatically"
        return False
