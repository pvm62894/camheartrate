"""Camera -> skin region -> DSP, on three threads.

  capture   (rppg_camera)  frames + hardware timestamps into a FrameHub
  measure   every frame: average the current skin mask, feed the DSP engine.
            Runs at the full camera rate (60 fps on cameras that can).
  track     newest frame only, ~30 Hz: face/hand landmarks, skin blobs, region
            choice, the next mask for `measure`, and the JPEG + telemetry the
            browser receives.

Frames are processed unmirrored; the picture and outline are mirrored for
display so the page behaves like a selfie camera.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from rppg_camera import CameraPermissionPending, FrameHub, list_cameras, open_camera
from rppg_dsp import RPPGEngine
from rppg_regions import Region, RegionSelector, face_region, palm_region, skin_regions
from rppg_synthetic import SyntheticRenderer, SyntheticSkinTrace, roi_pixel_count
from rppg_vision import (DETECT_WIDTH, FaceMeshTracker, HandTracker, LandmarkSmoother,
                         SkinSegmenter, to_mp_image)

log = logging.getLogger("rppg.pipeline")

SYNTH_FPS = 30.0
TRACK_HZ = 30.0
DISPLAY_WIDTH = 1280
JPEG_QUALITY = 80
MIN_PATCH_PIXELS = 150
CLIP_LEVEL = 250            # a channel at or above this is saturated: no pulse left in it
LEARN_EVERY = 15            # tracking cycles between skin-colour model updates


@dataclass
class RoiMask:
    gen: int                # region generation; measurement only counts the current one
    x0: int
    y0: int
    x1: int
    y1: int
    masks: list[np.ndarray]     # one per skin patch
    pixels: int


class Pipeline:
    def __init__(self, source: str = "camera", camera_id: str | None = None):
        self.hub = FrameHub()
        self.engine = RPPGEngine()
        self.synth = SyntheticSkinTrace(bpm=72.0, motion=0.3)
        self.renderer = SyntheticRenderer()
        self.synth_pixels = roi_pixel_count()
        self.selector = RegionSelector()
        self.segmenter = SkinSegmenter()
        self.camera_id = camera_id
        self.cameras = [c.as_dict() for c in list_cameras()]
        self.source = "idle"                      # until a camera or the simulation is running
        self.wants_camera = source == "camera"    # keep retrying until permission is granted
        self.message = ""

        self._camera = None
        self._face: FaceMeshTracker | None = None
        self._hand: HandTracker | None = None
        self._face_smooth = LandmarkSmoother()
        self._hand_smooth = LandmarkSmoother()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._rng = np.random.default_rng()
        self._next_synth_t: float | None = None
        self._gen = 0
        self._roi: RoiMask | None = None
        self._measured = (0, 0.0)               # (pixels, clipped fraction) of the last frame
        self._cycle = 0
        self._motion = 0.0                      # region speed, frame widths per second (smoothed)
        self._last_centre: tuple[float, float, float] | None = None
        self._notice = {"id": 0, "text": ""}

        self._seq = 0
        self._jpeg: bytes | None = None
        self._telemetry: dict = {}

        self.set_source("camera" if source == "camera" else "synthetic")

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        for name, target in (("rppg-measure", self._measure_loop), ("rppg-track", self._track_loop)):
            th = threading.Thread(target=target, name=name, daemon=True)
            th.start()
            self._threads.append(th)

    def stop(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=3.0)
        self._close_camera()
        for tracker in (self._face, self._hand):
            if tracker is not None:
                tracker.close()

    def latest(self) -> tuple[int, bytes | None, dict]:
        with self._lock:
            return self._seq, self._jpeg, self._telemetry

    # -- commands from the UI --------------------------------------------------

    def _close_camera(self) -> None:
        if self._camera is not None:
            self._camera.release()
            self._camera = None

    def set_source(self, source: str, camera_id: str | None = None) -> None:
        """Switch input. Cameras are opened on the calling thread, which is the
        server's main thread, so macOS can show its permission prompt."""
        if source == "camera":
            self.wants_camera = True
            if camera_id is not None and camera_id != self.camera_id:
                with self._lock:
                    self.source = "idle"
                self._close_camera()
                self.camera_id = camera_id
            if self._camera is None:
                try:
                    self.hub.clear()
                    self._camera = open_camera(self.hub, self.camera_id)
                except CameraPermissionPending:
                    self.message = ("Waiting for the camera. Allow camera access when your computer asks, "
                                    "and it will start here on its own.")
                    return
                except Exception as exc:  # noqa: BLE001 - surface any driver error in the UI
                    log.warning("camera unavailable: %s", exc)
                    self.message = f"Camera unavailable: {exc}."
                    return
            mode = self._camera.mode
            self.camera_id = mode.id
            with self._lock:
                self.source = "camera"
                self.engine = RPPGEngine(algorithm=self.engine.algorithm, fs=mode.fps)
                self.selector = RegionSelector()
                self._face_smooth, self._hand_smooth = LandmarkSmoother(), LandmarkSmoother()
                self._roi = None
                self._gen += 1
                self.message = ""
        elif source == "synthetic":
            self.wants_camera = False
            with self._lock:
                self.source = "synthetic"
                self.engine = RPPGEngine(algorithm=self.engine.algorithm, fs=SYNTH_FPS)
                self._next_synth_t = None
                self.message = ""
            self._close_camera()
        else:
            raise ValueError(f"unknown source {source!r}")

    def handle_command(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "algorithm":
            with self._lock:
                self.engine.set_algorithm(str(msg["value"]))
        elif kind == "source":
            self.set_source(str(msg["value"]))
        elif kind == "camera":
            self.set_source("camera", str(msg["value"]))
        elif kind == "synthetic":
            with self._lock:
                self.synth.set_params(bpm=msg.get("bpm"), motion=msg.get("motion"))
        elif kind == "select":
            with self._lock:                           # the page is mirrored
                self.selector.tap(1.0 - float(msg["x"]), float(msg["y"]))
        elif kind == "auto":
            with self._lock:
                self.selector.release()
        elif kind == "reset":
            with self._lock:
                self.engine.reset()
        else:
            log.debug("ignoring command %r", msg)

    # -- measure thread ----------------------------------------------------------

    def _measure_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.source == "camera":
                    self._measure_frames()
                elif self.source == "synthetic":
                    self._step_synthetic()
                else:
                    self._publish_idle()
                    time.sleep(0.25)
            except Exception as exc:  # noqa: BLE001 - keep the loop alive, report in UI
                log.exception("measurement failed")
                self.message = f"Processing error: {exc}"
                time.sleep(0.2)

    def _measure_frames(self) -> None:
        for t, frame in self.hub.drain(timeout=0.2):
            roi = self._roi
            if roi is None or roi.y1 > frame.shape[0] or roi.x1 > frame.shape[1]:
                continue
            crop = frame[roi.y0:roi.y1, roi.x0:roi.x1]
            ch = crop.shape[2]
            unclipped = cv2.inRange(crop, (0,) * ch, (CLIP_LEVEL - 1,) * 3 + (255,) * (ch - 3))
            means, total = [], 0
            for m in roi.masks:
                valid = cv2.bitwise_and(m, unclipped)
                n = cv2.countNonZero(valid)
                if n < MIN_PATCH_PIXELS:
                    break
                b, g, r = cv2.mean(crop, mask=valid)[:3]
                means.append((r, g, b))
                total += n
            self._measured = (total, 1.0 - total / max(roi.pixels, 1))
            if len(means) != len(roi.masks):
                continue
            with self._lock:
                if roi.gen == self._gen and self.source == "camera":
                    self.engine.add_patches(t, np.array(means))
                    self.engine.update(t)

    # -- track thread ------------------------------------------------------------

    def _track_loop(self) -> None:
        last_id = 0
        next_due = 0.0
        while not self._stop.is_set():
            if self.source != "camera":
                time.sleep(0.05)
                continue
            item = self.hub.wait_latest(last_id, timeout=0.5)
            if item is None:
                continue
            last_id, t, frame = item
            delay = next_due - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            next_due = time.perf_counter() + 1.0 / TRACK_HZ
            try:
                self._track(t, frame)
            except Exception as exc:  # noqa: BLE001
                log.exception("tracking failed")
                self.message = f"Processing error: {exc}"
                time.sleep(0.2)

    def _load_trackers(self) -> bool:
        if self._face is not None:
            return True
        try:
            self._face = FaceMeshTracker()
            self._hand = HandTracker()
            return True
        except Exception as exc:  # noqa: BLE001 - e.g. model download blocked
            log.exception("landmark models failed to load")
            self.message = f"Could not load the tracking models: {exc}."
            return False

    def _track(self, t: float, frame: np.ndarray) -> None:
        if not self._load_trackers():
            time.sleep(1.0)
            return
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        scale = w / DETECT_WIDTH
        small = cv2.resize(frame, (DETECT_WIDTH, round(h / scale)), interpolation=cv2.INTER_AREA)
        if small.shape[2] == 4:
            small = cv2.cvtColor(small, cv2.COLOR_BGRA2BGR)
        image = to_mp_image(small)
        face_lm = self._face_smooth(self._face.detect(image, t, w, h), t)
        hand_lm = self._hand_smooth(self._hand.detect(image, t, w, h), t)

        cands: dict[str, list[Region]] = {}
        if face_lm is not None:
            cands["face"] = [face_region(face_lm, w)]
        if hand_lm is not None:
            cands["palm"] = [palm_region(hand_lm, w)]
        with self._lock:
            want_blobs = self.selector.needs_skin_blobs("face" in cands, "palm" in cands)
        if want_blobs:
            exclude = [c[0].hit for c in cands.values()]
            cands["skin"] = skin_regions(self.segmenter.segment(small), scale, exclude)

        self._cycle += 1
        known = cands.get("face") or cands.get("palm")
        if known and self._cycle % LEARN_EVERY == 0:
            self.segmenter.learn(small, known[0].mask(small.shape, scale))

        with self._lock:
            if self.source != "camera":
                return
            region, changed = self.selector.update(cands, t, (w, h))
            if changed:
                self._gen += 1
                self.engine.reset()
            gen = self._gen
            shown = region or self.selector.current
            locked = self.selector.locked
            self._take_notice()
        if region is not None:
            x0, y0, x1, y1, masks = region.crop_masks(frame.shape)
            self._roi = RoiMask(gen, x0, y0, x1, y1, masks, sum(cv2.countNonZero(m) for m in masks))
        elif shown is None:
            self._roi = None

        with self._lock:
            tele = self.engine.telemetry()
        pixels, clipped = self._measured if shown is not None else (0, 0.0)
        tele["region"] = None if shown is None else {
            "kind": shown.kind, "locked": locked,
            "outline": [np.round(np.column_stack([1.0 - p.reshape(-1, 2)[:, 0] / w,
                                                  p.reshape(-1, 2)[:, 1] / h]), 4).tolist()
                        for p in shown.outline]}
        if shown is not None and self._last_centre is not None and t > self._last_centre[2]:
            speed = math.dist(shown.centre, self._last_centre[:2]) / (t - self._last_centre[2]) / w
            self._motion += 0.3 * (speed - self._motion)
        self._last_centre = (*shown.centre, t) if shown is not None else None
        tele["clipped"] = round(clipped, 3)
        tele["moving"] = self._motion > 0.05
        if shown is None:
            tele["status"] = "NO SKIN"

        disp_h = round(h * DISPLAY_WIDTH / w)
        disp = cv2.resize(frame, (DISPLAY_WIDTH, disp_h), interpolation=cv2.INTER_AREA) if w > DISPLAY_WIDTH else frame
        if disp.shape[2] == 4:
            disp = cv2.cvtColor(disp, cv2.COLOR_BGRA2BGR)
        disp = cv2.flip(disp, 1)
        self._publish(disp, tele, (w, h), pixels, (time.perf_counter() - t0) * 1000.0)

    # -- synthetic -----------------------------------------------------------------

    def _step_synthetic(self) -> None:
        # Pace at ~30 FPS with webcam-like jitter and occasional dropped frames,
        # so the resampling stage sees realistic irregular timestamps.
        now = time.perf_counter()
        if self._next_synth_t is None or now - self._next_synth_t > 0.5:
            self._next_synth_t = now
        delay = self._next_synth_t - now
        if delay > 0:
            time.sleep(delay)
        t = time.perf_counter()
        step = 1.0 / SYNTH_FPS + self._rng.normal(0.0, 0.002)
        if self._rng.random() < 0.02:
            step += 1.0 / SYNTH_FPS
        self._next_synth_t += step

        t0 = time.perf_counter()
        with self._lock:
            if self.source != "synthetic":
                return
            rgb = self.synth.sample(t)
            self.engine.add_sample(t, *rgb)
            self.engine.update(t)
            tele = self.engine.telemetry()
            truth = self.synth.truth
            motion = self.synth.last_motion
            self._take_notice()
        frame, outline = self.renderer.render(rgb, motion)
        tele["synthetic"] = truth
        tele["region"] = {"kind": "simulated", "locked": False, "outline": outline}
        h, w = frame.shape[:2]
        self._publish(frame, tele, (w, h), self.synth_pixels, (time.perf_counter() - t0) * 1000.0)

    # -- output ----------------------------------------------------------------------

    def _publish_idle(self) -> None:
        """No video yet (e.g. waiting for camera permission): still tell the page why."""
        tele = {"source": "idle", "status": "WAITING", "bpm": None, "message": self.message,
                "cameras": self.cameras, "notice": self._notice, "region": None}
        with self._lock:
            self._seq += 1
            self._jpeg = None
            self._telemetry = tele

    def _take_notice(self) -> None:
        """Move a selector notice (e.g. 'Reading your palm') into telemetry. Caller holds the lock."""
        if self.selector.notice:
            self._notice = {"id": self._notice["id"] + 1, "text": self.selector.notice}
            self.selector.notice = None

    def _publish(self, frame: np.ndarray, tele: dict, size: tuple[int, int], pixels: int,
                 proc_ms: float) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        camera = self._camera.mode.as_dict() if self._camera is not None and self.source == "camera" else None
        tele.update({
            "source": self.source,
            "frame": list(size),
            "roi_pixels": int(pixels),
            "noise_reduction": round(math.sqrt(pixels), 1) if pixels else 0.0,
            "proc_ms": round(proc_ms, 1),
            "camera": camera,
            "cameras": self.cameras,
            "dropped_frames": self.hub.dropped,
            "camera_fps": round(self.hub.rate, 1),
            "notice": self._notice,
            "message": self.message,
        })
        with self._lock:
            self._seq += 1
            self._jpeg = buf.tobytes() if ok else self._jpeg
            self._telemetry = tele
