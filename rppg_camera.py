"""Camera capture.

macOS: AVFoundation through PyObjC. For each camera it picks the format with
the highest frame rate, then the highest resolution, up to 1920x1080, and pins
the frame rate (cameras otherwise slow down in dim light, which changes the
sampling rate mid-measurement). Every frame carries the camera's own
presentation timestamp, so sample timing does not depend on Python scheduling.

Elsewhere, or without PyObjC: OpenCV VideoCapture asked for 1920x1080 at 60 fps.

Frames arrive on a capture thread and are handed over through a FrameHub.
macOS frames are BGRA, OpenCV frames BGR; downstream code handles both.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass

import cv2
import numpy as np

log = logging.getLogger("rppg.camera")

MAX_W, MAX_H = 1920, 1080

try:
    if sys.platform != "darwin":
        raise ImportError("not macOS")
    import AVFoundation as AV
    import CoreMedia as CM
    import dispatch
    import objc
    import Quartz
    from Foundation import NSObject
    HAVE_AVF = True
except ImportError:
    HAVE_AVF = False


class CameraPermissionPending(RuntimeError):
    """macOS is showing (or about to show) its camera permission prompt."""


@dataclass
class CameraMode:
    id: str
    name: str
    width: int
    height: int
    fps: float

    def as_dict(self) -> dict:
        return asdict(self)


class FrameHub:
    """Hands frames from the capture thread to the measurement and tracking threads."""

    def __init__(self, maxlen: int = 8):
        self._cond = threading.Condition()
        self._frames: deque = deque(maxlen=maxlen)
        self._latest = None
        self._latest_id = 0
        self._times: deque = deque(maxlen=121)
        self.dropped = 0

    @property
    def rate(self) -> float:
        """Frames per second actually arriving from the camera (by timestamp)."""
        with self._cond:
            if len(self._times) < 2 or self._times[-1] <= self._times[0]:
                return 0.0
            return (len(self._times) - 1) / (self._times[-1] - self._times[0])

    def push(self, t: float, frame: np.ndarray) -> None:
        with self._cond:
            self._times.append(t)
            if len(self._frames) == self._frames.maxlen:
                self.dropped += 1
            self._frames.append((t, frame))
            self._latest_id += 1
            self._latest = (self._latest_id, t, frame)
            self._cond.notify_all()

    def drain(self, timeout: float) -> list:
        """Every frame since the last call (for per-frame measurement)."""
        with self._cond:
            if not self._frames:
                self._cond.wait(timeout)
            out = list(self._frames)
            self._frames.clear()
            return out

    def wait_latest(self, after_id: int, timeout: float):
        """The newest frame, once there is one newer than `after_id` (for tracking)."""
        with self._cond:
            if self._latest_id <= after_id:
                self._cond.wait(timeout)
            return self._latest if self._latest and self._latest[0] > after_id else None

    def clear(self) -> None:
        with self._cond:
            self._frames.clear()
            self._times.clear()
            self._latest = None


# --------------------------------------------------------------------------
# macOS: AVFoundation
# --------------------------------------------------------------------------

def _avf_devices() -> list:
    types = [AV.AVCaptureDeviceTypeBuiltInWideAngleCamera]
    for name in ("AVCaptureDeviceTypeContinuityCamera", "AVCaptureDeviceTypeExternal",
                 "AVCaptureDeviceTypeExternalUnknown"):
        if hasattr(AV, name):
            types.append(getattr(AV, name))
    disc = AV.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        types, AV.AVMediaTypeVideo, AV.AVCaptureDevicePositionUnspecified)
    devices = list(disc.devices())
    builtin = [d for d in devices if d.deviceType() == AV.AVCaptureDeviceTypeBuiltInWideAngleCamera]
    return builtin + [d for d in devices if d not in builtin]


def _best_format(device):
    """(format, fps, width, height): highest fps, then most pixels, within 1920x1080."""
    best = None
    for fmt in device.formats():
        dims = CM.CMVideoFormatDescriptionGetDimensions(fmt.formatDescription())
        if dims.width > MAX_W or dims.height > MAX_H:
            continue
        fps = max((r.maxFrameRate() for r in fmt.videoSupportedFrameRateRanges()), default=0.0)
        key = (round(fps), dims.width * dims.height)
        if best is None or key > best[0]:
            best = (key, fmt, float(fps), int(dims.width), int(dims.height))
    if best is None:
        raise RuntimeError(f"{device.localizedName()} has no format up to {MAX_W}x{MAX_H}")
    return best[1:]


def list_cameras() -> list[CameraMode]:
    if not HAVE_AVF:
        return [CameraMode("0", "Camera", MAX_W, MAX_H, 60.0)]
    out = []
    for d in _avf_devices():
        try:
            _, fps, w, h = _best_format(d)
            out.append(CameraMode(str(d.uniqueID()), str(d.localizedName()), w, h, fps))
        except RuntimeError:
            continue
    return out


if HAVE_AVF:
    class _FrameDelegate(NSObject, protocols=[objc.protocolNamed("AVCaptureVideoDataOutputSampleBufferDelegate")]):
        def initWithHub_(self, hub):
            self = objc.super(_FrameDelegate, self).init()
            if self is None:
                return None
            self.hub = hub
            return self

        def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sample, connection):
            pts = CM.CMSampleBufferGetPresentationTimeStamp(sample)
            if not pts.timescale:
                return
            pb = CM.CMSampleBufferGetImageBuffer(sample)
            Quartz.CVPixelBufferLockBaseAddress(pb, Quartz.kCVPixelBufferLock_ReadOnly)
            try:
                w = Quartz.CVPixelBufferGetWidth(pb)
                h = Quartz.CVPixelBufferGetHeight(pb)
                bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
                raw = Quartz.CVPixelBufferGetBaseAddress(pb).as_buffer(bpr * h)
                frame = np.frombuffer(raw, np.uint8).reshape(h, bpr)[:, : w * 4].reshape(h, w, 4).copy()
            finally:
                Quartz.CVPixelBufferUnlockBaseAddress(pb, Quartz.kCVPixelBufferLock_ReadOnly)
            self.hub.push(pts.value / pts.timescale, frame)


class MacCamera:
    def __init__(self, hub: FrameHub, camera_id: str | None = None):
        status = AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeVideo)
        if status == AV.AVAuthorizationStatusNotDetermined:
            AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AV.AVMediaTypeVideo, lambda ok: None)
            raise CameraPermissionPending("waiting for camera permission")
        if status != AV.AVAuthorizationStatusAuthorized:
            raise PermissionError("camera access is turned off for this app. Turn it on in System Settings "
                                  "> Privacy & Security > Camera, then restart the server")
        devices = _avf_devices()
        if not devices:
            raise RuntimeError("no camera found")
        wanted = (camera_id or "").lower()
        device = next((d for d in devices if wanted and (wanted == str(d.uniqueID()).lower()
                                                         or wanted in str(d.localizedName()).lower())),
                      devices[0])
        fmt, fps, w, h = _best_format(device)

        self.session = AV.AVCaptureSession.alloc().init()
        inp, err = AV.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
        if inp is None:
            raise RuntimeError(f"could not open {device.localizedName()}: {err}")
        out = AV.AVCaptureVideoDataOutput.alloc().init()
        out.setVideoSettings_({Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA})
        out.setAlwaysDiscardsLateVideoFrames_(True)
        self._delegate = _FrameDelegate.alloc().initWithHub_(hub)
        self._queue = dispatch.dispatch_queue_create(b"rppg.camera", None)
        out.setSampleBufferDelegate_queue_(self._delegate, self._queue)

        self.session.beginConfiguration()
        self.session.addInput_(inp)
        self.session.addOutput_(out)
        self.session.commitConfiguration()

        # Some cameras (Continuity Camera among them) fall back to their default
        # format when the session starts, so apply the choice before and after.
        self._configure(device, fmt, fps)
        conn = out.connectionWithMediaType_(AV.AVMediaTypeVideo)
        if conn is not None and conn.isVideoMinFrameDurationSupported():
            conn.setVideoMinFrameDuration_(CM.CMTimeMake(1, int(round(fps))))
        self.session.startRunning()
        self._configure(device, fmt, fps)
        self.device = device
        self.effects = self._active_effects(device)
        self.mode = CameraMode(str(device.uniqueID()), str(device.localizedName()), w, h, fps)
        active = device.activeVideoMinFrameDuration()
        dims = CM.CMVideoFormatDescriptionGetDimensions(device.activeFormat().formatDescription())
        log.info("camera running: %s (active %dx%d, min frame duration %d/%d s, effects on: %s)", self.mode,
                 dims.width, dims.height, active.value, active.timescale, self.effects or "none")

    @staticmethod
    def _active_effects(device) -> list[str]:
        """macOS video effects that are on. They repaint skin tone and brightness
        (Portrait, Studio Light) or crop and pan the picture (Center Stage), and
        while any is on many cameras are capped at 30 fps."""
        names = {"isCenterStageActive": "Center Stage", "isPortraitEffectActive": "Portrait",
                 "isStudioLightActive": "Studio Light", "isBackgroundReplacementActive": "Background"}
        out = []
        for attr, label in names.items():
            fn = getattr(device, attr, None)
            try:
                if fn is not None and fn():
                    out.append(label)
            except Exception:  # noqa: BLE001 - property missing on older macOS
                pass
        return out

    @staticmethod
    def _configure(device, fmt, fps: float) -> None:
        ok, err = device.lockForConfiguration_(None)
        if not ok:
            log.warning("could not configure %s: %s", device.localizedName(), err)
            return
        try:
            device.setActiveFormat_(fmt)
            period = CM.CMTimeMake(1, int(round(fps)))
            device.setActiveVideoMinFrameDuration_(period)    # pin the rate so it
            device.setActiveVideoMaxFrameDuration_(period)    # cannot sag in dim light
        finally:
            device.unlockForConfiguration()

    def release(self) -> None:
        self.session.stopRunning()


# --------------------------------------------------------------------------
# Fallback: OpenCV
# --------------------------------------------------------------------------

class OpenCVCamera:
    def __init__(self, hub: FrameHub, camera_id: str | None = None):
        index = int(camera_id) if camera_id and camera_id.isdigit() else 0
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, MAX_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MAX_H)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)       # V4L2: manual exposure
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.mode = CameraMode(str(index), f"Camera {index}",
                               int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                               float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0))
        self._hub = hub
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read, name="rppg-capture", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok:
                self._hub.push(time.perf_counter(), frame)
            else:
                time.sleep(0.01)

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


def open_camera(hub: FrameHub, camera_id: str | None = None):
    return MacCamera(hub, camera_id) if HAVE_AVF else OpenCVCamera(hub, camera_id)
