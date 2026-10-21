"""FastAPI server: serves the dashboard and streams frames + telemetry over /ws.

Run:  uvicorn app:app --reload          then open http://localhost:8000

Environment:
    RPPG_SOURCE   camera (default) | synthetic
    RPPG_CAMERA   camera name or id to start with (default: the built-in camera)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rppg_pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rppg.app")

STATIC_DIR = Path(__file__).resolve().parent / "static"
SEND_INTERVAL = 1.0 / 40.0       # poll faster than the 30 FPS source; only new frames are sent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Built on the event loop's (main) thread so a first-run macOS camera
    # permission prompt can appear.
    pipeline = Pipeline(source=os.environ.get("RPPG_SOURCE", "camera").lower(),
                        camera_id=os.environ.get("RPPG_CAMERA") or None)
    pipeline.start()
    app.state.pipeline = pipeline
    log.info("dashboard ready at http://localhost:8000 (source: %s)", pipeline.source)
    retry = asyncio.create_task(retry_camera(pipeline))
    try:
        yield
    finally:
        retry.cancel()
        pipeline.stop()


async def retry_camera(pipeline: Pipeline) -> None:
    """On first run macOS answers 'not determined' while its permission prompt
    is up. Keep trying (on this main thread) so the camera starts once allowed."""
    while True:
        await asyncio.sleep(2.0)
        if pipeline.wants_camera and pipeline.source != "camera":
            pipeline.set_source("camera")


app = FastAPI(title="rPPG Heart Rate Monitor", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/telemetry")
async def telemetry() -> JSONResponse:
    """Latest telemetry snapshot (handy for debugging without the UI)."""
    _, _, tele = app.state.pipeline.latest()
    return JSONResponse(tele)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    """Server -> client: binary JPEG frame, then a JSON telemetry text message.
    Client -> server: JSON commands, e.g. {"type": "algorithm", "value": "CHROM"}."""
    await websocket.accept()
    pipeline: Pipeline = websocket.app.state.pipeline

    async def receive_commands() -> None:
        while True:
            msg = await websocket.receive_json()
            try:
                pipeline.handle_command(msg)
            except (KeyError, ValueError) as exc:
                log.warning("bad command %r: %s", msg, exc)

    receiver = asyncio.create_task(receive_commands())
    last_seq = -1
    try:
        while not receiver.done():
            seq, jpeg, tele = pipeline.latest()
            if seq != last_seq:
                last_seq = seq
                if jpeg is not None:
                    await websocket.send_bytes(jpeg)
                await websocket.send_text(json.dumps(tele))
            await asyncio.sleep(SEND_INTERVAL)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        receiver.cancel()
