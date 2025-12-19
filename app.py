import os
import uuid
import shutil
import threading
from pathlib import Path
from typing import Dict

import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# -----------------------------
# Environment safety (Render)
# -----------------------------
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

# -----------------------------
# App init
# -----------------------------
app = FastAPI(title="Kakinada Traffic AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
INPUT_DIR = OUTPUT_DIR / "inputs"
OUTPUT_VIDEO_DIR = OUTPUT_DIR / "videos"
OUTPUT_CSV_DIR = OUTPUT_DIR / "csv"

for d in [OUTPUT_DIR, INPUT_DIR, OUTPUT_VIDEO_DIR, OUTPUT_CSV_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Job Store (in-memory)
# -----------------------------
JOBS: Dict[str, Dict] = {}

def create_job() -> str:
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {
        "status": "queued",
        "progress": 0
    }
    return job_id

def update_job(job_id: str, **kwargs):
    if job_id in JOBS:
        JOBS[job_id].update(kwargs)

def get_job(job_id: str):
    return JOBS.get(job_id)

# -----------------------------
# Lazy detectors (NO import-time load)
# -----------------------------
_vehicle_tracker = None
_helmet_detector = None
_weapon_detector = None
_fight_detector = None
_event_manager = None
_dedup = None

def get_detectors():
    global _vehicle_tracker, _helmet_detector, _weapon_detector, _fight_detector, _event_manager, _dedup

    if _vehicle_tracker is None:
        from pipeline.vehicle_tracker import VehicleTracker
        from pipeline.helmet_detector import HelmetDetector
        from pipeline.weapon_detector import WeaponDetector
        from pipeline.fight_detector import FightDetector
        from pipeline.event_manager import EventManager
        from utils.dedup import EventDeduplicator

        _vehicle_tracker = VehicleTracker()
        _helmet_detector = HelmetDetector()
        _weapon_detector = WeaponDetector()
        _fight_detector = FightDetector()
        _event_manager = EventManager()
        _dedup = EventDeduplicator()

    return (
        _vehicle_tracker,
        _helmet_detector,
        _weapon_detector,
        _fight_detector,
        _event_manager,
        _dedup,
    )

# -----------------------------
# Core processing function
# -----------------------------
def process_video(job_id: str, input_path: Path):
    try:
        update_job(job_id, status="processing", progress=0)

        (
            vehicle_tracker,
            helmet_detector,
            weapon_detector,
            fight_detector,
            event_manager,
            dedup,
        ) = get_detectors()

        from utils.csv_logger import init_csv, log_event
        from utils.drawing import draw_box

        output_video = OUTPUT_VIDEO_DIR / f"out_{job_id}.mp4"
        output_csv = OUTPUT_CSV_DIR / f"events_{job_id}.csv"

        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)

        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        init_csv(output_csv)

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            progress = int((frame_id / total_frames) * 100)
            update_job(job_id, progress=min(progress, 99))

            tracks = vehicle_tracker.process_frame(frame)

            for t in tracks:
                helmet = helmet_detector.check(frame, t)
                weapon = weapon_detector.check(frame, t)
                fight = fight_detector.check(frame, t)

                events = event_manager.fuse(t, helmet, weapon, fight)

                for ev in events:
                    key = f"{t['track_id']}-{ev['type']}"
                    if dedup.allow(key):
                        log_event(output_csv, [
                            frame_id,
                            ev["type"],
                            t["track_id"],
                            ev["confidence"],
                            ev.get("details", "")
                        ])

                    draw_box(frame, t["bbox"], ev["label"], ev["color"])

            writer.write(frame)

        cap.release()
        writer.release()

        update_job(
            job_id,
            status="done",
            progress=100,
            video=f"/download/video/{output_video.name}",
            csv=f"/download/csv/{output_csv.name}"
        )

    except Exception as e:
        update_job(job_id, status="error", error=str(e))

# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
def health():
    return {"status": "running"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(400, "Unsupported video format")

    job_id = create_job()

    input_path = INPUT_DIR / f"in_{job_id}.mp4"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    thread = threading.Thread(
        target=process_video,
        args=(job_id, input_path),
        daemon=True
    )
    thread.start()

    return {"job_id": job_id}

@app.get("/status/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job

@app.get("/download/video/{filename}")
def download_video(filename: str):
    path = OUTPUT_VIDEO_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(path, media_type="video/mp4")

@app.get("/download/csv/{filename}")
def download_csv(filename: str):
    path = OUTPUT_CSV_DIR / filename
    if not path.exists():
        raise HTTPException(404, "CSV not found")
    return FileResponse(path, media_type="text/csv")
