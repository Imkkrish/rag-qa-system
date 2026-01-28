import json
import uuid
from datetime import datetime
from threading import Lock
from typing import Optional, Dict
from .config import JOBS_PATH, DATA_DIR

_lock = Lock()


def _ensure():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not JOBS_PATH.exists():
        JOBS_PATH.write_text(json.dumps({}))


def create_job(filename: str) -> str:
    _ensure()
    job_id = str(uuid.uuid4())
    payload = {
        "id": job_id,
        "filename": filename,
        "status": "queued",
        "error": None,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    with _lock:
        jobs = json.loads(JOBS_PATH.read_text())
        jobs[job_id] = payload
        JOBS_PATH.write_text(json.dumps(jobs, indent=2))
    return job_id


def update_job(job_id: str, status: str, error: Optional[str] = None):
    _ensure()
    with _lock:
        jobs = json.loads(JOBS_PATH.read_text())
        if job_id in jobs:
            jobs[job_id]["status"] = status
            jobs[job_id]["error"] = error
            jobs[job_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"
            JOBS_PATH.write_text(json.dumps(jobs, indent=2))


def get_job(job_id: str) -> Optional[Dict]:
    _ensure()
    with _lock:
        jobs = json.loads(JOBS_PATH.read_text())
    return jobs.get(job_id)
