from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_PATH = DATA_DIR / "index.faiss"
METADATA_PATH = DATA_DIR / "metadata.json"
JOBS_PATH = DATA_DIR / "jobs.json"
METRICS_PATH = DATA_DIR / "metrics.jsonl"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "4"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "imkrish/rag-qa-data")
