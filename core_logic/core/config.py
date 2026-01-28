from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    INDEX_PATH: str = "data/vector_store/index.faiss"
    METADATA_PATH: str = "data/vector_store/metadata.json"
    UPLOADS_DIR: str = "data/uploads"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    RATE_LIMIT: str = "20/minute"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

settings = Settings()
