import fitz  # PyMuPDF
import os
from typing import List
from core_logic.core.config import settings

class DocumentProcessor:
    @staticmethod
    def extract_text(file_path: str) -> str:
        extension = os.path.splitext(file_path)[1].lower()
        if extension == ".pdf":
            return DocumentProcessor._extract_from_pdf(file_path)
        elif extension == ".txt":
            return DocumentProcessor._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text

    @staticmethod
    def _extract_from_txt(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP) -> List[str]:
        chunks = []
        if not text:
            return chunks
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks
