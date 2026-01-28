import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from core_logic.core.config import settings
from typing import List, Dict

class VectorStore:
    def __init__(self):
        self._model = None
        self._index = None
        self.metadata = []
        self._metadata_loaded = False

    @property
    def model(self):
        if self._model is None:
            print(f"Loading embedding model: {settings.EMBEDDING_MODEL}...")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self._model

    @property
    def index(self):
        if self._index is None:
            self._load_index()
        return self._index

    def _load_metadata(self):
        if not self._metadata_loaded:
            if os.path.exists(settings.METADATA_PATH):
                with open(settings.METADATA_PATH, "r") as f:
                    self.metadata = json.load(f)
            self._metadata_loaded = True

    def _load_index(self):
        if os.path.exists(settings.INDEX_PATH):
            self._index = faiss.read_index(settings.INDEX_PATH)
        else:
            # Dimension for all-MiniLM-L6-v2 is 384
            self._index = faiss.IndexFlatL2(384)
        self._load_metadata()

    def add_documents(self, chunks: List[str], doc_name: str):
        if not chunks:
            return
        
        embeddings = self.model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")
        
        self.index.add(embeddings)
        
        for chunk in chunks:
            self.metadata.append({"text": chunk, "source": doc_name})
            
        self._save_index()

    def _save_index(self):
        faiss.write_index(self.index, settings.INDEX_PATH)
        with open(settings.METADATA_PATH, "w") as f:
            json.dump(self.metadata, f)

    def search(self, query: str, k: int = 4) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
            
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results

vector_store = VectorStore()
