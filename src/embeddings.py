import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import pickle
import os
from src.config import settings

class EmbeddingsManager:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.texts = []

    def create_embeddings(self, texts: List[str]):
        """Create embeddings for texts and build FAISS index"""
        self.texts = texts
        embeddings = self.model.encode(texts)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def save(self):
        """Save the index and texts"""
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        faiss.write_index(self.index, f"{settings.VECTOR_DB_PATH}/docs.index")
        with open(f"{settings.VECTOR_DB_PATH}/texts.pkl", 'wb') as f:
            pickle.dump(self.texts, f)

    def load(self):
        """Load the index and texts"""
        self.index = faiss.read_index(f"{settings.VECTOR_DB_PATH}/docs.index")
        with open(f"{settings.VECTOR_DB_PATH}/texts.pkl", 'rb') as f:
            self.texts = pickle.load(f)

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Search for similar texts"""
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        return [(self.texts[idx], dist) for idx, dist in zip(indices[0], distances[0])]