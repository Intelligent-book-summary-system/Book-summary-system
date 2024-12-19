# booookscore/retrieval_db.py

import faiss
import numpy as np
import pickle
from typing import List, Dict, Any

class RetrievalDatabase:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add(self, embeddings: np.ndarray, texts: List[str]):
        if len(embeddings) != len(texts):
            raise ValueError("Number of embeddings must match number of texts")

        embeddings = np.array(embeddings).astype('float32')
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_vector: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        query_vector = np.array(query_vector).astype('float32')
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        distances, indices = self.index.search(query_vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.texts):
                results.append({
                    'text': self.texts[idx],
                    'score': float(1.0 / (1.0 + dist))
                })

        return results

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}_index")
        with open(f"{path}_texts.pkl", 'wb') as f:
            pickle.dump(self.texts, f)

    @classmethod
    def load(cls, path: str) -> 'RetrievalDatabase':
        db = cls()
        db.index = faiss.read_index(f"{path}_index")
        with open(f"{path}_texts.pkl", 'rb') as f:
            db.texts = pickle.load(f)
        return db