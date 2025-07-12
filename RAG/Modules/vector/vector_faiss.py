import faiss
import logging
from typing import List, Union
import numpy as np

class VectorStoreBase:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        self.logger = logging.getLogger(self.__class__.__name__)

    def add(self, ids: List[Union[str, int]], vectors: List[List[float]]):
        raise NotImplementedError()

    def search(self, query_vector: List[float], top_k: int = 10):
        raise NotImplementedError()

class FaissVectorStore(VectorStoreBase):
    def __init__(self, dim: int, index_type: str = "flat", **kwargs):
        super().__init__(dim, **kwargs)
        self.index_type = index_type.lower()
        self.ids = []
        self.trained = False
        self._setup_index(kwargs)

    def _setup_index(self, kwargs):
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.dim)
        elif self.index_type == "ivf":
            nlist = kwargs.get("nlist", 100)
            quantizer = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
            self.trained = False
        elif self.index_type == "pq":
            self.index = faiss.IndexPQ(self.dim, M=8, nbits=8)
            self.trained = False
        elif self.index_type == "ivf_pq":
            nlist = kwargs.get("nlist", 100)
            quantizer = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, M=8, nbits=8)
            self.trained = False
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dim, 32)
        else:
            print(f"Unknown FAISS index type: {self.index_type}")

    def add(self, ids: List[Union[str, int]], vectors: List[List[float]]):
        self.logger.info(f"Adding {len(ids)} vectors to Faiss ({self.index_type})")
        vectors = np.array(vectors).astype("float32")
        if not self.trained and hasattr(self.index, "is_trained") and not self.index.is_trained:
            self.index.train(vectors)
            self.trained = True
        self.index.add(vectors)
        self.ids.extend(ids)

    def search(self, query_vector: List[float], top_k: int = 10):
        query_vector = np.array([query_vector]).astype("float32")
        D, I = self.index.search(query_vector, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(self.ids):
                results.append((self.ids[idx], dist))
        return results

class SmartVectorStore:
    def __init__(self, kwargs: dict, backend: str = "faiss", dim: int = 384, index_type: str = "flat"):
        backend = backend.lower()
        self.store = None
        if backend == "faiss":
            self.store = FaissVectorStore(dim, index_type=index_type, **kwargs)
        else:
            raise ValueError(f"Backend '{backend}' is not supported")

    def add(self, ids: List[Union[str, int]], vectors: List[List[float]]):
        return self.store.add(ids, vectors)

    def search(self, query_vector: List[float], top_k: int = 10):
        return self.store.search(query_vector, top_k)