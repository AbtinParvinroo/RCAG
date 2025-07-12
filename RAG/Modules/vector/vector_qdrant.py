from typing import List, Union
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import logging

class VectorStoreBase:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        self.logger = logging.getLogger(self.__class__.__name__)

    def add(self, ids: List[Union[str, int]], vectors: List[List[float]]):
        raise NotImplementedError()

    def search(self, query_vector: List[float], top_k: int = 10):
        raise NotImplementedError()

class QdrantVectorStore(VectorStoreBase):
    def __init__(self, dim: int, **kwargs):
        super().__init__(dim, **kwargs)
        self.client = QdrantClient(kwargs)
        self.collection = kwargs.get("collection_name", "smart_collection")
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
        )

    def add(self, ids: List[Union[str, int]], vectors: List[List[float]]):
        self.logger.info(f"Adding {len(ids)} vectors to Qdrant")
        points = [PointStruct(id=id, vector=vec) for id, vec in zip(ids, vectors)]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: List[float], top_k: int = 10):
        hits = self.client.search(collection_name=self.collection, query_vector=query_vector, limit=top_k)
        return [(hit.id, hit.score) for hit in hits]

class SmartVectorStore:
    def __init__(self, kwargs: dict, backend: str = "faiss", dim: int = 384, index_type: str = "flat"):
        backend = backend.lower()
        self.store = None
        if backend == "qdrant":
            self.store = QdrantVectorStore(dim, **kwargs)
        else:
            raise ValueError(f"Backend '{backend}' is not supported")

    def add(self, ids: List[Union[str, int]], vectors: List[List[float]]):
        return self.store.add(ids, vectors)

    def search(self, query_vector: List[float], top_k: int = 10):
        return self.store.search(query_vector, top_k)