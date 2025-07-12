from typing import List, Union
import chromadb
import logging

class VectorStoreBase:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        self.logger = logging.getLogger(self.__class__.__name__)

    def add(self, ids: List[Union[str, int]], vectors: List[List[float]]):
        raise NotImplementedError()

    def search(self, query_vector: List[float], top_k: int = 10):
        raise NotImplementedError()

class ChromaVectorStore(VectorStoreBase):
    def __init__(self, dim: int, **kwargs):
        super().__init__(dim, **kwargs)
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="smart_collection")

    def add(self, ids: List[Union[str, int]], vectors: List[List[float]]):
        self.logger.info(f"Adding {len(ids)} vectors to Chroma")
        self.collection.add(ids=ids, embeddings=vectors, documents=ids)

    def search(self, query_vector: List[float], top_k: int = 10):
        results = self.collection.query(query_vector=query_vector, n_results=top_k)
        return results

class SmartVectorStore:
    def __init__(self, kwargs: dict, backend: str = "faiss", dim: int = 384, index_type: str = "flat"):
        backend = backend.lower()
        self.store = None
        if backend == "chroma":
            self.store = ChromaVectorStore(dim, **kwargs)
        else:
            raise ValueError(f"Backend '{backend}' is not supported")

    def add(self, ids: List[Union[str, int]], vectors: List[List[float]]):
        return self.store.add(ids, vectors)

    def search(self, query_vector: List[float], top_k: int = 10):
        return self.store.search(query_vector, top_k)
