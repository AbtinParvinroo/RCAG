from typing import List, Union
import logging
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

class VectorStoreBase:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        self.logger = logging.getLogger(self.__class__.__name__)

    def add(self, ids: List[Union[str, int]], vectors: List[List[float]]):
        raise NotImplementedError()

    def search(self, query_vector: List[float], top_k: int = 10):
        raise NotImplementedError()

class MilvusVectorStore(VectorStoreBase):
    def __init__(self, dim: int, **kwargs):
        super().__init__(dim, **kwargs)
        connections.connect(alias="default", **kwargs)
        self.collection_name = kwargs.get("collection_name", "smart_collection")
        if not utility.has_collection(self.collection_name):
            schema = CollectionSchema([
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ])
            self.collection = Collection(name=self.collection_name, schema=schema)
        else:
            self.collection = Collection(name=self.collection_name)

    def add(self, ids: List[Union[str, int]], vectors: List[List[float]]):
        self.logger.info(f"Adding {len(ids)} vectors to Milvus")
        self.collection.insert([ids, vectors])

    def search(self, query_vector: List[float], top_k: int = 10):
        results = self.collection.search([query_vector], "embedding", param={"metric_type": "L2"}, limit=top_k)
        hits = []
        for result in results[0]:
            hits.append((result.id, result.distance))
        return hits

class SmartVectorStore:
    def __init__(self, kwargs: dict, backend: str = "faiss", dim: int = 384, index_type: str = "flat"):
        backend = backend.lower()
        self.store = None
        if backend == "milvus":
            self.store = MilvusVectorStore(dim, **kwargs)
        else:
            raise ValueError(f"Backend '{backend}' is not supported")

    def add(self, ids: List[Union[str, int]], vectors: List[List[float]]):
        return self.store.add(ids, vectors)

    def search(self, query_vector: List[float], top_k: int = 10):
        return self.store.search(query_vector, top_k)