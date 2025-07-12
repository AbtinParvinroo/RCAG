import logging
from typing import List, Union
import weaviate

class VectorStoreBase:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        self.logger = logging.getLogger(self.__class__.__name__)

    def add(self, ids: List[Union[str, int]], vectors: List[List[float]]):
        raise NotImplementedError()

    def search(self, query_vector: List[float], top_k: int = 10):
        raise NotImplementedError()

class WeaviateVectorStore(VectorStoreBase):
    def __init__(self, dim: int, **kwargs):
        super().__init__(dim, **kwargs)
        self.client = weaviate.Client(**kwargs)
        self.class_name = kwargs.get("class_name", "Document")
        if not self.client.schema.contains({"class": self.class_name}):
            self.client.schema.create_class({
                "class": self.class_name,
                "vectorizer": "none",
                "vectorIndexType": "hnsw",
                "properties": [{"name": "text", "dataType": ["text"]}]
            })

    def add(self, ids: List[Union[str, int]], vectors: List[List[float]]):
        self.logger.info(f"Adding {len(ids)} vectors to Weaviate")
        data_objects = [{"id": str(id), "vector": vec} for id, vec in zip(ids, vectors)]
        for obj in data_objects:
            self.client.data_object.create(obj, class_name=self.class_name)

    def search(self, query_vector: List[float], top_k: int = 10):
        response = self.client.query.get(self.class_name, ["_additional {distance}"])\
            .with_near_vector({"vector": query_vector}).with_limit(top_k).do()
        parsed = []
        for item in response["data"]["Get"][self.class_name]:
            parsed.append((item["_additional"]["id"], item["_additional"]["distance"]))
        return parsed

class SmartVectorStore:
    def __init__(self, kwargs: dict, backend: str = "faiss", dim: int = 384, index_type: str = "flat"):
        backend = backend.lower()
        self.store = None
        if backend == "weaviate":
            self.store = WeaviateVectorStore(dim, **kwargs)
        else:
            raise ValueError(f"Backend '{backend}' is not supported")

    def add(self, ids: List[Union[str, int]], vectors: List[List[float]]):
        return self.store.add(ids, vectors)

    def search(self, query_vector: List[float], top_k: int = 10):
        return self.store.search(query_vector, top_k)