from typing import List, Union
import numpy as np 
import logging
import faiss
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct 
import weaviate
import chromadb
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType


class SmartVectorStore: 
    def init(self, kwargs, backend: str = "faiss", dim: int = 384, index_type: str = "flat"):
        self.backend = backend.lower()
        self.index_type = index_type.lower()
        self.dim = dim
        self.logger = logging.getLogger("SmartVectorStore")
        self._setup_backend(kwargs)

def _setup_backend(self, kwargs):
    if self.backend == "faiss":
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
            return f"Unknown FAISS index type: {self.index_type}"
        self.ids = []

    elif self.backend == "qdrant":
        self.client = QdrantClient(kwargs)
        self.collection = kwargs.get("collection_name", "smart_collection")
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
        )

    elif self.backend == "chroma":
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="smart_collection")

    elif self.backend == "weaviate":
        self.client = weaviate.Client(**kwargs)
        self.class_name = kwargs.get("class_name", "Document")
        if not self.client.schema.contains({"class": self.class_name}):
            self.client.schema.create_class({
                "class": self.class_name,
                "vectorizer": "none",
                "vectorIndexType": "hnsw",
                "properties": [{"name": "text", "dataType": ["text"]}]
            })

    elif self.backend == "milvus":
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

    else:
        return f"Backend '{self.backend}' is not supported"

def add(self, ids: List[str], vectors: List[List[float]]):
    self.logger.info(f"Adding {len(ids)} vectors to {self.backend}")
    if self.backend == "faiss":
        vectors = np.array(vectors).astype("float32")

        if hasattr(self, "trained") and not self.index.is_trained:
            self.index.train(vectors)
        self.index.add(vectors)
        self.ids.extend(ids)

    elif self.backend == "qdrant":
        points = [PointStruct(id=id, vector=vec) for id, vec in zip(ids, vectors)]
        self.client.upsert(self.collection, points)

    elif self.backend == "chroma":
        self.collection.add(ids=ids, embeddings=vectors, documents=ids)