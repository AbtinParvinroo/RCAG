from typing import List, Dict, Optional, Any
import numpy as np
import logging
import json
import uuid
try: import faiss

except ImportError: faiss = None

try: from qdrant_client import QdrantClient, models

except ImportError: QdrantClient = None

try: import chromadb

except ImportError: chromadb = None

try: import weaviate

except ImportError: weaviate = None

try: from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

except ImportError: Collection = None

class VectorStoreInitializationError(Exception): pass
class VectorStoreOperationError(Exception): pass

class SmartVectorStore:
    """
    A unified, production-ready interface for multiple vector DB backends,
    rewritten for robustness, modern APIs, and consistent behavior.
    """
    def __init__(self, backend: str = "faiss", dim: int = 384, distance: str = "l2", **kwargs):
        self.backend = backend.lower()
        self.dim = dim
        self.distance = distance.lower()
        self.logger = logging.getLogger(f"SmartVectorStore-{self.backend}")
        self._client: Any = None
        self._collection: Any = None
        self._is_closed = False
        self._setup_backend(**kwargs)
        self.logger.info(f"Successfully initialized backend '{self.backend}'")

    def _setup_backend(self, **kwargs):
        """Initializes the chosen vector database backend."""
        try:
            if self.backend == "faiss":
                if not faiss: raise ImportError("FAISS not installed.")
                metric = faiss.METRIC_INNER_PRODUCT if self.distance == 'cosine' else faiss.METRIC_L2
                index = faiss.IndexFlat(self.dim, metric)
                self._collection = faiss.IndexIDMap(index)
                self.metadata_store: Dict[int, Dict[str, Any]] = {} # Hash to metadata
                self.id_map: Dict[int, str] = {} # Hash to original ID

            elif self.backend == "qdrant":
                if not QdrantClient: raise ImportError("qdrant-client not installed.")
                self._client = QdrantClient(**kwargs)
                collection_name = kwargs.get("collection_name", "smart_collection")
                dist = models.Distance.COSINE if self.distance == "cosine" else models.Distance.EUCLID
                try:
                    self._client.get_collection(collection_name=collection_name)

                except Exception:
                    self._client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(size=self.dim, distance=dist)
                    )

                self._collection = collection_name

            elif self.backend == "chroma":
                if not chromadb: raise ImportError("chromadb-client not installed.")

                self._client = kwargs.get("client", chromadb.EphemeralClient())
                collection_name = kwargs.get("collection_name", "smart_collection")
                metadata = {"hnsw:space": "cosine" if self.distance == "cosine" else "l2"}
                self._collection = self._client.get_or_create_collection(name=collection_name, metadata=metadata)

            elif self.backend == "weaviate":
                if not weaviate: raise ImportError("weaviate-client not installed.")

                self._client = weaviate.connect_to_local(**kwargs.get("connection_params", {}))
                class_name = kwargs.get("class_name", "Document")
                if not self._client.collections.exists(class_name):
                    distance_metric = "cosine" if self.distance == "cosine" else "l2-squared"
                    self._client.collections.create(
                        name=class_name,
                        vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
                        vector_index_config=weaviate.classes.config.Configure.VectorIndex.hnsw(
                            distance_metric=weaviate.classes.config.VectorDistance(distance_metric)
                        ),

                        properties=[
                            weaviate.classes.config.Property(name="metadata_json", data_type=weaviate.classes.config.DataType.TEXT)
                        ]
                    )
                self._collection = self._client.collections.get(class_name)

            elif self.backend == "milvus":
                if not Collection: raise ImportError("pymilvus not installed.")

                connections.connect(alias="default", **kwargs)
                collection_name = kwargs.get("collection_name", "smart_collection")
                if not utility.has_collection(collection_name):
                    fields = [
                        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
                        FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65_535),
                        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
                    ]

                    schema = CollectionSchema(fields)
                    self._collection = Collection(collection_name, schema)
                    metric_type = "IP" if self.distance == "cosine" else "L2"
                    index_params = {"metric_type": metric_type, "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
                    self._collection.create_index("embedding", index_params)

                else:
                    self._collection = Collection(collection_name)

                self._collection.load()

            else:
                raise ValueError(f"Backend '{self.backend}' is not supported.")

        except Exception as e:
            self.logger.error(f"Failed to setup backend '{self.backend}': {e}", exc_info=True)
            raise VectorStoreInitializationError(f"Failed to setup backend '{self.backend}': {e}")

    def add(self, vectors: np.ndarray, ids: List[str], metadatas: Optional[List[Dict]] = None, batch_size: int = 100):
        if self._is_closed: raise RuntimeError("Vector store is closed.")

        if not ids: raise ValueError("IDs must be provided.")

        metadatas = metadatas if metadatas is not None else [{} for _ in range(len(vectors))]
        for i in range(0, len(vectors), batch_size):
            batch_vectors = np.array(vectors[i:i+batch_size], dtype=np.float32)
            batch_ids = ids[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            self.logger.info(f"Adding batch of {len(batch_vectors)} vectors to {self.backend}")
            try:
                if self.backend == "faiss":
                    if self.distance == 'cosine': faiss.normalize_L2(batch_vectors)
                    id_hashes = np.array([hash(id_str) & (2**63 - 1) for id_str in batch_ids], dtype=np.int64)
                    self._collection.add_with_ids(batch_vectors, id_hashes)
                    for j, id_hash in enumerate(id_hashes):
                        self.metadata_store[int(id_hash)] = batch_metadatas[j]
                        self.id_map[int(id_hash)] = batch_ids[j]

                elif self.backend == "qdrant":
                    points = [models.PointStruct(id=id, vector=vec.tolist(), payload=meta) for id, vec, meta in zip(batch_ids, batch_vectors, batch_metadatas)]
                    self._client.upsert(collection_name=self._collection, points=points, wait=True)

                elif self.backend == "chroma":
                    self._collection.add(embeddings=batch_vectors.tolist(), ids=batch_ids, metadatas=batch_metadatas)

                elif self.backend == "weaviate":
                    with self._collection.batch.dynamic() as batch:
                        for id, vec, meta in zip(batch_ids, batch_vectors, batch_metadatas):
                            properties = {"metadata_json": json.dumps(meta)}
                            batch.add_object(properties=properties, vector=vec, uuid=id)

                elif self.backend == "milvus":
                    data = [batch_ids, [json.dumps(m) for m in batch_metadatas], batch_vectors.tolist()]
                    self._collection.insert(data)

            except Exception as e:
                self.logger.error(f"Failed to add batch to '{self.backend}': {e}", exc_info=True)
                raise VectorStoreOperationError(f"Failed to add batch to '{self.backend}': {e}")

    def search(self, query_vector: np.ndarray, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Searches for k similar vectors with optional metadata filtering."""
        if self._is_closed: raise RuntimeError("Vector store is closed.")
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)

        try:
            if self.backend == "faiss":
                if filters: self.logger.warning("FAISS backend does not support metadata filtering.")
                if self.distance == 'cosine': faiss.normalize_L2(query_vector)
                distances, id_hashes = self._collection.search(query_vector, k)
                results = []
                for i, id_hash in enumerate(id_hashes[0]):
                    if id_hash != -1:
                        original_id = self.id_map.get(int(id_hash))
                        if original_id:
                            results.append({"id": original_id, "score": float(distances[0][i]), "metadata": self.metadata_store.get(int(id_hash), {})})

                return results

            elif self.backend == "qdrant":
                must_conditions = []
                if filters:
                    for key, value in filters.items():
                        must_conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))

                hits = self._client.search(
                    collection_name=self._collection,
                    query_vector=query_vector[0],
                    query_filter=models.Filter(must=must_conditions) if must_conditions else None,
                    limit=k
                )

                return [{"id": hit.id, "score": hit.score, "metadata": hit.payload} for hit in hits]

            elif self.backend == "chroma":
                res = self._collection.query(
                    query_embeddings=[query_vector[0].tolist()],
                    n_results=k,
                    where=filters
                )

                if not res['ids'] or not res['ids'][0]: return []

                ids, distances, metadatas = res['ids'][0], res['distances'][0], res['metadatas'][0]
                return [{"id": id, "score": score, "metadata": meta} for id, score, meta in zip(ids, distances, metadatas)]

            elif self.backend == "weaviate":
                where_filter = weaviate.classes.query.Filter.by_property("metadata_json").contains_all(list(filters.values())) if filters else None
                res = self._collection.query.near_vector(
                    near_vector=query_vector[0],
                    limit=k,
                    filters=where_filter,
                    return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
                )

                return [{"id": o.uuid.hex, "score": o.metadata.distance, "metadata": json.loads(o.properties['metadata_json'])} for o in res.objects]

            elif self.backend == "milvus":
                filter_expr = " and ".join([f"json_extract(metadata_json, '$.{k}') == '{v}'" for k, v in filters.items()]) if filters else ""
                res = self._collection.search(
                    data=query_vector, anns_field="embedding",
                    param={"metric_type": "IP" if self.distance == "cosine" else "L2", "params": {"nprobe": 10}},
                    limit=k, expr=filter_expr, output_fields=["id", "metadata_json"]
                )[0]

                return [{"id": hit.id, "score": hit.distance, "metadata": json.loads(hit.entity.get("metadata_json"))} for hit in res]

        except Exception as e:
            self.logger.error(f"Failed to search in '{self.backend}': {e}", exc_info=True)
            raise VectorStoreOperationError(f"Failed to search in '{self.backend}': {e}")

        return []

    def delete(self, ids: List[str]):
        """Deletes vectors by their IDs."""
        if self._is_closed: raise RuntimeError("Vector store is closed.")
        self.logger.info(f"Deleting {len(ids)} vectors from {self.backend}")
        try:
            if self.backend == "faiss":
                id_hashes = np.array([hash(id_str) & (2**63 - 1) for id_str in ids], dtype=np.int64)
                self._collection.remove_ids(id_hashes)
                for id_hash in id_hashes:
                    self.metadata_store.pop(int(id_hash), None)
                    self.id_map.pop(int(id_hash), None)

            elif self.backend == "qdrant":
                self._client.delete(collection_name=self._collection, points_selector=models.PointIdsList(points=ids), wait=True)

            elif self.backend == "chroma":
                self._collection.delete(ids=ids)

            elif self.backend == "weaviate":
                self._collection.data.delete_many(where=weaviate.classes.query.Filter.by_id().contains_any([uuid.UUID(id_str) for id_str in ids]))

            elif self.backend == "milvus":
                expr = f"id in {json.dumps(ids)}"
                self._collection.delete(expr)

        except Exception as e:
            self.logger.error(f"Deletion failed in '{self.backend}': {e}", exc_info=True)
            raise VectorStoreOperationError(f"Deletion failed in '{self.backend}': {e}")

    def count(self) -> int:
        """Returns the total number of vectors in the store."""
        if self._is_closed: return 0
        try:
            if self.backend == "faiss":
                return self._collection.ntotal

            elif self.backend == "qdrant":
                return self._client.get_collection(collection_name=self._collection).vectors_count

            elif self.backend == "chroma":
                return self._collection.count()

            elif self.backend == "weaviate":
                response = self._collection.aggregate.over_all(total_count=True)
                return response.total_count

            elif self.backend == "milvus":
                self._collection.flush()
                return self._collection.num_entities

        except Exception as e:
            self.logger.error(f"Failed to get count from '{self.backend}': {e}")
            return 0

        return 0

    def close(self):
        """Closes connections and releases resources."""
        if self._is_closed: return

        self.logger.info(f"Closing connection for {self.backend}...")
        if self.backend == "weaviate" and self._client:
            self._client.close()

        elif self.backend == "milvus":
            connections.disconnect("default")

        self._is_closed = True
        self.logger.info(f"{self.backend} connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        return self.count()