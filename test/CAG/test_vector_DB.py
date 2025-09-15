from vector_DB import SmartVectorStore, VectorStoreInitializationError
from unittest.mock import MagicMock, ANY
import numpy as np
import pytest
import uuid

DIM = 8
IDS = [str(uuid.uuid4()) for _ in range(2)]
VECTORS = np.random.rand(2, DIM).astype(np.float32)
METADATAS = [{"doc": "doc1"}, {"doc": "doc2"}]

@pytest.fixture(autouse=True)
def mock_all_backends(mocker):
    """
    Mocks all vector database client libraries before each test.
    This fixture ensures tests are isolated and don't depend on installed libraries
    by using `create=True` to handle missing optional dependencies.
    """
    mock_faiss_index = MagicMock()
    mock_faiss_index.ntotal = 0
    mock_faiss_module = MagicMock()
    mock_faiss_module.IndexFlat.return_value = MagicMock()
    mock_faiss_module.IndexIDMap.return_value = mock_faiss_index
    mocker.patch('vector_db.faiss', mock_faiss_module, create=True)
    mock_qdrant_client = MagicMock()
    mock_qdrant_client.get_collection.side_effect = Exception("not found")
    mocker.patch('vector_db.QdrantClient', return_value=mock_qdrant_client, create=True)
    mocker.patch('vector_db.models', MagicMock(), create=True)
    mock_chroma_collection = MagicMock()
    mock_chroma_client = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    mock_chroma_module = MagicMock()
    mock_chroma_module.EphemeralClient.return_value = mock_chroma_client
    mocker.patch('vector_db.chromadb', mock_chroma_module, create=True)
    mock_weaviate_collection = MagicMock()
    mock_weaviate_client = MagicMock()
    mock_weaviate_client.collections.exists.return_value = False
    mock_weaviate_client.collections.get.return_value = mock_weaviate_collection
    mock_weaviate_module = MagicMock()
    mock_weaviate_module.connect_to_local.return_value = mock_weaviate_client
    mocker.patch('vector_db.weaviate', mock_weaviate_module, create=True)
    mock_milvus_collection = MagicMock()
    mock_utility = MagicMock()
    mock_utility.has_collection.return_value = False
    mocker.patch('vector_db.utility', mock_utility, create=True)
    mocker.patch('vector_db.connections', MagicMock(), create=True)
    mocker.patch('vector_db.Collection', return_value=mock_milvus_collection, create=True)
    mocker.patch('vector_db.CollectionSchema', MagicMock(), create=True)
    mocker.patch('vector_db.FieldSchema', MagicMock(), create=True)
    mocker.patch('vector_db.DataType', MagicMock(), create=True)
    return {
        "faiss_index": mock_faiss_index,
        "qdrant_client": mock_qdrant_client,
        "chroma_collection": mock_chroma_collection,
        "weaviate_collection": mock_weaviate_collection,
        "milvus_collection": mock_milvus_collection,
    }

@pytest.fixture(params=["faiss", "qdrant", "chroma", "weaviate", "milvus"])
def backend(request):
    """Pytest fixture to parametrize tests across all supported backends."""
    return request.param

def test_initialization(backend, mock_all_backends):
    """Tests if the store initializes the correct backend without errors."""
    store = SmartVectorStore(backend=backend, dim=DIM)
    assert store.backend == backend
    assert not store._is_closed

def test_add_vectors(backend, mock_all_backends):
    """Tests the `add` method for each backend."""
    store = SmartVectorStore(backend=backend, dim=DIM)
    store.add(vectors=VECTORS, ids=IDS, metadatas=METADATAS)

    if backend == "faiss":
        mock_all_backends["faiss_index"].add_with_ids.assert_called_once()
        assert len(store.metadata_store) == 2

    elif backend == "qdrant":
        mock_all_backends["qdrant_client"].upsert.assert_called_once()

    elif backend == "chroma":
        mock_all_backends["chroma_collection"].add.assert_called_once()

    elif backend == "weaviate":
        assert mock_all_backends["weaviate_collection"].batch.dynamic().__enter__().add_object.call_count == 2

    elif backend == "milvus":
        mock_all_backends["milvus_collection"].insert.assert_called_once()

def test_search_vectors(backend, mock_all_backends):
    """Tests the `search` method and result parsing for each backend."""
    if backend == "faiss":
        hashed_id = hash(IDS[0]) & (2**63 - 1)
        mock_all_backends["faiss_index"].search.return_value = (np.array([[0.1]]), np.array([[hashed_id]]))
    elif backend == "qdrant":
        hit = MagicMock(id=IDS[0], score=0.9, payload=METADATAS[0])
        mock_all_backends["qdrant_client"].search.return_value = [hit]
    elif backend == "chroma":
            mock_all_backends["chroma_collection"].query.return_value = {
                'ids': [[IDS[0]]], 'distances': [[0.9]], 'metadatas': [[METADATAS[0]]]
            }

    elif backend == "weaviate":
        obj = MagicMock(uuid=MagicMock(hex=IDS[0]), metadata=MagicMock(distance=0.9), properties={'metadata_json': '{"doc": "doc1"}'})
        mock_all_backends["weaviate_collection"].query.near_vector.return_value.objects = [obj]

    elif backend == "milvus":
        hit = MagicMock(id=IDS[0], distance=0.9, entity=MagicMock(get=lambda x: '{"doc": "doc1"}'))
        mock_all_backends["milvus_collection"].search.return_value = [[hit]]

    store = SmartVectorStore(backend=backend, dim=DIM)
    if backend == "faiss":
        store.add(VECTORS, IDS, METADATAS)

    results = store.search(query_vector=VECTORS[0], k=1)

    assert len(results) == 1
    assert results[0]['id'] == IDS[0]
    assert 'score' in results[0]
    assert results[0]['metadata'] == METADATAS[0]

def test_delete_vectors(backend, mock_all_backends):
    """Tests the `delete` method for each backend."""
    store = SmartVectorStore(backend=backend, dim=DIM)
    store.delete(ids=[IDS[0]])
    if backend == "faiss":
        mock_all_backends["faiss_index"].remove_ids.assert_called_once()

    elif backend == "qdrant":
        mock_all_backends["qdrant_client"].delete.assert_called_once()

    elif backend == "chroma":
        mock_all_backends["chroma_collection"].delete.assert_called_once()

    elif backend == "weaviate":
        mock_all_backends["weaviate_collection"].data.delete_many.assert_called_once()

    elif backend == "milvus":
        mock_all_backends["milvus_collection"].delete.assert_called_once()

def test_count_vectors(backend, mock_all_backends):
    """Tests the `count` method for each backend."""
    if backend == "faiss":
        mock_all_backends["faiss_index"].ntotal = 10

    elif backend == "qdrant":
        mock_all_backends["qdrant_client"].get_collection.return_value.vectors_count = 10
        mock_all_backends["qdrant_client"].get_collection.side_effect = None

    elif backend == "chroma":
        mock_all_backends["chroma_collection"].count.return_value = 10

    elif backend == "weaviate":
        mock_all_backends["weaviate_collection"].aggregate.over_all.return_value.total_count = 10

    elif backend == "milvus":
        mock_all_backends["milvus_collection"].num_entities = 10

    store = SmartVectorStore(backend=backend, dim=DIM)
    assert store.count() == 10

def test_context_manager_closes_connection(mocker):
    """Tests that the __exit__ method calls close()."""
    close_spy = mocker.spy(SmartVectorStore, 'close')
    with SmartVectorStore(backend='weaviate', dim=DIM):
        pass

    close_spy.assert_called_once()

def test_missing_dependency_raises_error(mocker):
    """Tests that initialization fails if a required library is not installed."""
    mocker.patch('vector_db.QdrantClient', None)
    with pytest.raises(VectorStoreInitializationError):
        SmartVectorStore(backend='qdrant', dim=DIM)