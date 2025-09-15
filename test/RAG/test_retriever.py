from unittest.mock import MagicMock, AsyncMock, ANY
from retriever import Retriever
import numpy as np
import pytest

@pytest.fixture
def mock_embedder():
    """Provides a mock embedder component."""
    embedder = MagicMock()
    embedder.embed_queries.return_value = [np.array([0.1, 0.2, 0.3])]
    return embedder

@pytest.fixture
def mock_vector_store():
    """Provides a mock vector store with both sync and async search methods."""
    candidates = [
        {"id": f"doc_{i}", "content": f"content {i}", "score": 1.0 - (i * 0.1)}
        for i in range(30)
    ]

    vector_store = MagicMock()
    vector_store.search.return_value = candidates
    vector_store.search_async = AsyncMock(return_value=candidates)
    return vector_store

@pytest.fixture
def mock_vector_store_sync_only():
    """Provides a mock vector store that only has a synchronous search method."""
    candidates = [
        {"id": f"doc_{i}", "content": f"content {i}", "score": 1.0 - (i * 0.1)}
        for i in range(30)
    ]

    vector_store = MagicMock()
    vector_store.search.return_value = candidates
    if hasattr(vector_store, 'search_async'):
        delattr(vector_store, 'search_async')

    return vector_store

@pytest.fixture
def mock_reranker():
    """Provides a mock reranker with both sync and async predict methods."""
    reranker = MagicMock()
    rerank_scores = [0.1, 0.9, 0.5, 0.8] + [0.0] * 26
    reranker.predict.return_value = rerank_scores
    reranker.predict_async = AsyncMock(return_value=rerank_scores)
    return reranker

def test_retrieve_without_reranker(mock_embedder, mock_vector_store):
    """Tests the basic synchronous retrieval flow without a reranker."""
    retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store)
    results = retriever.retrieve("test query", top_k=5)
    mock_embedder.embed_queries.assert_called_once_with(["test query"])
    mock_vector_store.search.assert_called_once_with(
        query_vector=mock_embedder.embed_queries.return_value[0],
        k=5,
        filters=None
    )

    assert len(results) == 5
    assert results[0]["id"] == "doc_0"

def test_retrieve_with_reranker(mock_embedder, mock_vector_store, mock_reranker):
    """Tests synchronous retrieval with a reranker, checking for expanded k and re-sorting."""
    retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store, reranker=mock_reranker)
    results = retriever.retrieve("test query", top_k=4)
    mock_vector_store.search.assert_called_once_with(
        query_vector=ANY,
        k=12,
        filters=None
    )

    mock_reranker.predict.assert_called_once()
    assert len(results) == 4
    assert results[0]["id"] == "doc_1"
    assert results[1]["id"] == "doc_3"

def test_retrieve_handles_errors_gracefully(mock_embedder, mock_vector_store, caplog):
    """Tests that if a component fails, the synchronous method logs an error and returns []."""
    mock_embedder.embed_queries.side_effect = ValueError("Embedding failed!")
    retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store)
    results = retriever.retrieve("test query")

    assert results == []
    assert "Error in synchronous retrieval pipeline" in caplog.text
    assert "Embedding failed!" in caplog.text

@pytest.mark.asyncio
async def test_retrieve_async_with_native_async_vectorstore(mock_embedder, mock_vector_store):
    """Tests the async path when the vector store has a native `search_async` method."""
    retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store)
    results = await retriever.retrieve_async("test query", top_k=5)
    mock_vector_store.search_async.assert_called_once()
    mock_vector_store.search.assert_not_called()
    assert len(results) == 5
    assert results[0]["id"] == "doc_0"

@pytest.mark.asyncio
async def test_retrieve_async_with_sync_vectorstore_fallback(mock_embedder, mock_vector_store_sync_only):
    """Tests the async path when the vector store lacks `search_async`, forcing a fallback."""
    retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store_sync_only)
    results = await retriever.retrieve_async("test query", top_k=5)
    mock_vector_store_sync_only.search.assert_called_once()
    assert len(results) == 5
    assert results[0]["id"] == "doc_0"

@pytest.mark.asyncio
async def test_retrieve_async_with_reranker(mock_embedder, mock_vector_store, mock_reranker):
    """Tests the full async pipeline with reranking."""
    retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store, reranker=mock_reranker)
    results = await retriever.retrieve_async("test query", top_k=4)

    mock_vector_store.search_async.assert_called_once_with(
        query_vector=ANY,
        k=12,
        filters=None
    )

    mock_reranker.predict_async.assert_called_once()
    mock_reranker.predict.assert_not_called()

    assert len(results) == 4
    assert results[0]["id"] == "doc_1"

@pytest.mark.asyncio
async def test_retrieve_async_handles_errors_gracefully(mock_embedder, mock_vector_store, caplog):
    """Tests that if a component fails, the async method logs an error and returns []."""
    mock_embedder.embed_queries.side_effect = ValueError("Async embedding failed!")
    retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store)
    results = await retriever.retrieve_async("test query")

    assert results == []
    assert "Error in asynchronous retrieval pipeline" in caplog.text
    assert "Async embedding failed!" in caplog.text