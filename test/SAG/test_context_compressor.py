from unittest.mock import MagicMock, patch
from context_compressor import (
    BaseCompressionStrategy,
    SummarizationCompressor,
    TopKCompressor,
    ClusterCompressor,
    MergeAndTrimCompressor,
    GraphCompressor,
    ContextCompressor
)

import numpy as np
import pytest

@pytest.fixture
def sample_chunks():
    """Provides a consistent list of text chunks for testing."""
    return [
        "The first sentence is about apples.",
        "The second sentence is about oranges.",
        "The third sentence is also about apples.",
        "The fourth sentence is about bananas.",
        "The fifth sentence discusses apples and oranges."
    ]

@pytest.fixture
def mock_embedder():
    """Provides a mock embedder that returns predictable vectors."""
    embedder = MagicMock()
    dummy_vectors = np.random.rand(5, 10).astype(np.float32)
    embedder.embed_passages.return_value = dummy_vectors
    embedder.embed_queries.return_value = [np.random.rand(10).astype(np.float32)]
    return embedder

def test_summarization_compressor(mocker):
    """Tests that the summarizer calls the transformers pipeline correctly."""
    mock_pipeline = MagicMock(return_value=[{"summary_text": "This is a summary."}])
    mocker.patch('context_compressor.pipeline', return_value=mock_pipeline)

    strategy = SummarizationCompressor()
    result = strategy.compress(["Chunk one.", "Chunk two."])

    mock_pipeline.assert_called_once_with("Chunk one.\\nChunk two.", max_length=512, min_length=30, do_sample=False)
    assert result == "This is a summary."

def test_topk_compressor(mocker, mock_embedder, sample_chunks):
    """Tests that the TopK compressor selects the highest-scoring chunks."""
    mock_scores = np.array([[0.9, 0.2, 0.8, 0.1, 0.7]])
    mocker.patch('context_compressor.cosine_similarity', return_value=mock_scores)
    strategy = TopKCompressor(embedder=mock_embedder, k=3)
    result = strategy.compress(sample_chunks, query="apples")
    mock_embedder.embed_queries.assert_called_once_with(["apples"])
    mock_embedder.embed_passages.assert_called_once_with(sample_chunks)
    assert result == [
        "The first sentence is about apples.",
        "The third sentence is also about apples.",
        "The fifth sentence discusses apples and oranges."
    ]

def test_topk_compressor_requires_query(mock_embedder):
    """Tests that TopKCompressor raises an error if no query is provided."""
    strategy = TopKCompressor(embedder=mock_embedder, k=3)
    with pytest.raises(ValueError, match="TopKCompressor به کوئری نیاز دارد."):
        strategy.compress(["some text"])

def test_cluster_compressor(mocker, mock_embedder, sample_chunks):
    """Tests that the Cluster compressor finds chunks closest to cluster centers."""
    mock_kmeans_instance = MagicMock()
    mock_kmeans_instance.fit.return_value = mock_kmeans_instance
    mock_kmeans_instance.labels_ = np.array([0, 1, 0, 1, 0])
    mock_kmeans_instance.cluster_centers_ = np.random.rand(2, 10)
    mock_kmeans_class = MagicMock(return_value=mock_kmeans_instance)
    mocker.patch('context_compressor.KMeans', mock_kmeans_class)
    mocker.patch('numpy.linalg.norm', side_effect=[
        np.array([0.1, 0.5, 0.8]), # Distances for cluster 0 (indices 0, 2, 4)
        np.array([0.2, 0.6]),      # Distances for cluster 1 (indices 1, 3)
    ])

    strategy = ClusterCompressor(embedder=mock_embedder, n_clusters=2)
    result = strategy.compress(sample_chunks)
    mock_kmeans_class.assert_called_once_with(n_clusters=2, random_state=42, n_init='auto')
    assert sorted(result) == sorted([
        "The first sentence is about apples.",
        "The second sentence is about oranges."
    ])

def test_merge_and_trim_compressor(mocker):
    """Tests that the MergeAndTrim compressor correctly truncates and decodes text."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [101, 102, 103, 104, 105]
    mock_tokenizer.decode.return_value = "Truncated text"
    mocker.patch('context_compressor.AutoTokenizer.from_pretrained', return_value=mock_tokenizer)
    strategy = MergeAndTrimCompressor(max_tokens=5)
    result = strategy.compress(["This is a long text.", "This is more text."])
    full_text = "This is a long text. This is more text."
    mock_tokenizer.encode.assert_called_once_with(full_text, truncation=True, max_length=5)
    mock_tokenizer.decode.assert_called_once_with([101, 102, 103, 104, 105], skip_special_tokens=True)
    assert result == "Truncated text"

def test_graph_compressor_raises_if_libs_missing(mocker):
    """Tests that GraphCompressor fails to initialize if spacy/networkx are missing."""
    mocker.patch('context_compressor.SPACY_AVAILABLE', False)
    with pytest.raises(ImportError):
        GraphCompressor()

def test_context_compressor_wrapper_delegates_to_strategy(mocker, sample_chunks):
    """
    Tests that the main ContextCompressor class correctly calls the 'compress'
    method of the strategy object it was initialized with.
    """
    mock_strategy = MagicMock(spec=BaseCompressionStrategy)
    mock_strategy.compress.return_value = ["mocked result"]
    compressor = ContextCompressor(strategy=mock_strategy)
    result = compressor.compress(sample_chunks, query="test query")
    mock_strategy.compress.assert_called_once_with(sample_chunks, query="test query")
    assert result == ["mocked result"]