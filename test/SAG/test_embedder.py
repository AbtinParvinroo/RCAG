
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

# Import the class to be tested
from embedder import SmartEmbedder

@pytest.fixture
def mock_sentence_transformer(mocker):
    """
    A comprehensive fixture that mocks the SentenceTransformer model and its
    dependencies like torch.
    """
    mocker.patch('embedder.torch.cuda.is_available', return_value=False)
    mocker.patch('embedder.torch.device', return_value='cpu')
    mock_model_instance = MagicMock()
    mock_model_instance.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    mock_st_class = mocker.patch('embedder.SentenceTransformer', return_value=mock_model_instance)
    return {
        "class": mock_st_class,
        "instance": mock_model_instance
    }

class TestSmartEmbedder:
    def test_initialization_is_lazy(self, mock_sentence_transformer):
        """Verifies that the model is NOT loaded when the class is instantiated."""
        st_class_mock = mock_sentence_transformer["class"]
        embedder = SmartEmbedder(model_name="test-model")
        st_class_mock.assert_not_called()
        assert embedder._model is None

    def test_lazy_loading_on_first_access(self, mock_sentence_transformer):
        """Verifies that accessing the `.model` property triggers the load exactly once."""
        st_class_mock = mock_sentence_transformer["class"]
        embedder = SmartEmbedder(model_name="test-model")
        _ = embedder.model
        st_class_mock.assert_called_once_with("test-model", device='cpu')
        assert embedder._model is not None

        _ = embedder.model
        st_class_mock.assert_called_once()

    def test_embed_queries_calls_model_encode(self, mock_sentence_transformer):
        """Tests that `embed_queries` correctly calls the underlying model's encode method."""
        st_instance_mock = mock_sentence_transformer["instance"]
        embedder = SmartEmbedder(model_name="test-model")
        queries = ("what is rag?", "what is sag?")
        embeddings = embedder.embed_queries(queries)
        st_instance_mock.encode.assert_called_once_with(queries, convert_to_numpy=True)

        assert np.array_equal(embeddings, st_instance_mock.encode.return_value)

    def test_embed_passages_calls_model_encode(self, mock_sentence_transformer):
        """Tests that `embed_passages` correctly calls the underlying model's encode method."""
        st_instance_mock = mock_sentence_transformer["instance"]
        embedder = SmartEmbedder(model_name="test-model")
        passages = ("RAG stands for Retrieval-Augmented Generation.", "SAG stands for...")
        embeddings = embedder.embed_passages(passages)
        st_instance_mock.encode.assert_called_once_with(passages, convert_to_numpy=True)

        assert np.array_equal(embeddings, st_instance_mock.encode.return_value)