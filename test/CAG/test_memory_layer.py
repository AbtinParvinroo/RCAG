from memory_layer import MemoryLayer
from unittest.mock import MagicMock
import pytest
import json

@pytest.fixture
def mock_redis_class(mocker):
    """Mocks the redis.Redis class and returns the mock class itself."""
    mock_client_instance = MagicMock()
    mock_class = mocker.patch('memory_layer.redis.Redis', return_value=mock_client_instance)
    return mock_class

@pytest.fixture
def memory_session(mock_redis_class):
    """Provides a MemoryLayer instance which will use the mocked Redis class."""
    return MemoryLayer(session_id="test_session", max_len=10)

class TestMemoryLayer:
    def test_initialization(self, mock_redis_class):
        """
        Tests that the MemoryLayer initializes correctly and creates a Redis client.
        """
        session_id = "unique_session_123"
        memory = MemoryLayer(session_id=session_id, host="my-redis", port=1234)
        mock_redis_class.assert_called_once_with(
            host="my-redis", port=1234, db=0, decode_responses=True
        )

        assert memory.session_id == f"history:{session_id}"

    def test_add_message(self, memory_session):
        """
        Tests that adding a message calls rpush and ltrim on the Redis client
        with the correct, JSON-formatted arguments.
        """
        role = "user"
        content = "Hello, world!"
        mock_client = memory_session.client
        memory_session.add_message(role=role, content=content)
        expected_message = json.dumps({"role": role, "content": content})
        mock_client.rpush.assert_called_once_with(memory_session.session_id, expected_message)
        mock_client.ltrim.assert_called_once_with(memory_session.session_id, -10, -1)

    def test_get_history(self, memory_session):
        """
        Tests that get_history calls lrange and correctly decodes the JSON results.
        """
        mock_client = memory_session.client
        mock_history_json = [
            '{"role": "user", "content": "Hello"}',
            '{"role": "assistant", "content": "Hi there!"}'
        ]

        mock_client.lrange.return_value = mock_history_json
        history = memory_session.get_history()
        mock_client.lrange.assert_called_once_with(memory_session.session_id, 0, -1)
        expected_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        assert history == expected_history

    def test_format_for_prompt(self, memory_session):
        """
        Tests that the history is formatted into a human-readable list of strings.
        """
        mock_client = memory_session.client
        mock_history_json = [
            '{"role": "system", "content": "You are a helpful assistant."}',
            '{"role": "user", "content": "What is pytest?"}'
        ]

        mock_client.lrange.return_value = mock_history_json
        formatted_history = memory_session.format_for_prompt()
        expected_format = [
            "System: You are a helpful assistant.",
            "User: What is pytest?"
        ]

        assert formatted_history == expected_format

    def test_modern_format(self, memory_session):
        """
        Tests that the modern format returns the history as a list of dicts.
        """
        mock_client = memory_session.client
        mock_history_json = [
            '{"role": "user", "content": "Hello"}',
            '{"role": "assistant", "content": "Hi there!"}'
        ]

        mock_client.lrange.return_value = mock_history_json
        modern_history = memory_session.modern_format()
        expected_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        assert modern_history == expected_history

    def test_clear(self, memory_session):
        """
        Tests that the clear method calls the delete command on the Redis client.
        """
        mock_client = memory_session.client
        memory_session.clear()
        mock_client.delete.assert_called_once_with(memory_session.session_id)