from context_assembler import ContextAssembler, PromptFormatter
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def mock_image_class(mocker):
    """Mocks the PIL.Image.Image *class* so that isinstance() works."""
    class MockImage:
        pass

    mocker.patch('context_assembler.Image', MockImage, create=True)
    return MockImage

@pytest.fixture
def mock_audio_class(mocker):
    """Mocks the pydub.AudioSegment *class*."""
    class MockAudio:
        pass

    mocker.patch('context_assembler.AudioSegment', MockAudio, create=True)
    return MockAudio

@pytest.fixture
def mock_image(mock_image_class):
    """Provides an *instance* of the mocked Image class."""
    return mock_image_class()

@pytest.fixture
def mock_audio(mock_audio_class):
    """Provides an *instance* of the mocked Audio class."""
    return mock_audio_class()

@pytest.fixture
def sample_chunks(mock_image, mock_audio):
    """Provides a consistent, mixed list of data chunks for testing."""
    return [
        "This is the first text chunk.",
        pd.DataFrame({'col1': [1], 'col2': ['A']}),
        mock_image,
        "This is the second text chunk.",
        {"city": "New York", "temp": 75},
        mock_audio,
        np.array([[1, 2], [3, 4]]),
    ]

class TestContextAssembler:
    @pytest.mark.parametrize("chunk_data, expected_type", [
        ("a string", "text"),
        (pd.DataFrame(), "table"),
        (np.array([]), "image_array"),
        ({}, "structured_data"),
        (123, "unknown"),
    ])

    def test_detect_type_basic(self, chunk_data, expected_type):
        """Tests type detection for basic Python types."""
        assembler = ContextAssembler()
        assert assembler._detect_type(chunk_data) == expected_type

    def test_detect_type_with_mocked_objects(self, mock_image, mock_audio):
        """Tests type detection for optional dependency types (Image, Audio)."""
        assembler = ContextAssembler()
        assert assembler._detect_type(mock_image) == "image"
        assert assembler._detect_type(mock_audio) == "audio"

    def test_assemble_groups_by_type(self, sample_chunks):
        """Tests the default behavior of grouping chunks into a dictionary."""
        assembler = ContextAssembler()
        result = assembler.assemble(sample_chunks)

        assert "text" in result
        assert len(result["text"]) == 2
        assert result["text"][0] == "This is the first text chunk."
        assert "table" in result
        assert len(result["table"]) == 1
        assert "image" in result
        assert "audio" in result
        assert "structured_data" in result
        assert "image_array" in result

    def test_assemble_with_priority_ordering(self, sample_chunks):
        """Tests that the 'priority' argument correctly orders and flattens the output."""
        priority = ["image", "table", "text"]
        assembler = ContextAssembler(priority=priority)
        result = assembler.assemble(sample_chunks)

        assert list(result.keys()) == ["mixed"]
        mixed_chunks = result["mixed"]
        assert len(mixed_chunks) == 7
        assert assembler._detect_type(mixed_chunks[0]) == "image"
        assert assembler._detect_type(mixed_chunks[1]) == "table"
        assert assembler._detect_type(mixed_chunks[2]) == "text"
        assert assembler._detect_type(mixed_chunks[3]) == "text"

    def test_assemble_handles_empty_list(self):
        """Tests that an empty input list results in an empty dictionary."""
        assembler = ContextAssembler()
        assert assembler.assemble([]) == {}

class TestPromptFormatter:
    @pytest.fixture
    def formatter(self):
        return PromptFormatter()

    def test_format_context_to_string(self, formatter):
        """Tests the detailed string conversion for different data types."""
        context = {
            "text": ["First line.", "Second line."],
            "table": [pd.DataFrame({'Header': ['Value']})],
            "image": [MagicMock(), MagicMock()],
            "structured_data": [{"city": "LA", "temp": 80}]
        }

        result_str = formatter.format_context_to_string(context)

        expected_table_str = "--- Table 1 ---\nHeader\n Value"
        expected_data_str = "--- Data ---\ncity  temp\n  LA    80"

        assert "First line.\nSecond line." in result_str
        assert expected_table_str in result_str
        assert "--- Image Content ---\n[2 image item(s) provided]" in result_str
        assert expected_data_str in result_str

    def test_build_prompt_full(self, formatter):
        """Tests the construction of a complete prompt with all parts."""
        context = {"text": ["Some context."]}
        query = "What is the context?"
        instruction = "You are a helpful assistant."
        prompt = formatter.build_prompt(context, query, instruction)

        assert prompt.startswith("### Instruction\nYou are a helpful assistant.")
        assert "### Context\nSome context." in prompt
        assert "### Question\nWhat is the context?" in prompt
        assert prompt.endswith("### Answer")

    def test_build_prompt_no_instruction(self, formatter):
        context = {"text": ["Some context."]}
        query = "What is the context?"
        prompt = formatter.build_prompt(context, query)
        assert not prompt.startswith("### Instruction")
        assert prompt.startswith("### Context")

    def test_build_prompt_no_context(self, formatter):
        query = "What is 2+2?"
        instruction = "Solve the math problem."
        prompt = formatter.build_prompt({}, query, instruction)
        assert "### Instruction\nSolve the math problem." in prompt
        assert "### Context" not in prompt
        assert "### Question\nWhat is 2+2?" in prompt