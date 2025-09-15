from detector import DataTypeDetector
import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def detector():
    """Provides a DataTypeDetector instance for testing."""
    return DataTypeDetector()

def test_detect_dataframe(detector):
    """
    Verifies that a pandas DataFrame is correctly identified.
    """
    test_data = pd.DataFrame({'column_a': [1, 2], 'column_b': ['x', 'y']})
    assert detector.detect(test_data) == "dataframe"

def test_detect_numpy_array(detector):
    """
    Verifies that a NumPy ndarray is correctly identified.
    """
    test_data = np.array([10, 20, 30])
    assert detector.detect(test_data) == "ndarray"

def test_detect_simple_list(detector):
    """
    Verifies that a standard list is correctly identified.
    """
    test_data = ["apple", "banana", "cherry"]
    assert detector.detect(test_data) == "list"

def test_detect_preprocessed_list(detector):
    """
    Verifies that a list of dictionaries (pre-chunked data) is identified.
    """
    test_data = [
        {"content": "This is the first chunk."},
        {"content": "This is the second chunk."}
    ]

    assert detector.detect(test_data) == "preprocessed"

def test_detect_folder_path(detector, monkeypatch):
    """
    Verifies folder path detection by mocking os.path.isdir.
    """
    monkeypatch.setattr('os.path.isdir', lambda path: True)
    assert detector.detect('/fake/folder') == "folder"

@pytest.mark.parametrize("file_path, expected_type", [
    ("data.csv", "table_file"),
    ("archive/report.xlsx", "table_file"),
    ("media/clip.mp4", "video_file"),
    ("assets/song.mp3", "audio_file"),
    ("images/photo.jpeg", "image_file"),
    ("docs/document.pdf", "document_file"),
    ("README.md", "document_file"),
    ("backup.zip", "file"),
])

def test_detect_file_paths(detector, monkeypatch, file_path, expected_type):
    """
    Verifies various file types using a parameterized test.
    Mocks os.path.isfile to always return True for these tests.
    """
    monkeypatch.setattr('os.path.isfile', lambda path: True)
    monkeypatch.setattr('os.path.isdir', lambda path: False)
    assert detector.detect(file_path) == expected_type

def test_detect_raw_text(detector, monkeypatch):
    """
    Verifies that a string which is not a file or folder is treated as raw text.
    """
    monkeypatch.setattr('os.path.isfile', lambda path: False)
    monkeypatch.setattr('os.path.isdir', lambda path: False)
    test_string = "This is a sentence, not a path."
    assert detector.detect(test_string) == "raw_text"

@pytest.mark.parametrize("unsupported_data", [
    42,
    3.14159,
    {"key": "value"},
    ("a", 1, None),
    None
])

def test_detect_unsupported_types(detector, unsupported_data):
    """
    Verifies that unhandled data types return the correct fallback string.
    """
    assert detector.detect(unsupported_data) == 'unsupported_data_type'