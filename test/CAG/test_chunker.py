from unittest.mock import patch, MagicMock, mock_open
from chunker import RAGChunker
import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def mock_tokenizer(mocker):
    """Mocks the Hugging Face tokenizer to avoid network calls."""
    mock = MagicMock()
    mock.encode = lambda text, add_special_tokens: text.split()
    mock.decode = lambda tokens: " ".join(tokens)
    mocker.patch('chunker.AutoTokenizer.from_pretrained', return_value=mock)
    return mock

@pytest.fixture
def chunker(mock_tokenizer):
    """Provides a RAGChunker instance with a mocked tokenizer and test-friendly settings."""
    return RAGChunker(
        chunk_size=10,
        overlap=2,
        image_tile_size=(128, 128),
        video_chunk_duration=1,
        max_audio_chunk_duration=5000
    )

class MockCv2Capture:
    """A mock for cv2.VideoCapture to simulate video reading without a real file."""
    def __init__(self, total_frames=50, fps=25):
        self._is_opened = True
        self._frame_count = 0
        self._total_frames = total_frames
        self._fps = fps

    def isOpened(self):
        return self._is_opened

    def get(self, prop_id):
        return self._fps

    def read(self):
        if self._frame_count < self._total_frames:
            self._frame_count += 1
            return True, np.zeros((100, 100, 3), dtype=np.uint8)

        else:
            self._is_opened = False
            return False, None

    def release(self):
        self._is_opened = False

def test_chunk_by_words(chunker):
    """Tests word-based text chunking with overlap."""
    text = " ".join([f"word{i}" for i in range(20)])
    chunks = chunker.chunk_by_words(text)
    assert len(chunks) == 3
    assert chunks[0] == " ".join([f"word{i}" for i in range(10)])
    assert chunks[1] == " ".join([f"word{i}" for i in range(8, 18)])
    assert "word8 word9" in chunks[1]

def test_chunk_by_sentences(chunker, mocker):
    """Tests sentence-based chunking."""
    mock_sent_tokenize = mocker.patch('chunker.nltk.sent_tokenize')
    sentences = ["This is sentence one.", "This is sentence two.", "This is a much longer third sentence that will form its own chunk."]
    mock_sent_tokenize.return_value = sentences
    chunker.tokenizer.encode = lambda text, add_special_tokens: ["token"] * len(text.split())
    chunks = chunker.chunk_by_sentences("Some text.")
    assert len(chunks) == 2
    assert chunks[0] == "This is sentence one. This is sentence two."
    assert chunks[1] == "This is a much longer third sentence that will form its own chunk."

def test_chunk_table(chunker):
    """Tests conversion of a DataFrame to a list of row-dictionaries."""
    df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
    chunks = chunker.chunk_table(df)
    assert len(chunks) == 2
    assert chunks[0] == {'id': 1, 'name': 'Alice'}
    assert chunks[1] == {'id': 2, 'name': 'Bob'}

def test_chunk_image(chunker, mocker):
    """Tests tiling an image into smaller chunks, mocking PIL."""
    mock_img = MagicMock()
    mock_img.size = (300, 200)
    mock_img.crop.return_value = "cropped_tile"
    mock_image_open = mocker.patch('chunker.Image.open', MagicMock())
    mock_image_open.return_value.__enter__.return_value = mock_img
    chunks = chunker.chunk_image('fake/path/to/image.png')

    assert len(chunks) == 6
    assert mock_img.crop.call_count == 6
    mock_img.crop.assert_any_call((0, 0, 128, 128))
    mock_img.crop.assert_any_call((256, 128, 384, 256))

def test_chunk_audio(chunker, mocker):
    """Tests audio splitting, mocking pydub."""
    long_segment = MagicMock()
    long_segment.__len__.return_value = 12000
    short_segment = MagicMock()
    short_segment.__len__.return_value = 3000
    mocker.patch('chunker.AudioSegment.from_file', return_value=MagicMock())
    mock_split = mocker.patch('chunker.silence.split_on_silence')
    mock_split.return_value = [long_segment, short_segment]
    chunks = chunker.chunk_audio('fake/audio.mp3')

    assert len(chunks) == 4
    assert long_segment.__getitem__.call_count == 3

def test_process_video_stream(chunker, mocker):
    """Tests the memory-efficient video chunking generator."""
    mock_vc = mocker.patch('chunker.cv2.VideoCapture')
    mock_vc.return_value = MockCv2Capture(total_frames=55, fps=25)
    video_path = 'fake/video.mp4'
    generator = chunker._process_video_stream(video_path)
    chunks = list(generator)

    assert len(chunks) == 3

    first_chunk = chunks[0]
    assert first_chunk['type'] == 'video'
    assert first_chunk['source'] == video_path
    assert first_chunk['chunk_id'] == 0
    assert len(first_chunk['content']) == 25

    last_chunk = chunks[2]
    assert last_chunk['chunk_id'] == 2
    assert len(last_chunk['content']) == 5

@pytest.mark.parametrize("input_data, file_ext, expected_type", [
    ("test.mp4", ".mp4", "video"),
    ("test.mp3", ".mp3", "audio"),
    ("test.jpg", ".jpg", "image"),
    ("test.txt", ".txt", "text_file"),
    (pd.DataFrame(), None, "table"),
    ("Just a string of text.", None, "text"),
    (np.array([]), None, "image_array"),
    (123, None, "unknown"),
])

def test_process_dispatcher_and_type_detection(chunker, mocker, input_data, file_ext, expected_type):
    """
    End-to-end test of the `process` method's dispatching logic.
    Mocks os and all internal chunking methods to focus on the dispatch logic.
    """
    mocker.patch.object(chunker, '_chunk_text', return_value=['text_chunk'])
    mocker.patch.object(chunker, 'chunk_table', return_value=['table_chunk'])
    mocker.patch.object(chunker, 'chunk_image', return_value=['image_chunk'])
    mocker.patch.object(chunker, 'chunk_audio', return_value=['audio_chunk'])
    mock_video_output = {
        "type": "video", "source": "test.mp4", "chunk_id": 0, "content": "video_chunk"
    }

    mocker.patch.object(chunker, '_process_video_stream', return_value=iter([mock_video_output]))
    mock_isfile = mocker.patch('chunker.os.path.isfile')
    if file_ext:
        mock_isfile.return_value = True
        mocker.patch('chunker.os.path.splitext', return_value=('test', file_ext))
        mocker.patch("builtins.open", mock_open(read_data="file content"))

    else:
        mock_isfile.return_value = False

    results = chunker.process(input_data)

    if expected_type == "video":
        results = list(results)

    assert len(results) > 0
    assert results[0]['type'] == expected_type

    content = results[0]['content']
    if expected_type == "video":
        assert content == "video_chunk"

    elif expected_type in ["text_file", "text"]:
        assert content == "text_chunk"

    elif expected_type == "table":
        assert content == "table_chunk"

    elif expected_type == "image":
        assert content == "image_chunk"

    elif expected_type == "audio":
        assert content == "audio_chunk"

    elif expected_type == "image_array":
        assert np.array_equal(content, input_data)

    elif expected_type == "unknown":
        assert content == input_data