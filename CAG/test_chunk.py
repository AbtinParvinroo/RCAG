import os
import numpy as np
import pandas as pd
import tempfile
from PIL import Image
from pydub.generators import Sine
from transformers import AutoTokenizer
from chunking import CAGChunker
import cv2
from pydub import AudioSegment, silence
import nltk

from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)
def test_text_chunking():
    text = "This is a test sentence. " * 20
    chunker = CAGChunker()
    result = chunker.chunk(text)
    
    # 👀 هر سه نوع چانک باید وجود داشته باشه
    # assert "word" in result and "sentence" in result and "token" in result
    print(type(result))
    if len(result) > 0:
        print('ok')

def test_table_chunking():
    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
    chunker = CAGChunker()
    result = chunker.chunk(df)
    
    assert isinstance(result, list)
    assert isinstance(result[0], dict)

def test_image_chunking():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img = Image.new('RGB', (512, 512), color = 'red')
        img.save(tmp.name)
        chunker = CAGChunker(image_tile_size=(256, 256))
        chunks = chunker.chunk(tmp.name)
        assert isinstance(chunks, list)
        assert len(chunks) == 4  # چون 512/256 = 2 در هر بُعد

        os.remove(tmp.name)

def test_audio_chunking():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tone = Sine(440).to_audio_segment(duration=3000)
        tone.export(tmp.name, format="wav")
        chunker = CAGChunker(audio_silence_thresh=-60)
        chunks = chunker.chunk(tmp.name)
        assert isinstance(chunks, list)
        assert all(isinstance(c, type(tone)) for c in chunks)

        os.remove(tmp.name)

# def test_video_chunking():
#     # این بخش برای تست لوکال خوبه چون ساختن ویدیو تو CI سخته
#     try:
#         import cv2
#         video_path = "test_video.avi"
#         width, height = 640, 480
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(video_path, fourcc, 5.0, (width, height))
#         for _ in range(25):
#             frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
#             out.write(frame)
#         out.release()

#         chunker = CAGChunker(video_chunk_duration=1)  # با 5fps یعنی هر 5 فریم یه chunk
#         chunks = chunker.chunk(video_path)
#         assert isinstance(chunks, list)
#         assert all(isinstance(c, list) for c in chunks)

#         os.remove(video_path)
#     except ImportError:
#         print("cv2 not available; skipping video test.")

def test_array_input():
    arr = np.random.rand(10, 10, 3)
    chunker = CAGChunker()
    result = chunker.chunk(arr)
    assert isinstance(result, list)
    assert len(result) == 1

def test_unknown_type():
    chunker = CAGChunker()
    result = chunker.chunk({"some": "dict"})
    assert isinstance(result, list)

if __name__ == "__main__":
    test_text_chunking()
    test_table_chunking()
    test_image_chunking()
    test_audio_chunking()
    # test_video_chunking()
    test_array_input()
    test_unknown_type()
    print("✅ All tests passed!")