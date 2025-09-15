from typing import List, Dict, Any, Tuple, Generator, Union
from pydub import AudioSegment, silence
from transformers import AutoTokenizer
from PIL import Image
import pandas as pd
import numpy as np
import logging
import nltk
import cv2
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)

class RAGChunker:
    """
    A versatile, multi-modal chunker designed to prepare various data types for RAG and CAG pipelines.
    This class is memory-optimized for handling large video files by using generators.
    """
    def __init__(
        self,
        chunk_size: int = 128,
        overlap: int = 32,
        model_name: str = "bert-base-uncased",
        image_tile_size: Tuple[int, int] = (256, 256),
        audio_silence_thresh: int = -40,
        video_chunk_duration: int = 5,
        max_audio_chunk_duration: int = 10_000
    ):
        """
        Initializes the RAGChunker with specified settings.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.image_tile_size = image_tile_size
        self.audio_silence_thresh = audio_silence_thresh
        self.video_chunk_duration = video_chunk_duration
        self.max_audio_chunk_duration = max_audio_chunk_duration

    def process(
        self,
        data: Any,
        text_chunk_method: str = 'token',
        video_in_memory: bool = False
    ) -> Union[List[Dict[str, Any]], Generator[Dict[str, Any], None, None]]:
        """
        Processes and chunks the input data. Returns a generator for video files
        by default for memory efficiency.
        """
        dtype = self._detect_type(data)

        if dtype == "video" and not video_in_memory:
            return self._process_video_stream(data)

        chunks = self._process_data_in_memory(data, dtype, text_chunk_method)
        source_name = data if isinstance(data, str) and os.path.isfile(data) else "inline_data"

        return [
            {
                "type": dtype,
                "source": source_name,
                "chunk_id": i,
                "content": chunk
            }

            for i, chunk in enumerate(chunks)
        ]

    def _process_data_in_memory(self, data: Any, dtype: str, text_chunk_method: str) -> List[Any]:
        """Processes all data types that can be safely loaded into memory."""
        if dtype == "text_file":
            try:
                with open(data, "r", encoding="utf-8") as f:
                    content = f.read()

                return self._chunk_text(content, text_chunk_method)

            except Exception as e:
                logger.error(f"Failed to read text file {data}: {e}", exc_info=True)
                return []

        elif dtype == "text":
            return self._chunk_text(data, text_chunk_method)

        elif dtype == "table":
            return self.chunk_table(data)

        elif dtype == "image":
            return self.chunk_image(data)

        elif dtype == "image_array":
            return [data]

        elif dtype == "audio":
            return self.chunk_audio(data)

        else:
            logger.warning(f"Data with unknown type '{dtype}' received. Treating as a single chunk.")
            return [data]

    def _process_video_stream(self, video_path: str) -> Generator[Dict[str, Any], None, None]:
        """A generator that yields video chunk dictionaries one by one."""
        chunk_id = 0
        for video_chunk in self._chunk_video_generator(video_path):
            yield {
                "type": "video",
                "source": video_path,
                "chunk_id": chunk_id,
                "content": video_chunk
            }

            chunk_id += 1

    def _chunk_video_generator(self, video_path: str) -> Generator[List[np.ndarray], None, None]:
        """A memory-efficient generator that yields chunks of video frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames_per_chunk = int(fps * self.video_chunk_duration)
        try:
            while cap.isOpened():
                current_chunk_frames = []
                for _ in range(frames_per_chunk):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    current_chunk_frames.append(frame)

                if current_chunk_frames:
                    yield current_chunk_frames

                else:
                    break

        finally:
            cap.release()

    def _detect_type(self, data: Any) -> str:
        """Detects the data type of the input."""
        if isinstance(data, str):
            if os.path.isfile(data):
                ext = os.path.splitext(data)[1].lower()
                if ext in ['.mp4', '.avi', '.mov']: return "video"
                if ext in ['.wav', '.mp3']: return "audio"
                if ext in ['.jpg', '.jpeg', '.png']: return "image"
                if ext in ['.txt']: return "text_file"
                return "unknown_file"

            else:
                return "text"

        elif isinstance(data, pd.DataFrame): return "table"
        elif isinstance(data, np.ndarray): return "image_array"
        else: return "unknown"

    def _chunk_text(self, text: str, method: str) -> List[str]:
        """Internal dispatcher for text chunking methods."""
        if method == 'token': return self.chunk_by_tokens(text)
        if method == 'sentence': return self.chunk_by_sentences(text)
        if method == 'word': return self.chunk_by_words(text)
        logger.warning(f"Unknown text chunking method '{method}', falling back to 'token'.")
        return self.chunk_by_tokens(text)

    def chunk_by_tokens(self, text: str) -> List[str]:
        """Chunks text by a specified number of tokens with overlap."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        return [self.tokenizer.decode(tokens[i:i + self.chunk_size]) for i in range(0, len(tokens), step)]

    def chunk_by_sentences(self, text: str) -> List[str]:
        """Chunks text by sentences, trying to respect chunk_size."""
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        token_count = 0
        for sent in sentences:
            sent_tokens = len(self.tokenizer.encode(sent, add_special_tokens=False))
            if token_count + sent_tokens > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                token_count = sent_tokens

            else:
                current_chunk.append(sent)
                token_count += sent_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def chunk_by_words(self, text: str) -> List[str]:
        """Chunks text by a specified number of words with overlap."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [" ".join(words)]

        step = self.chunk_size - self.overlap
        return [" ".join(words[i:i + self.chunk_size]) for i in range(0, len(words), step)]

    def chunk_table(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Chunks a DataFrame by converting each row to a dictionary."""
        return [row.to_dict() for _, row in df.iterrows()]

    def chunk_image(self, image_path: str) -> List[np.ndarray]:
        """Chunks an image into smaller tiles."""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                chunks = []
                for y in range(0, height, self.image_tile_size[1]):
                    for x in range(0, width, self.image_tile_size[0]):
                        box = (x, y, x + self.image_tile_size[0], y + self.image_tile_size[1])
                        tile = img.crop(box)
                        chunks.append(np.array(tile))

                return chunks
        except Exception as e:
            logger.error(f"Image chunking failed for {image_path}: {e}", exc_info=True)
            return []

    def chunk_audio(self, audio_path: str) -> List[AudioSegment]:
        """Chunks an audio file based on silence and max duration."""
        try:
            audio = AudioSegment.from_file(audio_path)
            raw_chunks = silence.split_on_silence(
                audio,
                min_silence_len=500,
                silence_thresh=self.audio_silence_thresh,
                keep_silence=250
            )
            final_chunks = []
            for ch in raw_chunks:
                if len(ch) > self.max_audio_chunk_duration:
                    for i in range(0, len(ch), self.max_audio_chunk_duration):
                        final_chunks.append(ch[i:i + self.max_audio_chunk_duration])

                else:
                    final_chunks.append(ch)

            return final_chunks

        except Exception as e:
            logger.error(f"Audio chunking failed for {audio_path}: {e}", exc_info=True)
            return []