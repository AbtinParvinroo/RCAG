import nltk
import numpy as np
import pandas as pd
from PIL import Image
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment, silence
import cv2
from transformers import AutoTokenizer

nltk.download('punkt', quiet=True)
class CAGChunker:
    def init(
        self,
        chunk_size=128,
        overlap=32,
        model_name="bert-base-uncased",
        image_tile_size=(256, 256),
        audio_silence_thresh=-40,
        video_chunk_duration=5
    ):

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.image_tile_size = image_tile_size
        self.audio_silence_thresh = audio_silence_thresh
        self.video_chunk_duration = video_chunk_duration

    def detect_type(self, data):
        if isinstance(data, str):
            if data.endswith(('.mp4', '.avi')):
                return "video"

            elif data.endswith(('.wav', '.mp3')):
                return "audio"

            elif data.endswith(('.jpg', '.png')):
                return "image"

            else:
                return "text"

        elif isinstance(data, pd.DataFrame):
            return "table"

        elif isinstance(data, np.ndarray):
            return "image_array"

        else:
            return "unknown"

    def chunk(self, data):
        dtype = self.detect_type(data)
        if dtype == "text":
            return {
                "word": self.chunk_by_words(data),
                "sentence": self.chunk_by_sentences(data),
                "token": self.chunk_by_tokens(data),
            }

        elif dtype == "table":
            return self.chunk_table(data)

        elif dtype == "image":
            return self.chunk_image(data)

        elif dtype == "image_array":
            return [data]

        elif dtype == "audio":
            return self.chunk_audio(data)

        elif dtype == "video":
            return self.chunk_video(data)

        else:
            return [data]

    def chunk_by_words(self, text):
        words = text.split()
        step = self.chunk_size - self.overlap
        return [
            " ".join(words[i:i + self.chunk_size])
            for i in range(0, len(words), step)
        ]

    def chunk_by_sentences(self, text):
        sentences = sent_tokenize(text)
        step = self.chunk_size - self.overlap
        return [
            " ".join(sentences[i:i + self.chunk_size])
            for i in range(0, len(sentences), step)
        ]

    def chunk_by_tokens(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        step = self.chunk_size - self.overlap
        chunks = [
            tokens[i:i + self.chunk_size]
            for i in range(0, len(tokens), step)
        ]

        return [self.tokenizer.decode(chunk) for chunk in chunks]

    def chunk_table(self, df):
        return [row.to_dict() for _, row in df.iterrows()]

    def chunk_image(self, image_path):
        img = Image.open(image_path)
        width, height = img.size
        chunks = []
        for y in range(0, height, self.image_tile_size[1]):
            for x in range(0, width, self.image_tile_size[0]):
                box = (x, y, x + self.image_tile_size[0], y + self.image_tile_size[1])
                chunk = img.crop(box)
                chunks.append(chunk)
        return chunks

    def chunk_audio(self, audio_path):
        audio = AudioSegment.from_file(audio_path)
        return silence.split_on_silence(
            audio,
            min_silence_len=500,
            silence_thresh=self.audio_silence_thresh
        )

    def chunk_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_per_chunk = int(fps * self.video_chunk_duration)
        chunks = []
        chunk = []
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            chunk.append(frame)
            count += 1

            if count % frames_per_chunk == 0:
                chunks.append(chunk)
                chunk = []

        if chunk:
            chunks.append(chunk)

        cap.release()
        return chunks