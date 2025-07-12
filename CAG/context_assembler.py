from imagebind.data import load_and_transform_audio_data, load_and_transform_image_data, load_and_transform_text
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from imagebind.models.imagebind_model import ImageBindModel
from sentence_transformers import SentenceTransformer
from imagebind import data as imagebind_data
from pydub import AudioSegment, silence
from nltk.tokenize import sent_tokenize
from laion_clap import CLAP_Module
from PIL import Image
import pandas as pd
import numpy as np
import requests
import openai
import torch
import nltk
import cv2
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from transformers import pipeline, AutoTokenizer
import spacy
import networkx as nx
import re
import nltk
import torch

class ContextAssembler:
    def __init__(self, mode="text", sort_key=None, priority=None):
        self.mode = mode
        self.sort_key = sort_key
        self.priority = priority or []

    def assemble(self, chunks):
        if not chunks:
            return ""
        if self.mode == "text":
            return self._assemble_text(chunks)
        elif self.mode == "table":
            return self._assemble_table(chunks)
        elif self.mode == "image":
            return self._assemble_images(chunks)
        elif self.mode == "audio":
            return self._assemble_audio(chunks)
        elif self.mode == "mixed":
            return self._assemble_mixed(chunks)
        else:
            raise ValueError("Unsupported mode")

    def _assemble_text(self, chunks):
        if self.sort_key:
            chunks = sorted(chunks, key=self.sort_key)
        return "\n".join(str(c) for c in chunks)

    def _assemble_table(self, chunks):
        return pd.DataFrame(chunks)

    def _assemble_images(self, chunks):
        return sorted(chunks, key=self.sort_key) if self.sort_key else chunks

    def _assemble_audio(self, chunks):
        return sorted(chunks, key=self.sort_key) if self.sort_key else chunks

    def _assemble_mixed(self, chunks):
        grouped = {}
        for c in chunks:
            dtype = self._detect_type(c)
            grouped.setdefault(dtype, []).append(c)
        ordered = []
        for p in self.priority:
            if p in grouped:
                ordered.extend(grouped[p])
        return ordered

    def _detect_type(self, chunk):
        if isinstance(chunk, str):
            return "text"
        elif isinstance(chunk, dict):
            return "table"
        elif "PIL" in str(type(chunk)):
            return "image"
        elif "AudioSegment" in str(type(chunk)):
            return "audio"
        elif isinstance(chunk, np.ndarray):
            return "image_array"
        else:
            return "unknown"

    def merge_with_titles(self, structured_chunks, title_key="title", content_key="content"):
        merged = []
        for item in structured_chunks:
            title = item.get(title_key, "Untitled")
            content = item.get(content_key, "")
            merged.append(f"### {title}\n{content}")
        return "\n\n".join(merged)

    def build_prompt(self, chunks, query, instruction=None):
        """
        ساخت پرامپت استاندارد یا ادپتیو بر اساس نوع داده و query
        """
        assembled = self.assemble(chunks)
        prompt = ""

        if instruction:
            prompt += f"[Instruction]\n{instruction.strip()}\n\n"

        # 🔀 Adaptive formatting
        if isinstance(assembled, pd.DataFrame):
            prompt += f"[Context - Table]\n{assembled.to_string(index=False)}\n\n"
        elif isinstance(assembled, list):
            # تصویر، صوت و غیره
            prompt += "[Context - List]\n" + "\n".join(str(x) for x in assembled) + "\n\n"
        else:
            prompt += f"[Context]\n{assembled.strip()}\n\n"

        prompt += f"[Question]\n{query.strip()}\n\n[Answer]"
        return prompt