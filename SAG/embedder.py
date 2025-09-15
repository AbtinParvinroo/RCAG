from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Any, Union, Tuple
from functools import lru_cache
import numpy as np
import logging
import torch
try:
    from PIL.Image import Image
    PIL_AVAILABLE = True

except ImportError:
    PIL_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SmartEmbedder:
    """
    A powerful, multi-modal embedder that lazily loads models to optimize
    resource usage. Primarily designed for sentence-transformer models in a SAG context.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = self._detect_model_type()
        self._model = None
        logger.info(f"SmartEmbedder initialized for model '{model_name}'. Model will be loaded on first use.")

    @property
    def model(self):
        """Property to lazily load the SentenceTransformer model."""
        if self._model is None:
            logger.info(f"Lazily loading model '{self.model_name}' to device '{self.device}'...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Model loaded successfully.")

        return self._model

    def _detect_model_type(self) -> str:
        """Detects the model type. Simplified for SAG, assuming sentence-transformers."""
        return "sentence"

    @lru_cache(maxsize=512)
    def embed_queries(self, texts: Tuple[str, ...]) -> np.ndarray:
        """Encodes a tuple of query strings into embeddings."""
        return self.model.encode(texts, convert_to_numpy=True)

    @lru_cache(maxsize=512)
    def embed_passages(self, texts: Tuple[str, ...]) -> np.ndarray:
        """Encodes a tuple of passage or document strings into embeddings."""
        return self.model.encode(texts, convert_to_numpy=True)