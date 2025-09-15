from transformers import (CLIPProcessor, CLIPModel, BlipProcessor,
from typing import List, Optional, Dict, Any, Union
from sentence_transformers import SentenceTransformer
    BlipForConditionalGeneration)
from functools import lru_cache
from PIL import Image
import numpy as np
import logging
import torch
try:
    from imagebind.models.imagebind_model import ImageBindModel
    from imagebind import data as imagebind_data
    IMAGEBIND_AVAILABLE = True

except ImportError:
    IMAGEBIND_AVAILABLE = False

try:
    from laion_clap import CLAP_Module
    CLAP_AVAILABLE = True

except ImportError:
    CLAP_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartEmbedder:
    """
    A powerful, multi-modal embedder that lazily loads models to optimize
    resource usage in RAG pipelines.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = self._detect_model_type()
        self._model = None
        self._processor = None
        self._prompt_style = None
        self._needs_prompt = None
        logger.info(f"SmartEmbedder initialized for model '{model_name}'. Model will be loaded on first use.")

    @property
    def model(self):
        """Property to lazily load the model."""
        if self._model is None:
            self._load_model_and_processor()

        return self._model

    @property
    def processor(self):
        """Property to lazily load the processor."""
        if self._processor is None:
            _ = self.model

        return self._processor

    @property
    def needs_prompt(self) -> bool:
        """Property to lazily determine if the model needs a prompt."""
        if self._needs_prompt is None:
            _ = self.model

        return self._needs_prompt

    @property
    def prompt_style(self) -> Optional[Dict[str, str]]:
        """Property to lazily get the prompt style."""
        if self._prompt_style is None:
            _ = self.model

        return self._prompt_style

    def _load_model_and_processor(self):
        """
        Loads the appropriate model and processor based on model_type.
        """
        logger.info(f"Lazily loading model '{self.model_name}' to device '{self.device}'...")
        if self.model_type == "sentence":
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._needs_prompt = any(kw in self.model_name.lower() for kw in ["e5", "instructor", "gte"])
            self._prompt_style = self._detect_prompt_style() if self._needs_prompt else None

        elif self.model_type == "clip":
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)

        elif self.model_type == "blip":
            self._model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            self._processor = BlipProcessor.from_pretrained(self.model_name)

        elif self.model_type == "clap":
            if not CLAP_AVAILABLE: raise ImportError("laion_clap is not installed.")
            self._model = CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
            self._model.load_ckpt()
            self._model.to(self.device)

        elif self.model_type == "imagebind":
            if not IMAGEBIND_AVAILABLE: raise ImportError("imagebind-client is not installed.")
            self._model = ImageBindModel.from_pretrained("imagebind-huge").to(self.device)
            self._model.eval()

        else:
            raise ValueError(f"Unknown or unsupported model type for '{self.model_name}'")
        logger.info("Model loaded successfully.")

    def embed(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Smartly routes a data chunk to the correct embedding method.
        """
        ctype = chunk.get("type")
        content = chunk.get("content")
        source = chunk.get("source")
        vec = None

        try:
            if ctype == "text":
                if self.model_type == "sentence": vec = self.embed_passages((content,))[0]
                elif self.model_type == "clip": _, vec = self.embed_clip(texts=[content]); vec = vec[0]
                elif self.model_type == "clap": res = self.embed_clap(texts=[content]); vec = res.get("text")[0]
                elif self.model_type == "imagebind": res = self.embed_imagebind(texts=[content]); vec = res.get("text")[0]
                else: logger.warning(f"Model '{self.model_name}' cannot embed text chunks.")

            elif ctype in ["image", "image_array"]:
                image = Image.fromarray(content) if isinstance(content, np.ndarray) else content
                if self.model_type == "clip": vec, _ = self.embed_clip(images=[image]); vec = vec[0]

                elif self.model_type == "imagebind":
                    if not source: raise ValueError("ImageBind requires a file path ('source') for images.")

                    res = self.embed_imagebind(image_paths=[source]); vec = res.get("vision")[0]

                else: logger.warning(f"Model '{self.model_name}' cannot embed image chunks.")

            elif ctype == "audio":
                if not source: raise ValueError(f"{self.model_type} requires a file path ('source') for audio.")

                if self.model_type == "clap": res = self.embed_clap(audio_paths=[source]); vec = res.get("audio")[0]

                elif self.model_type == "imagebind": res = self.embed_imagebind(audio_paths=[source]); vec = res.get("audio")[0]

                else: logger.warning(f"Model '{self.model_name}' cannot embed audio chunks.")

            elif ctype == "table":
                text_repr = ", ".join(f"{k}: {v}" for k, v in content.items())
                if self.model_type == "sentence":
                    vec = self.embed_passages((text_repr,))[0]

                else:
                    logger.warning(f"Model '{self.model_name}' cannot directly embed tables. Converted to text.")
                    if self.model_type == "clip": _, vec = self.embed_clip(texts=[text_repr]); vec = vec[0]

                    elif self.model_type == "clap": res = self.embed_clap(texts=[text_repr]); vec = res.get("text")[0]

                    elif self.model_type == "imagebind": res = self.embed_imagebind(texts=[text_repr]); vec = res.get("text")[0]

            else: logger.warning(f"Unsupported chunk type '{ctype}' for embedding.")

        except Exception as e:
            logger.error(f"Embedding failed for chunk {chunk.get('chunk_id')} with source '{source}': {e}", exc_info=True)
            vec = None

        return {**chunk, "embedding": vec.tolist() if isinstance(vec, np.ndarray) else vec}

    def _detect_model_type(self) -> str:
        name_lower = self.model_name.lower()
        if "clap" in name_lower: return "clap"

        if "clip" in name_lower: return "clip"

        if "blip" in name_lower: return "blip"

        if "imagebind" in name_lower: return "imagebind"

        return "sentence"

    def _detect_prompt_style(self) -> Optional[Dict[str, str]]:
        name_lower = self.model_name.lower()
        if "e5" in name_lower: return {"query": "query: ", "passage": "passage: "}

        if "instructor" in name_lower: return {"query": "Represent the question for retrieving supporting documents: ", "passage": "Represent the document for retrieval: "}

        if "gte" in name_lower: return {"query": "", "passage": ""}

        return None

    @lru_cache(maxsize=512)
    def embed_queries(self, texts: Union[tuple, List[str]]):
        if self.model_type != "sentence": raise TypeError(f"embed_queries is only for 'sentence' models.")

        prefixed_texts = list(texts)
        if self.needs_prompt and self.prompt_style:
            prefixed_texts = [self.prompt_style["query"] + text for text in texts]

        return self.model.encode(prefixed_texts, convert_to_numpy=True)

    @lru_cache(maxsize=512)
    def embed_passages(self, texts: Union[tuple, List[str]]):
        if self.model_type != "sentence": raise TypeError(f"embed_passages is only for 'sentence' models.")

        prefixed_texts = list(texts)
        if self.needs_prompt and self.prompt_style:
            prefixed_texts = [self.prompt_style["passage"] + text for text in texts]

        return self.model.encode(prefixed_texts, convert_to_numpy=True)

    def embed_clip(self, texts: Optional[List[str]] = None, images: Optional[List[Image]] = None, batch_size: int = 8):
        if self.model_type != "clip": raise TypeError(f"embed_clip is only for 'clip' models.")

        if not texts and not images: return None, None

        num_items = len(texts) if texts else len(images)
        image_embeds, text_embeds = [], []
        for i in range(0, num_items, batch_size):
            batch_texts = texts[i:i+batch_size] if texts else None
            batch_images = images[i:i+batch_size] if images else None
            inputs = self.processor(text=batch_texts, images=batch_images, return_tensors='pt', padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            if batch_images: image_embeds.append(outputs.image_embeds.cpu().numpy())

            if batch_texts: text_embeds.append(outputs.text_embeds.cpu().numpy())

        final_image_embeds = np.vstack(image_embeds) if image_embeds else None
        final_text_embeds = np.vstack(text_embeds) if text_embeds else None
        return final_image_embeds, final_text_embeds

    def embed_clap(self, texts: Optional[List[str]] = None, audio_paths: Optional[List[str]] = None):
        if self.model_type != "clap": raise TypeError(f"embed_clap is only for 'clap' models.")
        with torch.no_grad():
            embeddings = {}
            if texts: embeddings['text'] = self.model.get_text_embedding(texts, use_tensor=False)

            if audio_paths: embeddings['audio'] = self.model.get_audio_embedding_from_filelist(x=audio_paths, use_tensor=False)

        return embeddings

    def embed_imagebind(self, texts: Optional[List[str]] = None, image_paths: Optional[List[str]] = None, audio_paths: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        if self.model_type != "imagebind": raise TypeError(f"embed_imagebind is only for 'imagebind' models.")

        inputs = {}
        if image_paths: inputs[imagebind_data.ModalityType.VISION] = imagebind_data.load_and_transform_image(image_paths, self.device)

        if audio_paths: inputs[imagebind_data.ModalityType.AUDIO] = imagebind_data.load_and_transform_audio(audio_paths, self.device)

        if texts: inputs[imagebind_data.ModalityType.TEXT] = imagebind_data.load_and_transform_text(texts, self.device)

        if not inputs: return {}

        with torch.no_grad():
            embeddings = self.model(inputs)

        return {modality.name.lower(): emb.cpu().numpy() for modality, emb in embeddings.items()}