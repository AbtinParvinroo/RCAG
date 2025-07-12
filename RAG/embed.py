from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import torch
from laion_clap import CLAP_Module
from imagebind.models.imagebind_model import ImageBindModel
from imagebind.data import load_and_transform_audio_data, load_and_transform_image_data, load_and_transform_text
from imagebind import data as imagebind_data


class SmartEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model_type = self._detect_model_type()

        if self.model_type == "sentence":
            self.model = SentenceTransformer(model_name)
            self.needs_prompt = any(kw in model_name.lower() for kw in ["e5", "instructor", "gte"])
            self.prompt_style = self._detect_prompt_style()

        elif self.model_type == "clap":
            self.model = CLAP_Module(enable_fusion=False)
            self.model.load_ckpt()

        elif self.model_type == "imagebind":
            self.model = ImageBindModel.from_pretrained("/tmp/imagebind")
            self.model.eval()

    def _detect_model_type(self):
        if "clip" in self.model_name.lower():
            return "clip"

        elif "blip" in self.model_name.lower():
            return "blip"

        elif "clap" in self.model_name.lower():
            return "clap"

        elif "imagebind" in self.model_name.lower():
            return "imagebind"

        else:
            return "sentence"

    def _detect_prompt_style(self):
        if "e5" in self.model_name.lower():
            return {"query": "query: ", "passage": "passage: "}

        elif "instructor" in self.model_name.lower():
            return {
                "query": "Represent the question for retrieving supporting documents:",
                "passage": "Represent the document for retrieval:"
            }

        elif "gte" in self.model_name.lower():
            return {"query": "", "passage": ""}

        else:
            return None

    def embed_query(self, texts):
        if self.model_type != "sentence":
            return "embed_query is only supported for sentence-transformer models"

        if self.needs_prompt and self.prompt_style:
            texts = [self.prompt_style["query"] + text for text in texts]

        return self.model.encode(texts, show_progress_bar=True)

    def embed_passage(self, texts):
        if self.model_type != "sentence":
            return "embed_passage is only supported for sentence-transformer models"

        if self.needs_prompt and self.prompt_style:
            texts = [self.prompt_style["passage"] + text for text in texts]

        return self.model.encode(texts, show_progress_bar=True)

    def clip(self, image, text):
        if self.model_type != "clip":
            return "clip method requires a CLIP-compatible model"

        clip_model = CLIPModel.from_pretrained(self.model_name)
        clip_processor = CLIPProcessor.from_pretrained(self.model_name)
        inputs = clip_processor(text=[text], images=image, return_tensors='pt', padding=True)
        outputs = clip_model(**inputs)
        return outputs.image_embeds, outputs.text_embeds

    def blip(self, image):
        if self.model_type != "blip":
            return "blip method requires a BLIP-compatible model"

        blip_processor = BlipProcessor.from_pretrained(self.model_name)
        blip_model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        inputs = blip_processor(images=image, return_tensors='pt')
        outputs = blip_model.generate(**inputs)
        return blip_processor.batch_decode(outputs, skip_special_tokens=True)[0]

    def clap_audio(self, audio_path):
        if self.model_type != "clap":
            return "clap_audio requires a CLAP-compatible model"

        return self.model.get_audio_embedding_from_filelist([audio_path], use_tensor=True)[0]

    def clap_text(self, texts):
        if self.model_type != "clap":
            return "clap_text requires a CLAP-compatible model"

        return self.model.get_text_embedding(texts, use_tensor=True)[0]

    def imagebind_embed(self, image_path=None, audio_path=None, text=None):
        if self.model_type != "imagebind":
            return "imagebind_embed requires an ImageBind-compatible model"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        inputs = {}
        if image_path:
            inputs[imagebind_data.ModalityType.VISION] = load_and_transform_image_data([image_path], device)
        if audio_path:
            inputs[imagebind_data.ModalityType.AUDIO] = load_and_transform_audio_data([audio_path], device)
        if text:
            inputs[imagebind_data.ModalityType.TEXT] = load_and_transform_text([text], device)

        with torch.no_grad():
            embeddings = self.model(inputs)

        return embeddings