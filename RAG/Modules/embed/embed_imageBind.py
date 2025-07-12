from imagebind.models.imagebind_model import ImageBindModel
from imagebind.data import load_and_transform_audio_data, load_and_transform_image_data, load_and_transform_text
import torch

class BaseEmbedder:
    def embed(self, input_data, modality):
        return None

class ImageBindEmbedder(BaseEmbedder):
    def __init__(self, device="cpu"):
        self.device = device
        self.model = ImageBindModel(device=device)
        self.model.eval()

    def embed(self, input_data, modality):
        modality = modality.lower()
        if modality == "image":
            input_data = load_and_transform_image_data([input_data], self.device)
        elif modality == "audio":
            input_data = load_and_transform_audio_data([input_data], self.device)
        elif modality == "text":
            input_data = load_and_transform_text([input_data], self.device)
        else:
            return None
        with torch.no_grad():
            embeddings = self.model(input_data)
        return embeddings[ImageBindModel.modality_keys[modality]]

class SmartEmbedder:
    def __init__(self, device="cpu"):
        self.device = device
        self.models = {
            "imagebind": ImageBindEmbedder(device=device)
        }

    def embed(self, input_data, model_type, modality):
        model = self.models.get(model_type.lower())
        if model is None:
            print(f"Warning: Model '{model_type}' not found.")
            return None
        return model.embed(input_data, modality)