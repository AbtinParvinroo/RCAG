from sentence_transformers import SentenceTransformer

class BaseEmbedder:
    def embed(self, input_data, modality):
        return None

class TextEmbedder(BaseEmbedder):
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu"):
        self.model = SentenceTransformer(model_name)
        self.device = device

    def embed(self, input_data, modality="text"):
        return self.model.encode(input_data, convert_to_tensor=True)

class SmartEmbedder:
    def __init__(self, device="cpu"):
        self.device = device
        self.models = {
            "text": TextEmbedder(device=device),
        }

    def embed(self, input_data, model_type, modality):
        model = self.models.get(model_type.lower())
        if model is None:
            print(f"Warning: Model '{model_type}' not found.")
            return None
        return model.embed(input_data, modality)