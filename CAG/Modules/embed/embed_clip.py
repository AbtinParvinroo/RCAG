from transformers import CLIPProcessor, CLIPModel

class BaseEmbedder:
    def embed(self, input_data, modality):
        return None

class CLIPEmbedder(BaseEmbedder):
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed(self, input_data, modality):
        inputs = self.processor(text=input_data if modality == "text" else None,
                                images=input_data if modality == "image" else None,
                                return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        return outputs.pooler_output

class SmartEmbedder:
    def __init__(self, device="cpu"):
        self.device = device
        self.models = {
            "clip": CLIPEmbedder(device=device),
        }

    def embed(self, input_data, model_type, modality):
        model = self.models.get(model_type.lower())
        if model is None:
            print(f"Warning: Model '{model_type}' not found.")
            return None
        return model.embed(input_data, modality)
