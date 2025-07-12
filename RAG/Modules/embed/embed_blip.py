from transformers import BlipProcessor, BlipForConditionalGeneration

class BaseEmbedder:
    def embed(self, input_data, modality):
        return None

class BLIPEembedder(BaseEmbedder):
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device="cpu"):
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    def embed(self, input_data, modality="image"):
        inputs = self.processor(images=input_data, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        return outputs

class SmartEmbedder:
    def __init__(self, device="cpu"):
        self.device = device
        self.models = {
            "blip": BLIPEembedder(device=device),
        }

    def embed(self, input_data, model_type, modality):
        model = self.models.get(model_type.lower())
        if model is None:
            print(f"Warning: Model '{model_type}' not found.")
            return None
        return model.embed(input_data, modality)