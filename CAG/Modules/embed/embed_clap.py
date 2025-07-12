from laion_clap import CLAP_Module

class BaseEmbedder:
    def embed(self, input_data, modality):
        return None

class CLAPEmbedder(BaseEmbedder):
    def __init__(self, device="cpu"):
        self.device = device
        self.model = CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()
        self.model = self.model.to(device)

class SmartEmbedder:
    def __init__(self, device="cpu"):
        self.device = device
        self.models = {
            "clap": CLAPEmbedder(device=device),
        }

    def embed(self, input_data, model_type, modality):
        model = self.models.get(model_type.lower())
        if model is None:
            print(f"Warning: Model '{model_type}' not found.")
            return None
        return model.embed(input_data, modality)