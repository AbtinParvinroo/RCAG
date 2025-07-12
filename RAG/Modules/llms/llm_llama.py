from transformers import pipeline

class BaseLLM:
    def __init__(self, device="cpu", api_key=None):
        self.device = device
        self.api_key = api_key

    def generate(self, prompt: str, max_length: int = 200) -> str:
        raise NotImplementedError("Subclasses must implement generate method.")

class LlamaLLM(BaseLLM):
    def __init__(self, device="cpu", **kwargs):
        super().__init__(device=device)
        self.pipeline = pipeline(
            "text-generation",
            model="decapoda-research/llama-7b-hf",
            device=0 if device == "cuda" else -1,
        )

    def generate(self, prompt: str, max_length: int = 200) -> str:
        outputs = self.pipeline(prompt, max_length=max_length, do_sample=True)
        return outputs[0]["generated_text"]

class LLMManager:
    def __init__(self, model_name="mistral", use_api=False, device="cpu", api_key=None):
        self.model_name = model_name.lower()
        self.use_api = use_api
        self.device = device
        self.api_key = api_key
        self.llm = self._create_llm()

    def _create_llm(self) -> BaseLLM:
        if not self.use_api:
            if self.model_name == "llama":
                return LlamaLLM(device=self.device)
            else:
                raise ValueError(f"مدل لوکال {self.model_name} پشتیبانی نمی‌شود.")
        else:
            raise ValueError(f"API مدل {self.model_name} پشتیبانی نمی‌شود.")

    def generate(self, prompt: str, max_length: int = 200) -> str:
        return self.llm.generate(prompt, max_length)