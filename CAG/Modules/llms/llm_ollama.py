class BaseLLM:
    def __init__(self, device="cpu", api_key=None):
        self.device = device
        self.api_key = api_key

    def generate(self, prompt: str, max_length: int = 200) -> str:
        raise NotImplementedError("Subclasses must implement generate method.")

class OllamaLLM(BaseLLM):
    def __init__(self, **kwargs):
        super().__init__()
        print("[Ollama] باید CLI یا API خودش رو جدا اجرا کنی.")
        self.pipeline = None

    def generate(self, prompt: str, max_length: int = 200) -> str:
        return "[Ollama] لطفا CLI یا API جدا را اجرا کنید."

class LLMManager:
    def __init__(self, model_name="mistral", use_api=False, device="cpu", api_key=None):
        self.model_name = model_name.lower()
        self.use_api = use_api
        self.device = device
        self.api_key = api_key
        self.llm = self._create_llm()

    def _create_llm(self) -> BaseLLM:
        if not self.use_api:
            if self.model_name == "ollama":
                return OllamaLLM()
            else:
                raise ValueError(f"مدل لوکال {self.model_name} پشتیبانی نمی‌شود.")
        else:
                raise ValueError(f"مدل {self.model_name} پشتیبانی نمی‌شود.")

    def generate(self, prompt: str, max_length: int = 200) -> str:
        return self.llm.generate(prompt, max_length)