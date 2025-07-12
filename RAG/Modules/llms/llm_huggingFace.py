import requests

class BaseLLM:
    def __init__(self, device="cpu", api_key=None):
        self.device = device
        self.api_key = api_key

    def generate(self, prompt: str, max_length: int = 200) -> str:
        raise NotImplementedError("Subclasses must implement generate method.")

class HuggingfaceLLM(BaseLLM):
    def __init__(self, api_key: str, model: str = "gpt2", **kwargs):
        super().__init__(api_key=api_key)
        self.model = model
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def generate(self, prompt: str, max_length: int = 200) -> str:
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_length}}
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list):
                return result[0].get("generated_text", "")
            else:
                return str(result)
        else:
            return f"Huggingface API error: {response.status_code} {response.text}"

class LLMManager:
    def __init__(self, model_name="mistral", use_api=False, device="cpu", api_key=None):
        self.model_name = model_name.lower()
        self.use_api = use_api
        self.device = device
        self.api_key = api_key
        self.llm = self._create_llm()

    def _create_llm(self) -> BaseLLM:
        if not self.use_api:
            raise ValueError(f"مدل لوکال {self.model_name} پشتیبانی نمی‌شود.")
        else:
            if self.model_name == "openai":
                if not self.api_key:
                    raise ValueError("API key برای OpenAI نیاز است.")
            elif self.model_name == "hf_api":
                if not self.api_key:
                    raise ValueError("API key برای Huggingface نیاز است.")
                return HuggingfaceLLM(api_key=self.api_key)
            elif self.model_name in ["gpt_neox", "gpt_j"]:
                if not self.api_key:
                    raise ValueError(f"API key برای {self.model_name} نیاز است.")
                return HuggingfaceLLM(api_key=self.api_key, model=self.model_name)
            else:
                raise ValueError(f"API مدل {self.model_name} پشتیبانی نمی‌شود.")

    def generate(self, prompt: str, max_length: int = 200) -> str:
        return self.llm.generate(prompt, max_length)