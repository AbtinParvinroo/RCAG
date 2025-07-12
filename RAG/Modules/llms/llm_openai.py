import openai

class BaseLLM:
    def __init__(self, device="cpu", api_key=None):
        self.device = device
        self.api_key = api_key

    def generate(self, prompt: str, max_length: int = 200) -> str:
        raise NotImplementedError("Subclasses must implement generate method.")

class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key=api_key)
        openai.api_key = api_key

    def generate(self, prompt: str, max_length: int = 200) -> str:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=max_length,
            temperature=0.7,
            n=1,
            stop=None,
        )
        return response.choices[0].text.strip()

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
                return OpenAILLM(api_key=self.api_key)
            elif self.model_name == "hf_api":
                if not self.api_key:
                    raise ValueError("API key برای Huggingface نیاز است.")
            elif self.model_name in ["gpt_neox", "gpt_j"]:
                if not self.api_key:
                    raise ValueError(f"API key برای {self.model_name} نیاز است.")
            else:
                raise ValueError(f"API مدل {self.model_name} پشتیبانی نمی‌شود.")

    def generate(self, prompt: str, max_length: int = 200) -> str:
        return self.llm.generate(prompt, max_length)