from transformers import pipeline
import requests
import openai
import os

class LLMManager:
    def __init__(self, model_name="mistral", use_api=False, device="cpu", api_key=None):
        self.model_name = model_name.lower()
        self.use_api = use_api
        self.device = device
        self.api_key = api_key
        self.pipeline = None
        self._init_model()

    def _init_model(self):
        if not self.use_api:
            if self.model_name == "mistral":
                self.pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1", device=0 if self.device=="cuda" else -1)

            elif self.model_name == "llama":
                self.pipeline = pipeline("text-generation", model="decapoda-research/llama-7b-hf", device=0 if self.device=="cuda" else -1)

            elif self.model_name == "ollama":
                self.pipeline = None
                print("[Ollama] باید CLI یا API خودش رو جدا اجرا کنی.")

            else:
                return f"مدل لوکال {self.model_name} پشتیبانی نمی‌شود."
        else:
            if self.model_name == "openai":
                if not self.api_key:
                    return "API key برای OpenAI نیاز است."

            elif self.model_name == "hf_api":
                if not self.api_key:
                    return "API key برای Huggingface نیاز است."

            elif self.model_name in ["gpt_neox", "gpt_j"]:

                if not self.api_key:
                    return f"API key برای {self.model_name} نیاز است."

            else:
                return f"API مدل {self.model_name} پشتیبانی نمی‌شود."

    def generate(self, prompt, max_length=200):
        if not self.use_api:
            if self.model_name == "ollama":
                return "[Ollama] لطفا CLI یا API جدا را اجرا کنید."

            outputs = self.pipeline(prompt, max_length=max_length, do_sample=True)
            return outputs[0]['generated_text']

        else:
            if self.model_name == "openai":
                return self._openai_generate(prompt, max_length)

            elif self.model_name == "hf_api":
                return self._hf_api_generate(prompt, max_length)

            elif self.model_name in ["gpt_neox", "gpt_j"]:
                return self._hf_api_generate(prompt, max_length, model=self.model_name)

            else:
                return f"API مدل {self.model_name} پشتیبانی نمی‌شود."

    def _openai_generate(self, prompt, max_length):
        openai.api_key = self.api_key
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=max_length,
            temperature=0.7,
            n=1,
            stop=None
        )

        return response.choices[0].text.strip()

    def _hf_api_generate(self, prompt, max_length, model="gpt2"):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        API_URL = f"https://api-inference.huggingface.co/models/{model}"
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_length}}
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()

            if isinstance(result, list):
                return result[0].get("generated_text", "")

            else:
                return str(result)

        else:
            return f"Huggingface API error: {response.status_code} {response.text}"