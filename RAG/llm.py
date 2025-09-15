from typing import Dict, Any, Iterator, List, Optional, Union
from requests.adapters import HTTPAdapter, Retry
from threading import Thread
import requests
import logging
import json
import os

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    TRANSFORMERS_AVAILABLE = True

except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True

except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMManager:
    """
    An advanced, unified manager for various Large Language Models (LLMs),
    supporting local execution, API calls, dynamic parameter control, streaming,
    and connection to external retrievers.
    """
    def __init__(self, engine: str, model_name: str, api_key: Optional[str] = None,
            retriever: Optional[Any] = None, **kwargs):
        self.engine = engine.lower()
        self.model_name = model_name
        self.api_key = api_key or os.getenv(f"{engine.upper()}_API_KEY")
        self.model_kwargs = kwargs
        self.retriever = retriever
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.client: Optional[Any] = None
        self.session: Optional[requests.Session] = None
        self._validate_config()
        self._initialize_client()

    def _validate_config(self):
        supported_engines = ['local', 'openai', 'huggingface_api', 'ollama']
        if self.engine not in supported_engines:
            raise ValueError(f"Unsupported engine: '{self.engine}'. Supported: {supported_engines}")

        if self.engine in ["openai", "huggingface_api"] and not self.api_key:
            raise ValueError(f"API key is required for the '{self.engine}' engine.")

        if self.engine == 'local' and not TRANSFORMERS_AVAILABLE:
            raise ImportError("Please install 'torch' and 'transformers' to use local models.")

        if self.engine == 'openai' and not OPENAI_AVAILABLE:
            raise ImportError("Please install 'openai>=1.0' to use OpenAI models.")

    def _initialize_client(self):
        """Initializes the API client or session for network-based engines."""
        if self.engine == "openai":
            self.client = OpenAI(api_key=self.api_key)

        elif self.engine in ["huggingface_api", "ollama"]:
            self.session = requests.Session()
            retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
            self.session.mount('http://', HTTPAdapter(max_retries=retries))
            self.session.mount('https://', HTTPAdapter(max_retries=retries))
            if self.engine == "huggingface_api":
                self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _load_local_model(self):
        if self.model is None:
            logger.info(f"Loading local model '{self.model_name}'... This may take a while.")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                **self.model_kwargs
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("Local model loaded successfully.")

    def _build_rag_prompt(self, prompt: str) -> str:
        """Constructs a prompt with retrieved context."""
        if not self.retriever:
            return prompt

        logger.info(f"Retrieving context for prompt: '{prompt[:80]}...'")
        context_docs = self.retriever.retrieve(prompt, top_k=3)
        if not context_docs:
            logger.warning("Retriever found no relevant documents.")
            return prompt

        context_texts = [doc.get("content", "") for doc in context_docs]
        context_block = "\n\n---\n\n".join(context_texts).strip()

        return (
            "You are a helpful assistant. Use the following context to answer the user's question. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION:\n{prompt}"
        )

    def generate(self, prompt: str, stream: bool = False, use_rag: bool = True,
            **generation_kwargs) -> Union[str, Iterator[str]]:
        if use_rag:
            prompt = self._build_rag_prompt(prompt)

        engine_map = {
            'local': self._generate_local,
            'openai': self._generate_openai,
            'huggingface_api': self._generate_hf_api,
            'ollama': self._generate_ollama,
        }

        return engine_map[self.engine](prompt, stream, **generation_kwargs)

    def _generate_local(self, prompt: str, stream: bool, **kwargs) -> Union[str, Iterator[str]]:
        self._load_local_model()
        kwargs.setdefault("max_new_tokens", 512)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        if not stream:
            outputs = self.model.generate(**inputs, **kwargs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        else:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(inputs, streamer=streamer, **kwargs)
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            return streamer

    def _generate_openai(self, prompt: str, stream: bool, **kwargs) -> Union[str, Iterator[str]]:
        kwargs.setdefault("max_tokens", 512)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=stream,
            **kwargs
        )

        if not stream:
            return response.choices[0].message.content.strip()

        else:
            def stream_generator():
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content

            return stream_generator()

    def _generate_hf_api(self, prompt: str, stream: bool, **kwargs) -> Union[str, Iterator[str]]:
        api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        kwargs.setdefault("max_new_tokens", 512)
        payload = {"inputs": prompt, "parameters": kwargs, "stream": stream}
        response = self.session.post(api_url, json=payload, stream=stream, timeout=60)
        response.raise_for_status()
        if not stream:
            return response.json()[0]['generated_text']

        else:
            def stream_generator():
                for byte_payload in response.iter_lines():
                    if byte_payload and byte_payload.startswith(b'data:'):
                        data = json.loads(byte_payload[5:])
                        yield data['token']['text']

            return stream_generator()

    def _generate_ollama(self, prompt: str, stream: bool, **kwargs) -> Union[str, Iterator[str]]:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        payload = {"model": self.model_name, "prompt": prompt, "stream": stream, "options": kwargs}
        response = self.session.post(ollama_url, json=payload, stream=stream)
        response.raise_for_status()

        if not stream:
            return response.json()['response']

        else:
            def stream_generator():
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        yield chunk['response']
                        if chunk.get('done'):
                            break

            return stream_generator()