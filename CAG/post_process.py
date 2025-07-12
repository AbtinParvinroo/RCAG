import re
import nltk
import spacy
import torch
import numpy as np
import openai
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt', quiet=True)

class PostProcessor:
    def init(self,
                summarizer_model: str = "facebook/bart-large-cnn",
                embedder_model: str = "all-MiniLM-L6-v2",
                device: str = "cpu",
                openai_api_key: str = None):
        
        self.device = device
        self.summarizer = pipeline("summarization",
                                    model=summarizer_model,
                                    device=0 if device == "cuda" and torch.cuda.is_available() else -1)
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        self.embedder = SentenceTransformer(embedder_model)

        if openai_api_key:
            openai.api_key = openai_api_key

    # 1. Clean text
    def clean(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\. *\.+', '. ', text)
        return text.strip()

    # 2. Summarize text
    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        if len(text.split()) < min_length:
            return text
        summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']

    # 3. Extract entities
    def extract_entities(self, text: str) -> dict:
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            entities.setdefault(ent.label_, []).append(ent.text)
        return entities

    # 4. Sentence splitter
    def split_sentences(self, text: str) -> list:
        return sent_tokenize(text)

    # 5. Check semantic relevance
    def is_relevant(self, query: str, answer: str, threshold: float = 0.5) -> bool:
        vec_query = self.embedder.encode(query, convert_to_numpy=True)
        vec_answer = self.embedder.encode(answer, convert_to_numpy=True)
        similarity = cosine_similarity([vec_query], [vec_answer])[0][0]
        return similarity >= threshold

    def relevance_score(self, query: str, answer: str) -> float:
        vec_query = self.embedder.encode(query, convert_to_numpy=True)
        vec_answer = self.embedder.encode(answer, convert_to_numpy=True)
        return cosine_similarity([vec_query], [vec_answer])[0][0]

    # 6. Final validation with LLM (OpenAI GPT)
    def llm_validate_answer(self, query: str, answer: str, model: str = "gpt-4", temperature=0.0) -> dict:
        if not openai.api_key:
            raise ValueError("OpenAI API key not set! Provide openai_api_key in constructor.")
        
        prompt = f"""
        You are an expert assistant. Check if the following answer correctly and relevantly addresses the query.
        
        Query: "{query}"
        
        Answer: "{answer}"
        
        Respond with:
        - 'valid' if the answer is correct and relevant,
        - 'invalid' otherwise,
        and a brief explanation why.
        """
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=150,
        )
        text = response['choices'][0]['message']['content'].strip().lower()
        valid = 'valid' in text
        return {"valid": valid, "feedback": text}