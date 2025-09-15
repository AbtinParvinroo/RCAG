from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from transformers import pipeline
import numpy as np
import logging
import re
try:
    import spacy
    SPACY_AVAILABLE = True

except ImportError:
    SPACY_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True

except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseProcessingStep:
    """An abstract base class for a single step in the post-processing pipeline."""
    def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> Any:
        raise NotImplementedError("Each processing step must implement the 'process' method.")

class CleaningStep(BaseProcessingStep):
    """A step to perform basic text cleaning using regular expressions."""
    def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\s*\.){2,}', '.', text)
        text = re.sub(r'\.(?=[a-zA-Z])', '. ', text)
        return text.strip()

class SummarizationStep(BaseProcessingStep):
    """A step to summarize long texts."""
    def __init__(self, model_name: str = "facebook/bart-large-cnn", max_length: int = 150, min_length: int = 40):
        self.pipeline = pipeline("summarization", model=model_name)
        self.max_length = max_length
        self.min_length = min_length
        logger.info(f"SummarizationStep initialized with model: {model_name}")

    def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        if len(text.split()) < self.min_length:
            return text

        summary = self.pipeline(text, max_length=self.max_length, min_length=self.min_length, do_sample=False)
        return summary[0]['summary_text']

class EntityExtractionStep(BaseProcessingStep):
    """A step to extract named entities using SpaCy."""
    def __init__(self, model: str = "en_core_web_sm"):
        if not SPACY_AVAILABLE:
            raise ImportError("Please install 'spacy' and download the model: python -m spacy download en_core_web_sm")

        self.nlp = spacy.load(model)
        logger.info(f"EntityExtractionStep initialized with SpaCy model: {model}")

    def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        doc = self.nlp(text)
        entities: Dict[str, List[str]] = {}
        for ent in doc.ents:
            entities.setdefault(ent.label_, []).append(ent.text)

        return entities

class RelevanceCheckStep(BaseProcessingStep):
    """A step to check the semantic relevance between a query and an answer."""
    def __init__(self, embedder_model: str = "all-MiniLM-L6-v2", threshold: float = 0.5):
        self.embedder = SentenceTransformer(embedder_model)
        self.threshold = threshold
        logger.info(f"RelevanceCheckStep initialized with embedder: {embedder_model}")

    def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not context or "query" not in context:
            raise ValueError("RelevanceCheckStep requires 'query' in the context dictionary.")

        query = context["query"]
        answer = text
        vec_query, vec_answer = self.embedder.encode([query, answer], convert_to_numpy=True)
        similarity = cosine_similarity([vec_query], [vec_answer])[0][0]
        return {
            "is_relevant": similarity >= self.threshold,
            "relevance_score": float(similarity)
        }

class LLMValidationStep(BaseProcessingStep):
    """A step that uses a powerful LLM to validate or critique an answer."""
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        if not OPENAI_AVAILABLE:
            raise ImportError("Please install 'openai>=1.0' to use this step.")

        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("OpenAI API key not found. Pass it or set OPENAI_API_KEY environment variable.")

        logger.info(f"LLMValidationStep initialized with model: {model}")

    def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not context or "query" not in context:
            raise ValueError("LLMValidationStep requires 'query' in the context dictionary.")

        query = context["query"]
        answer = text
        prompt = (
            "You are an expert evaluator. Check if the following answer correctly and relevantly addresses the query.\n\n"
            f'Query: "{query}"\n\n'
            f'Answer: "{answer}"\n\n'
            "Respond with only 'valid' or 'invalid', followed by a colon and a brief explanation. For example: 'valid: The answer directly addresses the user's question.'"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100
        )

        feedback = response.choices[0].message.content.strip()
        is_valid = feedback.lower().startswith("valid")
        return {"is_valid": is_valid, "feedback": feedback}

class PostProcessor:
    """A flexible post-processing pipeline that applies a series of steps to an LLM's output."""
    def __init__(self, steps: List[BaseProcessingStep]):
        self.pipeline = steps
        logger.info(f"PostProcessor initialized with {len(steps)} steps.")

    def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        results = {"original_text": text}
        current_text = text
        for step in self.pipeline:
            step_name = step.__class__.__name__
            try:
                step_result = step.process(current_text, context)
                if isinstance(step_result, str):
                    current_text = step_result
                    results["processed_text"] = current_text

                else:
                    results[step_name] = step_result

            except Exception as e:
                logger.error(f"Error in pipeline step '{step_name}': {e}", exc_info=True)
                results[step_name] = {"error": str(e)}
        if "processed_text" not in results:
            results["processed_text"] = text

        return results