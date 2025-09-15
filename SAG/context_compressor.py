import logging
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer
from typing import List, Union, Optional
from embedders import SmartEmbedder
from sklearn.cluster import KMeans
import numpy as np
try:
    import spacy
    import networkx as nx
    SPACY_AVAILABLE = True

except ImportError:
    SPACY_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseCompressionStrategy:
    def compress(self, chunks: List[str], query: Optional[str] = None) -> Union[str, List[str]]:
        raise NotImplementedError("زیرکلاس‌ها باید متد 'compress' را پیاده‌سازی کنند.")

class SummarizationCompressor(BaseCompressionStrategy):
    def __init__(self, model_name: str = "facebook/bart-large-cnn", max_tokens: int = 512):
        self.pipeline = pipeline("summarization", model=model_name)
        self.tokenizer = self.pipeline.tokenizer
        self.max_tokens = max_tokens

    def compress(self, chunks: List[str], query: Optional[str] = None) -> str:
        if not chunks: return ""
        text_to_summarize = "\\n".join(chunks)
        summary = self.pipeline(text_to_summarize, max_length=self.max_tokens, min_length=30, do_sample=False)
        return summary[0]["summary_text"]

class TopKCompressor(BaseCompressionStrategy):
    def __init__(self, embedder, k: int = 5):
        self.embedder = embedder
        self.k = k

    def compress(self, chunks: List[str], query: Optional[str] = None) -> List[str]:
        if not query: raise ValueError("TopKCompressor به کوئری نیاز دارد.")

        if not chunks: return []

        query_vec = self.embedder.embed_queries([query])
        chunk_vecs = self.embedder.embed_passages(chunks)
        scores = cosine_similarity(query_vec, chunk_vecs)[0]
        top_indices = np.argsort(scores)[-self.k:][::-1]
        return [chunks[i] for i in top_indices]

class ClusterCompressor(BaseCompressionStrategy):
    def __init__(self, embedder, n_clusters: int = 5):
        self.embedder = embedder
        self.n_clusters = n_clusters

    def compress(self, chunks: List[str], query: Optional[str] = None) -> List[str]:
        if not chunks: return []

        num_clusters = min(self.n_clusters, len(chunks))
        if num_clusters < 1: return []

        chunk_vecs = np.array(self.embedder.embed_passages(chunks))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(chunk_vecs)
        representative_chunks = []
        for i in range(num_clusters):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            if len(cluster_indices) == 0: continue

            center = kmeans.cluster_centers_[i]
            cluster_vectors = chunk_vecs[cluster_indices]
            closest_idx_in_cluster = np.argmin(np.linalg.norm(cluster_vectors - center, axis=1))
            original_idx = cluster_indices[closest_idx_in_cluster]
            representative_chunks.append(chunks[original_idx])

        return representative_chunks

class MergeAndTrimCompressor(BaseCompressionStrategy):
    def __init__(self, tokenizer_model: str = "bert-base-uncased", max_tokens: int = 2048):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.max_tokens = max_tokens

    def compress(self, chunks: List[str], query: Optional[str] = None) -> str:
        if not chunks: return ""
        full_text = " ".join(chunks)
        tokens = self.tokenizer.encode(full_text, truncation=True, max_length=self.max_tokens)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

class GraphCompressor(BaseCompressionStrategy):
    def __init__(self, top_n: int = 5):
        if not SPACY_AVAILABLE:
            raise ImportError("لطفاً 'spacy' و 'networkx' را نصب کنید. همچنین 'python -m spacy download en_core_web_sm' را اجرا کنید.")

        self.nlp = spacy.load("en_core_web_sm")
        self.top_n = top_n

    def compress(self, chunks: List[str], query: Optional[str] = None) -> str:
        if not chunks: return ""
        text = " ".join(chunks)
        doc = self.nlp(text)
        graph = nx.Graph()

        for token in doc:
            if not token.is_stop and not token.is_punct:
                for child in token.children:
                    if not child.is_stop and not child.is_punct:
                        graph.add_edge(token.lemma_, child.lemma_)

        if not graph.nodes: return " ".join([sent.text for sent in doc.sents][:self.top_n])

        centrality = nx.pagerank(graph)
        sorted_terms = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        keywords = {term for term, _ in sorted_terms[:self.top_n * 2]}
        summary_sents = []
        for sent in doc.sents:
            if any(k in sent.text.lower() for k in keywords):
                summary_sents.append(sent.text)
                if len(summary_sents) >= self.top_n: break

        return " ".join(summary_sents)

class ContextCompressor:
    def __init__(self, strategy: BaseCompressionStrategy):
        self.strategy = strategy
        logger.info(f"ContextCompressor با استراتژی {strategy.__class__.__name__} راه‌اندازی شد.")

    def compress(self, chunks: List[str], query: Optional[str] = None) -> Union[str, List[str]]:
        return self.strategy.compress(chunks, query=query)