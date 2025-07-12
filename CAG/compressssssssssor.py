from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from transformers import pipeline, AutoTokenizer
import numpy as np
import spacy
import networkx as nx

class BaseCompressionStrategy:
    def compress(self, chunks, query=None):
        raise NotImplementedError("Subclasses must implement compress method.")


class SummarizationCompressor(BaseCompressionStrategy):
    def __init__(self, model_name="facebook/bart-large-cnn", max_tokens=512):
        self.pipeline = pipeline("summarization", model=model_name)
        self.max_tokens = max_tokens

    def compress(self, chunks, query=None):
        text = "\n".join(chunks)
        result = self.pipeline(text, max_length=self.max_tokens, min_length=30, do_sample=False)
        return result[0]["summary_text"]


class TopKCompressor(BaseCompressionStrategy):
    def __init__(self, embedder, k=5):
        self.embedder = embedder
        self.k = k

    def compress(self, chunks, query):
        query_vec = self.embedder.embed(query)
        chunk_vecs = [self.embedder.embed(c) for c in chunks]
        scores = cosine_similarity([query_vec], chunk_vecs)[0]
        top_indices = np.argsort(scores)[-self.k:][::-1]
        return [chunks[i] for i in top_indices]


class ClusterCompressor(BaseCompressionStrategy):
    def __init__(self, embedder, n_clusters=5):
        self.embedder = embedder
        self.n_clusters = n_clusters

    def compress(self, chunks, query=None):
        vecs = [self.embedder.embed(c) for c in chunks]
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42).fit(vecs)
        centers = kmeans.cluster_centers_
        representative_chunks = []
        for center in centers:
            closest_idx = np.argmin([np.linalg.norm(center - v) for v in vecs])
            representative_chunks.append(chunks[closest_idx])
        return representative_chunks


class MergeAndTrimCompressor(BaseCompressionStrategy):
    def __init__(self, tokenizer_model="bert-base-uncased", max_tokens=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.max_tokens = max_tokens

    def compress(self, chunks, query=None):
        text = " ".join(chunks)
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_tokens)
        return self.tokenizer.decode(tokens)


class QueryAwareCompressor(BaseCompressionStrategy):
    def __init__(self, embedder, top_n=5):
        self.embedder = embedder
        self.top_n = top_n

    def compress(self, chunks, query):
        query_vec = self.embedder.embed(query)
        sentences = [s for c in chunks for s in c.split(". ") if len(s) > 20]
        sent_vecs = [self.embedder.embed(s) for s in sentences]
        scores = cosine_similarity([query_vec], sent_vecs)[0]
        top_indices = np.argsort(scores)[-self.top_n:][::-1]
        return ". ".join([sentences[i] for i in top_indices])


class CrossModalCompressor(BaseCompressionStrategy):
    def __init__(self, embedder, top_n=3):
        self.embedder = embedder
        self.top_n = top_n

    def compress(self, chunks, query):
        query_vec = self.embedder.embed(query)
        scored = []
        for chunk in chunks:
            chunk_vec = self.embedder.embed(chunk)
            score = cosine_similarity([query_vec], [chunk_vec])[0][0]
            scored.append((score, chunk))
        scored.sort(reverse=True)
        return [c for _, c in scored[:self.top_n]]


class GraphCompressor(BaseCompressionStrategy):
    def __init__(self, top_n=5):
        self.nlp = spacy.load("en_core_web_sm")
        self.top_n = top_n

    def compress(self, chunks, query=None):
        text = " ".join(chunks)
        doc = self.nlp(text)
        graph = nx.Graph()
        for token in doc:
            for child in token.children:
                graph.add_edge(token.text, child.text)
        centrality = nx.pagerank(graph)
        sorted_terms = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        keywords = [term for term, _ in sorted_terms[:self.top_n]]
        summary_sents = [sent.text for sent in doc.sents if any(k in sent.text for k in keywords)]
        return " ".join(summary_sents[:self.top_n])


class ContextCompressor:
    def __init__(self, strategy: BaseCompressionStrategy):
        self.strategy = strategy

    def compress(self, chunks, query=None):
        return self.strategy.compress(chunks, query=query)
