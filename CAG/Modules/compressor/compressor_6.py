from sklearn.metrics.pairwise import cosine_similarity
from typing import List
class BaseCompressionStrategy:
    def compress(self, chunks, query):
        raise NotImplementedError("Subclasses must implement compress method.")

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

class ContextCompressor:
    def __init__(self, strategy: BaseCompressionStrategy):
        self.strategy = strategy

    def compress(self, chunks, query=None):
        return self.strategy.compress(chunks, query=query)

class ContextCompressor:
    def __init__(self, strategy: BaseCompressionStrategy):
        self.strategy = strategy

    def compress(self, chunks, query=None):
        return self.strategy.compress(chunks, query=query)
