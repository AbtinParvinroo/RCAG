class BaseCompressionStrategy:
    def compress(self, chunks, query):
        raise NotImplementedError("Subclasses must implement compress method.")

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

class ContextCompressor:
    def __init__(self, strategy: BaseCompressionStrategy):
        self.strategy = strategy

    def compress(self, chunks, query=None):
        return self.strategy.compress(chunks, query=query)
