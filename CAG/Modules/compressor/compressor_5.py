class BaseCompressionStrategy:
    def compress(self, chunks, query=None):
        raise NotImplementedError("Subclasses must implement compress method.")

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

class ContextCompressor:
    def __init__(self, strategy: BaseCompressionStrategy):
        self.strategy = strategy

    def compress(self, chunks, query=None):
        return self.strategy.compress(chunks, query=query)
