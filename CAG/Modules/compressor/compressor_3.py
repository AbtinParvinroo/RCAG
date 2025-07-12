class BaseCompressionStrategy:
    def compress(self, chunks, query=None):
        raise NotImplementedError("Subclasses must implement compress method.")

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

class ContextCompressor:
    def __init__(self, strategy: BaseCompressionStrategy):
        self.strategy = strategy

    def compress(self, chunks, query=None):
        return self.strategy.compress(chunks, query=query)