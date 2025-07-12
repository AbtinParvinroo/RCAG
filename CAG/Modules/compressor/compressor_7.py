import networkx as nx
import spacy

class BaseCompressionStrategy:
    def compress(self, chunks, query=None):
        raise NotImplementedError("Subclasses must implement compress method.")

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
    
