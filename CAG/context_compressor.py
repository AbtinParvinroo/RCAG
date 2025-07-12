from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from transformers import pipeline, AutoTokenizer
import numpy as np
import spacy
import networkx as nx

class ContextCompressor:
    def init(self, embedder=None, summarizer_model="facebook/bart-large-cnn", tokenizer_model="bert-base-uncased", max_tokens=512):
        self.embedder = embedder
        self.summarizer = pipeline("summarization", model=summarizer_model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.max_tokens = max_tokens

    # 1. Summarization
    def summarizer_method(self, chunks):
        text = "\n".join(chunks)
        result = self.summarizer(text, max_length=self.max_tokens, min_length=30, do_sample=False)
        return result[0]["summary_text"]

    # 2. Top-K Selection
    def top_k_selector(self, chunks, query, k=5):
        query_vec = self.embedder.embed(query)
        chunk_vecs = [self.embedder.embed(c) for c in chunks]
        scores = cosine_similarity([query_vec], chunk_vecs)[0]
        top_indices = np.argsort(scores)[-k:][::-1]
        return [chunks[i] for i in top_indices]

    # 3. Clustering + Representative Selection
    def cluster_compressor(self, chunks, n_clusters=5):
        vecs = [self.embedder.embed(c) for c in chunks]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(vecs)
        centers = kmeans.cluster_centers_
        representative_chunks = []
        for center in centers:
            closest_idx = np.argmin([np.linalg.norm(center - v) for v in vecs])
            representative_chunks.append(chunks[closest_idx])
        return representative_chunks

    # 4. Merge & Trim
    def merge_and_trim(self, chunks):
        text = " ".join(chunks)
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_tokens)
        return self.tokenizer.decode(tokens)

    # 5. Query-Aware Compression
    def query_aware_compressor(self, chunks, query, top_n=5):
        query_vec = self.embedder.embed(query)
        sentences = [s for c in chunks for s in c.split(". ") if len(s) > 20]
        sent_vecs = [self.embedder.embed(s) for s in sentences]
        scores = cosine_similarity([query_vec], sent_vecs)[0]
        top_indices = np.argsort(scores)[-top_n:][::-1]
        return ". ".join([sentences[i] for i in top_indices])

    # 6. Cross-Modal Attention Filtering
    def cross_modal_filter(self, multimodal_chunks, query, top_n=3):
        query_vec = self.embedder.embed(query)
        chunk_scores = []
        for chunk in multimodal_chunks:
            chunk_vec = self.embedder.embed(chunk)
            score = cosine_similarity([query_vec], [chunk_vec])[0][0]
            chunk_scores.append((score, chunk))
        chunk_scores.sort(reverse=True)
        return [c for _, c in chunk_scores[:top_n]]

    # 7. Semantic Graph Reduction
    def graph_compressor(self, text, top_n=5):
        nlp = spacy.load("en_core_web_sm")
        graph = nx.Graph()
        doc = nlp(text)
        for token in doc:
            for child in token.children:
                graph.add_edge(token.text, child.text)
        centrality = nx.pagerank(graph)
        sorted_terms = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        keywords = [term for term, _ in sorted_terms[:top_n]]
        summary_sents = [sent.text for sent in doc.sents if any(k in sent.text for k in keywords)]
        return " ".join(summary_sents[:top_n])