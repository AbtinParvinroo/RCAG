from typing import List, Tuple
import numpy as np

class Retriever:
    def __init__(self, embedder, vector_store, reranker=None, metadata_store=None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.metadata_store = metadata_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict = None,
        hybrid_keywords: List[str] = None
    ) -> List[Tuple[str, float]]:

        q_vec = self.embedder.embed_query([query])[0]
        vector_results = self.vector_store.search(q_vec, top_k=top_k * 3)
        if filters and self.metadata_store:
            vector_results = self._apply_filters(vector_results, filters)

        if hybrid_keywords:
            keyword_results = self._keyword_match(hybrid_keywords)
            vector_results = self._merge_results(vector_results, keyword_results)

        if self.reranker:
            reranked = self._rerank(query, vector_results)
            return reranked[:top_k]

        return vector_results[:top_k]

    def _apply_filters(self, results, filters):
        filtered = []
        for doc_id, score in results:
            meta = self.metadata_store.get(doc_id, {})

            if all(meta.get(k) == v for k, v in filters.items()):
                filtered.append((doc_id, score))

        return filtered

    def _keyword_match(self, keywords: List[str]) -> List[Tuple[str, float]]:
        matches = []
        for doc_id, meta in self.metadata_store.items():
            text = meta.get("text", "").lower()

            if any(kw.lower() in text for kw in keywords):
                matches.append((doc_id, 0.5))

        return matches

    def _merge_results(self, vector_res, keyword_res):
        merged = {doc_id: score for doc_id, score in vector_res}
        for doc_id, score in keyword_res:
            if doc_id in merged:
                merged[doc_id] += score

            else:
                merged[doc_id] = score

        return sorted(merged.items(), key=lambda x: x[1], reverse=True)

    def _rerank(self, query: str, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        docs = [self.metadata_store[cid]["text"] for cid, _ in candidates if cid in self.metadata_store]
        scores = self.reranker.predict(query, docs)
        return sorted(zip([cid for cid, _ in candidates], scores), key=lambda x: x[1], reverse=True)