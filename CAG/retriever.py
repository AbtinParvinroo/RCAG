from typing import List, Dict, Any, Optional
import logging
import asyncio

class Retriever:
    """
    An advanced retrieval pipeline with both synchronous and native asynchronous
    execution paths.
    """
    def __init__(self, embedder: Any, vector_store: Any, reranker: Optional[Any] = None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.logger = logging.getLogger(self.__class__.__name__)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Executes the full retrieval pipeline synchronously."""
        self.logger.info(f"Starting synchronous retrieval for query: '{query[:80]}...'")
        try:
            query_embedding = self.embedder.embed_queries([query])[0]
            initial_k = top_k * 3 if self.reranker and top_k > 1 else top_k
            initial_candidates = self.vector_store.search(
                query_vector=query_embedding, k=initial_k, filters=filters
            )

            if not initial_candidates:
                self.logger.warning("No candidates found in vector search.")
                return []

            final_results = self._rerank(query, initial_candidates) if self.reranker else initial_candidates
            return final_results[:top_k]

        except Exception as e:
            self.logger.error(f"Error in synchronous retrieval pipeline: {e}", exc_info=True)
            return []

    async def retrieve_async(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Executes the full retrieval pipeline using a native asynchronous approach."""
        self.logger.info(f"Starting asynchronous retrieval for query: '{query[:80]}...'")
        loop = asyncio.get_running_loop()
        try:
            query_embedding = await loop.run_in_executor(
                None, self.embedder.embed_queries, [query]
            )

            query_embedding = query_embedding[0]

            initial_k = top_k * 3 if self.reranker and top_k > 1 else top_k
            if not hasattr(self.vector_store, 'search_async'):
                self.logger.warning("vector_store has no 'search_async' method. Falling back to executor.")
                initial_candidates = await loop.run_in_executor(
                    None, self.vector_store.search, query_embedding, initial_k, filters
                )

            else:
                    initial_candidates = await self.vector_store.search_async(
                    query_vector=query_embedding, k=initial_k, filters=filters
                )

            if not initial_candidates:
                self.logger.warning("No candidates found in async vector search.")
                return []

            if self.reranker:
                final_results = await self._rerank_async(query, initial_candidates)

            else:
                final_results = initial_candidates

            return final_results[:top_k]
        except Exception as e:
            self.logger.error(f"Error in asynchronous retrieval pipeline: {e}", exc_info=True)
            return []

    def _rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Synchronous reranking helper."""
        try:
            docs_to_rerank = [cand.get("content", "") for cand in candidates]
            if not any(docs_to_rerank): return candidates

            rerank_scores = self.reranker.predict(query, docs_to_rerank)

            for cand, score in zip(candidates, rerank_scores):
                cand['rerank_score'] = score

            return sorted(candidates, key=lambda x: x.get('rerank_score', 0.0), reverse=True)

        except Exception as e:
            self.logger.error(f"Error during reranking: {e}", exc_info=True)
            return candidates

    async def _rerank_async(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Asynchronous reranking helper."""
        try:
            docs_to_rerank = [cand.get("content", "") for cand in candidates]
            if not any(docs_to_rerank):
                self.logger.warning("No content for async reranking. Skipping.")
                return candidates

            loop = asyncio.get_running_loop()
            if hasattr(self.reranker, 'predict_async'):
                rerank_scores = await self.reranker.predict_async(query, docs_to_rerank)

            else:
                rerank_scores = await loop.run_in_executor(
                    None, self.reranker.predict, query, docs_to_rerank
                )

            for cand, score in zip(candidates, rerank_scores):
                cand['rerank_score'] = score

            return sorted(candidates, key=lambda x: x.get('rerank_score', 0.0), reverse=True)

        except Exception as e:
            self.logger.error(f"Error during async reranking: {e}", exc_info=True)
            return candidates

def build_retriever_with_vectorstore(
    embedder: Any,
    vector_backend: str,
    vector_dim: int,
    vector_config: dict,
    reranker: Optional[Any] = None
) -> Retriever:
    class MockVectorStore:
        def __init__(self, **kwargs): pass
        def search(self, qv, k, filters): return []
        async def search_async(self, qv, k, filters):
            # In a real scenario, this would be a non-blocking DB call
            await asyncio.sleep(0.01)
            return []

    vector_store = MockVectorStore(backend=vector_backend, dim=vector_dim, **vector_config)
    return Retriever(embedder=embedder, vector_store=vector_store, reranker=reranker)