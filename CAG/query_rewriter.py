from typing import Dict, Optional
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryRewriter:
    """
    A component to rewrite user queries for better retrieval performance.
    It can expand abbreviations, correct common typos, and add synonyms.
    This version is simplified for robust, dependency-free testing.
    """
    def __init__(self, expansion_rules: Optional[Dict[str, str]] = None):
        """
        Initializes the rewriter with a set of expansion and correction rules.
        Args:
            expansion_rules (Optional[Dict[str, str]]): A dictionary where keys are
                terms to be replaced (like typos or abbreviations) and values are
                their replacements.
        """
        if expansion_rules is None:
            self.expansion_rules = {
                "llm": "large language model",
                "rag": "retrieval-augmented generation",
                "nlp": "natural language processing",
                "vektor": "vector",
            }

        else:
            self.expansion_rules = expansion_rules

        logger.info(f"QueryRewriter initialized with {len(self.expansion_rules)} rules.")

    def rewrite(self, query: str) -> str:
        """
        Rewrites a query by applying the defined expansion and correction rules.
        Args:
            query (str): The original user query.
        Returns:
            str: The rewritten query.
        """
        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided. Returning original query.")
            return query

        rewritten_query = query
        for term, expansion in self.expansion_rules.items():
            rewritten_query = re.sub(r'\b' + re.escape(term) + r'\b', expansion, rewritten_query, flags=re.IGNORECASE)

        if rewritten_query != query:
            logger.info(f"Query rewritten: '{query}' -> '{rewritten_query}'")

        return rewritten_query