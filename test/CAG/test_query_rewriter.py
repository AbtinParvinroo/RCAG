from query_rewriter import QueryRewriter
import pytest

@pytest.fixture
def default_rewriter():
    """Provides a QueryRewriter instance with default rules."""
    return QueryRewriter()

@pytest.fixture
def custom_rewriter():
    """Provides a QueryRewriter instance with custom, specific rules for testing."""
    custom_rules = {
        "k8s": "kubernetes",
        "ci/cd": "continuous integration continuous deployment"
    }

    return QueryRewriter(expansion_rules=custom_rules)

class TestQueryRewriter:
    def test_default_abbreviation_expansion(self, default_rewriter):
        """Tests if default abbreviations like 'llm' and 'rag' are expanded."""
        query = "What is a RAG system?"
        expected = "What is a retrieval-augmented generation system?"
        assert default_rewriter.rewrite(query) == expected

    def test_default_typo_correction(self, default_rewriter):
        """Tests if a default common typo like 'vektor' is corrected."""
        query = "How to create a vektor index?"
        expected = "How to create a vector index?"
        assert default_rewriter.rewrite(query) == expected

    def test_custom_rules_are_applied(self, custom_rewriter):
        """Tests that the rewriter uses the custom rules it was initialized with."""
        query = "Deploying on k8s with a ci/cd pipeline."
        expected = "Deploying on kubernetes with a continuous integration continuous deployment pipeline."
        assert custom_rewriter.rewrite(query) == expected

    def test_rewrite_is_case_insensitive(self, default_rewriter):
        """Tests that rewriting works regardless of the case of the term in the query."""
        query = "Explain rag and LLM."
        expected = "Explain retrieval-augmented generation and large language model."
        assert default_rewriter.rewrite(query) == expected

    def test_no_match_returns_original_query(self, default_rewriter):
        """Tests that if no rules match, the original query is returned unchanged."""
        query = "A query about something else."
        assert default_rewriter.rewrite(query) == query

    def test_partial_word_is_not_matched(self, default_rewriter):
        """Tests that the rewriter only matches whole words (e.g., 'rag' not 'drags')."""
        query = "He drags the heavy luggage."

        assert default_rewriter.rewrite(query) == query

    @pytest.mark.parametrize("invalid_input", [None, "", 123, []])
    def test_invalid_input_is_handled_gracefully(self, default_rewriter, invalid_input):
        """Tests that non-string or empty inputs are handled without errors."""
        assert default_rewriter.rewrite(invalid_input) == invalid_input