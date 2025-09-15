from unittest.mock import MagicMock, patch, ANY
from post_processor import (
    BaseProcessingStep,
    CleaningStep,
    SummarizationStep,
    EntityExtractionStep,
    RelevanceCheckStep,
    LLMValidationStep,
    PostProcessor
)

import numpy as np
import pytest

class TestCleaningStep:
    def test_process_collapses_whitespace(self):
        step = CleaningStep()
        text = "This   has   \n\n multiple   spaces."
        assert step.process(text) == "This has multiple spaces."

    def test_process_fixes_multiple_dots(self):
        step = CleaningStep()
        text = "Wait for it... .. ok."
        assert step.process(text) == "Wait for it. ok."

class TestSummarizationStep:
    @pytest.fixture
    def mock_pipeline(self, mocker):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{'summary_text': 'This is a summary.'}]
        mocker.patch('post_processor.pipeline', return_value=mock_pipe)
        return mock_pipe

    def test_process_summarizes_long_text(self, mock_pipeline):
        step = SummarizationStep(min_length=5)
        long_text = "This is a very long text that absolutely needs to be summarized."
        summary = step.process(long_text)
        mock_pipeline.assert_called_once_with(long_text, max_length=150, min_length=5, do_sample=False)
        assert summary == 'This is a summary.'

    def test_process_skips_short_text(self, mock_pipeline):
        step = SummarizationStep(min_length=10)
        short_text = "This is short."
        result = step.process(short_text)
        mock_pipeline.assert_not_called()
        assert result == short_text

class TestEntityExtractionStep:
    @pytest.fixture
    def mock_spacy(self, mocker):
        mock_ent = MagicMock()
        mock_ent.label_ = "PERSON"
        mock_ent.text = "John Doe"
        mock_doc = MagicMock(ents=[mock_ent])
        mock_nlp = MagicMock(return_value=mock_doc)
        mocker.patch('post_processor.spacy.load', return_value=mock_nlp)
        return mock_nlp

    def test_process_extracts_entities(self, mock_spacy):
        step = EntityExtractionStep()
        entities = step.process("Some text about John Doe.")
        mock_spacy.assert_called_once_with("Some text about John Doe.")
        assert entities == {"PERSON": ["John Doe"]}

    def test_init_raises_if_spacy_unavailable(self, mocker):
        mocker.patch('post_processor.SPACY_AVAILABLE', False)
        with pytest.raises(ImportError):
            EntityExtractionStep()

class TestRelevanceCheckStep:
    @pytest.fixture
    def mock_embedder(self, mocker):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.9], [0.12, 0.88]])
        mocker.patch('post_processor.SentenceTransformer', return_value=mock_model)
        return mock_model

    def test_process_detects_relevant_answer(self, mock_embedder, mocker):
        mocker.patch('post_processor.cosine_similarity', return_value=np.array([[0.95]]))
        step = RelevanceCheckStep(threshold=0.7)
        result = step.process("This is a relevant answer.", context={"query": "A query."})
        assert result["is_relevant"] == True
        assert result["relevance_score"] == 0.95

    def test_process_detects_irrelevant_answer(self, mock_embedder, mocker):
        mocker.patch('post_processor.cosine_similarity', return_value=np.array([[0.45]]))
        step = RelevanceCheckStep(threshold=0.7)
        result = step.process("This is not a relevant answer.", context={"query": "A query."})
        assert result["is_relevant"] == False
        assert result["relevance_score"] == 0.45

    def test_process_raises_on_missing_context(self):
        step = RelevanceCheckStep()
        with pytest.raises(ValueError, match="requires 'query' in the context"):
            step.process("Some text.")

class TestLLMValidationStep:
    @pytest.fixture
    def mock_openai_client(self, mocker):
        mock_choice = MagicMock()
        mock_response = MagicMock(choices=[mock_choice])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mocker.patch('post_processor.OpenAI', return_value=mock_client)
        return mock_client, mock_choice

    def test_process_parses_valid_response(self, mock_openai_client):
        client, choice = mock_openai_client
        choice.message.content = "valid: The answer is correct."
        step = LLMValidationStep(api_key="fake_key")
        result = step.process("An answer.", context={"query": "A query."})
        client.chat.completions.create.assert_called_once()
        assert result["is_valid"] == True
        assert result["feedback"] == "valid: The answer is correct."

    def test_process_parses_invalid_response(self, mock_openai_client):
        client, choice = mock_openai_client
        choice.message.content = "invalid: The answer is wrong."
        step = LLMValidationStep(api_key="fake_key")
        result = step.process("An answer.", context={"query": "A query."})
        assert result["is_valid"] == False
        assert result["feedback"] == "invalid: The answer is wrong."

class TestPostProcessor:
    def test_pipeline_runs_steps_in_order(self, mocker):
        mock_step1 = MagicMock(spec=BaseProcessingStep)
        mock_step2 = MagicMock(spec=BaseProcessingStep)
        mock_step1.__class__ = MagicMock(__name__="StepOne")
        mock_step2.__class__ = MagicMock(__name__="StepTwo")
        processor = PostProcessor(steps=[mock_step1, mock_step2])
        processor.process("initial text")
        mock_step1.process.assert_called_once_with("initial text", None)
        mock_step2.process.assert_called_once_with("initial text", None)

    def test_pipeline_updates_text_between_steps(self, mocker):
        mock_step1 = MagicMock(spec=BaseProcessingStep)
        mock_step1.process.return_value = "cleaned text"
        mock_step2 = MagicMock(spec=BaseProcessingStep)
        mock_step2.process.return_value = {"entities": []}
        mock_step1.__class__ = MagicMock(__name__="Cleaning")
        mock_step2.__class__ = MagicMock(__name__="Extraction")
        processor = PostProcessor(steps=[mock_step1, mock_step2])
        results = processor.process(" dirty text ")
        mock_step1.process.assert_called_with(" dirty text ", ANY)
        mock_step2.process.assert_called_with("cleaned text", ANY)

        assert results["processed_text"] == "cleaned text"
        assert "Extraction" in results

    def test_pipeline_handles_step_error_gracefully(self, mocker, caplog):
        failing_step = MagicMock(spec=BaseProcessingStep)
        failing_step.process.side_effect = ValueError("This specific step failed!")
        failing_step.__class__ = MagicMock(__name__='MyFailingStep')
        working_step = MagicMock(spec=BaseProcessingStep)
        working_step.__class__ = MagicMock(__name__='MyWorkingStep')
        processor = PostProcessor(steps=[failing_step, working_step])
        results = processor.process("initial text")
        working_step.process.assert_called_once()
        assert "Error in pipeline step 'MyFailingStep'" in caplog.text
        assert "This specific step failed!" in caplog.text
        assert "MyFailingStep" in results
        assert results["MyFailingStep"]["error"] == "This specific step failed!"