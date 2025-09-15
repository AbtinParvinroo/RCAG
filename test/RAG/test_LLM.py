from unittest.mock import MagicMock, patch, ANY
from collections.abc import Iterator
from llm import LLMManager
import pytest
import json

@pytest.fixture
def mock_retriever():
    """Provides a mock retriever that returns a predefined context."""
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {"content": "The sky is blue."},
        {"content": "Water is wet."}
    ]

    return retriever

@pytest.fixture(autouse=True)
def mock_all_dependencies(mocker):
    """
    A comprehensive fixture to mock all heavy external libraries and network calls
    before each test runs.
    """
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "Local response"
    mock_streamer = MagicMock()
    mocker.patch('llm.AutoModelForCausalLM.from_pretrained', return_value=mock_model)
    mocker.patch('llm.AutoTokenizer.from_pretrained', return_value=mock_tokenizer)
    mocker.patch('llm.TextIteratorStreamer', return_value=mock_streamer)
    mocker.patch('llm.torch', MagicMock())
    mocker.patch('llm.Thread', MagicMock())
    mock_openai_choice = MagicMock()
    mock_openai_choice.message.content = "OpenAI response"
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = MagicMock(choices=[mock_openai_choice])
    mocker.patch('llm.OpenAI', return_value=mock_openai_client)
    mock_session = MagicMock()
    mocker.patch('requests.Session', return_value=mock_session)
    return {
        "local_model": mock_model,
        "local_tokenizer": mock_tokenizer,
        "local_streamer": mock_streamer,
        "openai_client": mock_openai_client,
        "requests_session": mock_session,
    }

def test_initialization_success():
    llm = LLMManager(engine="openai", model_name="gpt-4", api_key="test_key")
    assert llm.engine == "openai"
    assert llm.client is not None

def test_initialization_unsupported_engine_fails():
    with pytest.raises(ValueError, match="Unsupported engine"):
        LLMManager(engine="unsupported_engine", model_name="test")

def test_initialization_missing_api_key_fails():
    with pytest.raises(ValueError, match="API key is required"):
        LLMManager(engine="openai", model_name="gpt-4", api_key=None)

def test_initialization_missing_local_dependency_fails(mocker):
    mocker.patch('llm.TRANSFORMERS_AVAILABLE', False)
    with pytest.raises(ImportError, match="install 'torch' and 'transformers'"):
        LLMManager(engine="local", model_name="gpt2")

def test_build_rag_prompt_with_retriever(mock_retriever):
    llm = LLMManager(engine="local", model_name="test", retriever=mock_retriever)
    final_prompt = llm._build_rag_prompt("What color is the sky?")

    mock_retriever.retrieve.assert_called_once_with("What color is the sky?", top_k=3)
    assert "CONTEXT:\nThe sky is blue.\n\n---\n\nWater is wet." in final_prompt
    assert "QUESTION:\nWhat color is the sky?" in final_prompt

def test_build_rag_prompt_without_retriever():
    llm = LLMManager(engine="local", model_name="test", retriever=None)
    final_prompt = llm._build_rag_prompt("What color is the sky?")
    assert final_prompt == "What color is the sky?"

def test_generate_uses_rag_prompt(mocker, mock_retriever):
    rag_spy = mocker.spy(LLMManager, '_build_rag_prompt')
    llm = LLMManager(engine="local", model_name="test", retriever=mock_retriever)

    llm.generate("test prompt", use_rag=True)
    rag_spy.assert_called_once()

    rag_spy.reset_mock()
    llm.generate("test prompt", use_rag=False)
    rag_spy.assert_not_called()

@pytest.mark.parametrize("engine", ["local", "openai", "huggingface_api", "ollama"])
def test_generate_non_stream(engine, mock_all_dependencies):
    if engine == "huggingface_api":
        mock_all_dependencies["requests_session"].post.return_value.json.return_value = [{'generated_text': 'HF response'}]

    elif engine == "ollama":
        mock_all_dependencies["requests_session"].post.return_value.json.return_value = {'response': 'Ollama response'}

    llm = LLMManager(engine=engine, model_name="test", api_key="test_key")
    response = llm.generate("A test prompt", stream=False)

    assert isinstance(response, str)
    if engine == "local":
        assert response == "Local response"

    elif engine == "openai":
        assert response == "OpenAI response"

def test_generate_stream_local(mock_all_dependencies):
    llm = LLMManager(engine="local", model_name="test")
    response_iterator = llm.generate("A test prompt", stream=True)

    assert isinstance(response_iterator, Iterator)
    assert response_iterator == mock_all_dependencies["local_streamer"]

def test_generate_stream_openai(mock_all_dependencies):
    mock_chunk = MagicMock()
    mock_chunk.choices[0].delta.content = "token"
    mock_all_dependencies["openai_client"].chat.completions.create.return_value = [mock_chunk, mock_chunk]
    llm = LLMManager(engine="openai", model_name="test", api_key="test_key")
    response_iterator = llm.generate("A test prompt", stream=True)

    assert isinstance(response_iterator, Iterator)
    assert "".join(list(response_iterator)) == "tokentoken"

def test_generate_stream_api_backends(mock_all_dependencies):
    mock_session = mock_all_dependencies["requests_session"]
    mock_response = MagicMock()
    mock_session.post.return_value = mock_response
    hf_lines = [b'data:{"token":{"text":"HF "}}', b'data:{"token":{"text":"response"}}']
    ollama_lines = [json.dumps({'response': 'Ollama '}).encode(), json.dumps({'response': 'response', 'done': True}).encode()]
    mock_response.iter_lines.return_value = hf_lines
    llm_hf = LLMManager(engine="huggingface_api", model_name="test", api_key="test_key")
    hf_stream = llm_hf.generate("test", stream=True)
    assert "".join(list(hf_stream)) == "HF response"

    mock_response.iter_lines.return_value = ollama_lines
    llm_ollama = LLMManager(engine="ollama", model_name="test")
    ollama_stream = llm_ollama.generate("test", stream=True)
    assert "".join(list(ollama_stream)) == "Ollama response"