from unittest.mock import MagicMock, patch
from collections.abc import Iterator
from LLM import LLMManager
import pytest
import json

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
    """Tests that a valid configuration initializes without error."""
    llm = LLMManager(engine="openai", model_name="gpt-4", api_key="test_key")
    assert llm.engine == "openai"
    assert llm.client is not None

def test_initialization_unsupported_engine_fails():
    """Tests that an unsupported engine raises a ValueError."""
    with pytest.raises(ValueError, match="Unsupported engine"):
        LLMManager(engine="unsupported_engine", model_name="test")

def test_initialization_missing_api_key_fails():
    """Tests that an API-based engine fails without an API key."""
    with pytest.raises(ValueError, match="API key is required"):
        LLMManager(engine="openai", model_name="gpt-4", api_key=None)

def test_initialization_missing_local_dependency_fails(mocker):
    """Tests that the 'local' engine fails if transformers is not installed."""
    mocker.patch('llm.TRANSFORMERS_AVAILABLE', False)
    with pytest.raises(ImportError, match="install 'torch' and 'transformers'"):
        LLMManager(engine="local", model_name="gpt2")

def test_format_prompt_with_context():
    """Tests that context is correctly formatted into the prompt."""
    llm = LLMManager(engine="local", model_name="test")
    final_prompt = llm._format_prompt(
        user_query="What is the capital of France?",
        context="France is a country in Europe."
    )

    assert "CONTEXT:\nFrance is a country in Europe." in final_prompt
    assert "QUESTION:\nWhat is the capital of France?" in final_prompt

def test_format_prompt_without_context():
    """Tests that the prompt is unchanged if no context is provided."""
    llm = LLMManager(engine="local", model_name="test")
    final_prompt = llm._format_prompt(
        user_query="What is the capital of France?",
        context=None
    )

    assert final_prompt == "What is the capital of France?"

@pytest.mark.parametrize("engine", ["local", "openai", "huggingface_api", "ollama"])
def test_generate_calls_format_prompt_and_dispatches(mocker, engine):
    """
    Tests that the main `generate` call correctly uses the prompt formatter
    and dispatches to the correct internal engine method.
    """
    llm = LLMManager(engine=engine, model_name="test", api_key="test_key")
    format_spy = mocker.spy(llm, '_format_prompt')
    generate_spy = mocker.spy(llm, f'_generate_{engine}')
    llm.generate(prompt="the query", context="the context")
    format_spy.assert_called_once_with(user_query="the query", context="the context")
    generate_spy.assert_called_once()
    final_prompt_arg = generate_spy.call_args[0][0]
    assert "CONTEXT:\nthe context" in final_prompt_arg
    assert "QUESTION:\nthe query" in final_prompt_arg

@pytest.mark.parametrize("engine", ["local", "openai", "huggingface_api", "ollama"])
def test_generate_non_stream(engine, mock_all_dependencies):
    """Tests the non-streaming output for all supported engines."""
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

def test_generate_stream_api_backends(mock_all_dependencies):
    """Tests streaming for requests-based backends (HF, Ollama)."""
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