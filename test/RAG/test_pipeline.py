from pipeline import RAGBuilder, ProjectConfig
from unittest.mock import MagicMock, patch
import pytest
import shutil
import yaml
import json
import os

@pytest.fixture
def sample_config():
    """Provides a valid sample configuration dictionary for tests."""
    return {
        "vector_store": {"backend": "faiss", "dim": 384},
        "llm": {"engine": "openai", "model_name": "gpt-4o-mini"},
        "post_processor": {"steps": ["clean", "relevance_check"]}
    }

@pytest.fixture
def mock_component_source_dir(tmp_path):
    """
    Creates a fake source directory ('rag_components') inside the temporary
    test directory, so that the file copy operations can succeed.
    """
    source_dir = tmp_path / "rag_components"
    source_dir.mkdir()
    component_files = [
        "llm.py", "embedder.py", "vector_db.py", "retriever.py",
        "chunker.py", "post_processor.py"
    ]

    for filename in component_files:
        (source_dir / filename).touch()

    return source_dir

class TestRAGBuilder:

    def test_initialization_from_dict(self, sample_config):
        builder = RAGBuilder(project_name="test_project", config=sample_config)
        assert builder.project_name == "test_project"
        assert isinstance(builder.config, ProjectConfig)
        assert builder.config.llm.engine == "openai"

    def test_initialization_from_yaml(self, tmp_path, sample_config):
        config_path = tmp_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)

        builder = RAGBuilder.from_yaml(project_name="yaml_project", config_path=str(config_path))
        assert builder.project_name == "yaml_project"

    def test_prepare_output_dir_creates_and_cleans(self, tmp_path, sample_config):
        builder = RAGBuilder("test_project", sample_config)
        builder.output_dir = tmp_path / "test_project"

        builder._prepare_output_dir()
        assert os.path.isdir(builder.output_dir)

        (builder.output_dir / "dummy_file.txt").touch()
        builder._prepare_output_dir()
        assert not os.path.exists(os.path.join(builder.output_dir, "dummy_file.txt"))

    def test_build_orchestration_flow(self, mocker, tmp_path, sample_config):
        mocker.patch('os.getcwd', return_value=str(tmp_path))
        builder = RAGBuilder("test_project", sample_config)

        prepare_spy = mocker.spy(builder, '_prepare_output_dir')
        copy_spy = mocker.spy(builder, '_copy_component_file')
        reqs_spy = mocker.spy(builder, '_create_requirements_file')
        main_spy = mocker.spy(builder, '_create_main_script')
        zip_spy = mocker.patch('shutil.make_archive', return_value='test_project.zip')
        mocker.patch('os.path.exists', return_value=True)
        mocker.patch('shutil.rmtree')
        mocker.patch('os.makedirs')
        mocker.patch('shutil.copy')
        mocker.patch("builtins.open")
        builder.build()
        prepare_spy.assert_called_once()
        assert copy_spy.call_count == 6
        reqs_spy.assert_called_once()
        main_spy.assert_called_once()
        zip_spy.assert_called_once_with("test_project", 'zip', str(tmp_path / "test_project"))

    @patch('os.getcwd')
    def test_build_creates_correct_files_and_content(self, mock_getcwd, mocker, tmp_path, sample_config, mock_component_source_dir):
        mock_getcwd.return_value = str(tmp_path)
        builder = RAGBuilder(project_name="my_rag_app", config=sample_config)
        original_os_path_join = os.path.join
        def mock_join(base, *p):
            if base == "rag_components":
                return str(mock_component_source_dir / p[-1])

            return original_os_path_join(base, *p)

        mocker.patch('pipeline.os.path.join', side_effect=mock_join)
        zip_path = builder.build()
        project_dir = tmp_path / "my_rag_app"
        assert project_dir.is_dir()
        assert (project_dir / "main.py").is_file()
        assert (project_dir / "requirements.txt").is_file()
        assert (project_dir / "components" / "llm.py").is_file()
        with open(project_dir / "requirements.txt", 'r') as f:
            content = f.read()
            assert "faiss-cpu" in content
            assert "openai" in content

        with open(project_dir / "main.py", 'r') as f:
            content = f.read()
            assert '"backend": "faiss"' in content
            assert '"engine": "openai"' in content

        assert zip_path == str(tmp_path / "my_rag_app.zip")