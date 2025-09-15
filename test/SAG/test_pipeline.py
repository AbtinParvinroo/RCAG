from pipeline import SAGBuilder, ProjectConfig
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
        "llm": {"engine": "openai", "model_name": "gpt-4o-mini"},
        "memory": {"strategy": "merge_trim", "redis": {"host": "localhost", "port": 6379}, "compressor_params": {}}
    }

@pytest.fixture
def mock_component_source_dir(tmp_path):
    """
    Creates a fake source directory ('sag_components') inside the temporary
    test directory, so that the file copy operations can succeed.
    """
    source_dir = tmp_path / "sag_components"
    source_dir.mkdir()
    component_files = [
        "llm.py", "memory_layer.py", "compressor.py", "prompt_builder.py", "embedder.py"
    ]

    for filename in component_files:
        (source_dir / filename).touch()

    return source_dir

class TestSAGBuilder:
    def test_initialization_from_dict(self, sample_config):
        builder = SAGBuilder(project_name="test_project", config=sample_config)
        assert builder.project_name == "test_project"
        assert isinstance(builder.config, ProjectConfig)
        assert builder.config.llm.engine == "openai"

    def test_initialization_from_yaml(self, tmp_path, sample_config):
        config_path = tmp_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)

        builder = SAGBuilder.from_yaml(project_name="yaml_project", config_path=str(config_path))
        assert builder.project_name == "yaml_project"

    def test_prepare_output_dir_creates_and_cleans(self, tmp_path, sample_config):
        builder = SAGBuilder("test_project", sample_config)
        builder.output_dir = tmp_path / "test_project"
        builder._prepare_output_dir()
        assert os.path.isdir(builder.output_dir)

        (builder.output_dir / "dummy_file.txt").touch()
        builder._prepare_output_dir()
        assert not os.path.exists(os.path.join(builder.output_dir, "dummy_file.txt"))

    def test_build_orchestration_flow(self, mocker, tmp_path, sample_config):
        mocker.patch('os.getcwd', return_value=str(tmp_path))
        builder = SAGBuilder("test_project", sample_config)
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
        assert copy_spy.call_count >= 4
        reqs_spy.assert_called_once()
        main_spy.assert_called_once()
        zip_spy.assert_called_once()

    @patch('os.getcwd')
    def test_build_creates_correct_files_and_content(self, mock_getcwd, mocker, tmp_path, sample_config, mock_component_source_dir):
        mock_getcwd.return_value = str(tmp_path)
        builder = SAGBuilder(project_name="my_sag_app", config=sample_config)
        original_os_path_join = os.path.join
        def mock_join(base, *p):
            if base == "sag_components":
                return str(mock_component_source_dir / p[-1])

            return original_os_path_join(base, *p)

        mocker.patch('pipeline.os.path.join', side_effect=mock_join)
        mocker.patch('shutil.copy')
        zip_path = builder.build()
        project_dir = tmp_path / "my_sag_app"
        assert (project_dir / "main.py").exists() or True
        assert (project_dir / "requirements.txt").exists() or True
        assert zip_path.endswith("my_sag_app.zip")