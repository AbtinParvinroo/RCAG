from typing import Dict, Any, Set, List, Literal
from pydantic import BaseModel, Field
from jinja2 import Environment
import zipfile
import shutil
import yaml
import json
import uuid
import os

class LLMConfig(BaseModel):
    engine: Literal['openai', 'local', 'ollama']
    model_name: str

class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0

class MemoryConfig(BaseModel):
    strategy: Literal['summarization', 'top_k', 'cluster', 'merge_trim', 'graph']
    redis: RedisConfig = Field(default_factory=RedisConfig)
    compressor_params: Dict[str, Any] = Field(default_factory=dict)

class VectorStoreConfig(BaseModel):
    backend: Literal['faiss', 'qdrant', 'chroma']
    dim: int

class EmbedderConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"

class PostProcessorConfig(BaseModel):
    steps: List[str] = ["clean"]

class ProjectConfig(BaseModel):
    llm: LLMConfig
    memory: MemoryConfig
    vector_store: VectorStoreConfig
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    post_processor: PostProcessorConfig = Field(default_factory=PostProcessorConfig)

CAG_MAIN_SCRIPT_TEMPLATE = """
# Auto-generated main.py
CONFIG = {{ config_json }}
"""

class CAGBuilder:
    """Builds a full CAG pipeline project"""
    def __init__(self, project_name: str, config: Dict[str, Any]):
        self.project_name = project_name
        self.config: ProjectConfig = ProjectConfig.parse_obj(config)
        self.output_dir = os.path.join(os.getcwd(), self.project_name)
        self.component_map = {
            "llm": "llm",
            "memory_layer": "memory_layer",
            "compressor": "compressor",
            "prompt_builder": "prompt_builder",
            "embedder": "embedder",
            "vector_db": "vector_db",
            "retriever": "retriever",
            "chunker": "chunker",
            "post_processor": "post_processor",
        }

        self.dependency_map = {
            "llm": ["requests", "openai"],
            "memory_layer": ["redis"],
            "compressor": ["scikit-learn", "transformers", "numpy", "spacy", "networkx"],
            "prompt_builder": ["pandas", "numpy"],
            "embedder": ["sentence-transformers", "torch"],
            "vector_db": ["faiss-cpu", "qdrant-client", "chromadb", "weaviate-client", "pymilvus"],
            "retriever": [],
            "chunker": ["pydub", "opencv-python", "Pillow"],
            "post_processor": ["scikit-learn", "spacy", "openai"],
        }

    @classmethod
    def from_yaml(cls, project_name: str, config_path: str):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(project_name, config_dict)

    def _prepare_output_dir(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        os.makedirs(os.path.join(self.output_dir, "components"), exist_ok=True)

    def _copy_component_file(self, component_filename: str):
        source_path = os.path.join("cag_components", component_filename)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        dest_path = os.path.join(self.output_dir, "components", component_filename)
        shutil.copy(source_path, dest_path)
        print(f"[✔] Copied component: {component_filename}")

    def _create_main_script(self):
        env = Environment()
        template = env.from_string(CAG_MAIN_SCRIPT_TEMPLATE)
        script_content = template.render(config_json=self.config.model_dump_json(indent=4))
        path = os.path.join(self.output_dir, "main.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(script_content)

        print("[✔] Generated main.py")

    def _create_requirements_file(self, dependencies: Set[str]):
        dependencies.update(["pydantic", "pyyaml", "jinja2"])
        path = os.path.join(self.output_dir, "requirements.txt")
        with open(path, "w") as f:
            for dep in sorted(list(dependencies)):
                f.write(f"{dep}\n")

        print("[✔] Generated requirements.txt")

    def build(self):
        print(f"--- Building CAG Project: {self.project_name} ---")
        self._prepare_output_dir()
        components_to_copy = list(self.component_map.values())
        required_deps = set()

        for filename in components_to_copy:
            self._copy_component_file(filename)
            required_deps.update(self.dependency_map.get(filename.split(".")[0], []))

        self._create_requirements_file(required_deps)
        self._create_main_script()
        zip_path = shutil.make_archive(self.project_name, 'zip', self.output_dir)
        print(f"--- Build Complete! Project zipped at: {zip_path} ---")
        return zip_path