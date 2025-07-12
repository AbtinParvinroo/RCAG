import os
import shutil
import importlib
from datetime import datetime
import zipfile

class RAG:
    def __init__(self, config): 
        self.config = config
        self.output_dir = os.path.join(os.getcwd(), "generated_rag")
        self._prepare_output_dir()
        self.stage_map = {
        "detect": ("components.detect", "DataTypeDetector"),
        "chunk": ("components.chunk", "CAGChunker"),
        "embed": ("components.embed", "SmartEmbedder"),
        "compress": ("components.compress", "ContextCompressor"),
        "assemble": ("components.assemble", "ContextAssembler"),
        "llm": ("components.llm", "LLMManager"),
    }

    def _prepare_output_dir(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_stage_code(self, module_path, class_name):
        module = importlib.import_module(module_path)
        file_path = module.__file__
        shutil.copy(file_path, os.path.join(self.output_dir, f"{class_name}.py"))

    def build(self):
        print("[RAG Builder] Start building...")
        for stage, choice in self.config.items():
            if isinstance(choice, str):
                if choice not in ["default", "word", "sentence", "token", "clap", "clip", "imagebind"]:
                    print(f"[!] Skipping unknown method for stage {stage}: {choice}")
                    continue
                module_path, class_name = self.stage_map.get(stage, (None, None))
                if module_path and class_name:
                    print(f"[Stage: {stage}] → Using method: {choice}")
                    self._save_stage_code(module_path, class_name)
            else:
                # Custom user-defined class
                print(f"[Stage: {stage}] → Using user-defined class")
                filename = getattr(choice, '__name__', f"custom_{stage}") + ".py"
                # simulate saving user-defined code
                with open(os.path.join(self.output_dir, filename), "w") as f:
                    f.write("# Custom implementation\n")
                    f.write(str(choice))

        zip_path = self._zip_output()
        return zip_path

    def _zip_output(self):
        zip_filename = f"generated_cag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = os.path.join(os.getcwd(), zip_filename)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(self.output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.output_dir)
                    zipf.write(file_path, arcname)
        print(f"[✔] Project built and zipped at: {zip_path}")
        return zip_path

def build(config):
    builder = RAG(config)
    return builder.build()