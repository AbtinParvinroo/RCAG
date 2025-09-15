from utils import get_user_choice, get_text_input
import json
import yaml
import sys
import os
try:
    from pipeline import CAGBuilder

except ImportError:
    print("[!] Could not find CAGBuilder. Make sure it's in pipeline.py or accessible in your PYTHONPATH.")
    sys.exit(1)

def run_interactive_configuration():
    print("--- Welcome to the Interactive CAG Project Builder! ---")

    project_name = get_text_input("Enter a name for your CAG project", "MyCAGApp")
    cag_config = {
        "embedder": {},
        "vector_store": {},
        "llm": {}
    }

    vs_backend = get_user_choice("Choose your Vector Store backend:", ["faiss", "qdrant", "chroma"])
    vs_dim = int(get_text_input("Enter the vector dimension for embeddings (e.g., 384, 768)", "384"))
    vs_config = {"backend": vs_backend, "dim": vs_dim}
    if vs_backend == "qdrant":
        vs_config["host"] = get_text_input("Enter Qdrant host", "localhost")
        vs_config["port"] = int(get_text_input("Enter Qdrant port", "6333"))

    cag_config["vector_store"] = vs_config
    llm_engine = get_user_choice("Choose your LLM engine:", ["openai", "local", "ollama"])
    default_model = "gpt-4o-mini" if llm_engine == "openai" else "llama3"
    llm_model_name = get_text_input(f"Enter the model name for '{llm_engine}'", default_model)
    llm_config = {"engine": llm_engine, "model_name": llm_model_name}
    if llm_engine == "openai":
        llm_config["api_key"] = get_text_input("Enter your OpenAI API key (or press Enter to use env)", "")

    cag_config["llm"] = llm_config
    embedder_model = get_text_input("Enter the embedding model name", "all-MiniLM-L6-v2")
    cag_config["embedder"]["model_name"] = embedder_model

    print("\n--- Configuration Summary ---")
    print(json.dumps(cag_config, indent=2))
    confirm = get_text_input("Do you want to build the CAG project with this config? (yes/no)", "yes").lower()
    if confirm in ["yes", "y"]:
        try:
            builder = CAGBuilder(project_name=project_name, config=cag_config)
            final_zip_path = builder.build()
            config_save_path = os.path.join(builder.output_dir, "config.yaml")
            os.makedirs(builder.output_dir, exist_ok=True)  # ✅ make sure directory exists
            with open(config_save_path, "w") as f:
                yaml.dump(cag_config, f, indent=2, sort_keys=False)

            print(f"[✔] Configuration saved to: {config_save_path}")
            print("\n🎉 CAG project created successfully! 🎉")
            print(f"Zip file: {final_zip_path}")
            print(f"Project dir: ./{project_name}/")

        except Exception as e:
            print(f"[!] Error: {e}")

    else:
        print("Build cancelled.")