import json
import yaml
import sys
import os

try:
    import yaml

except ImportError:
    print("[!] PyYAML is not installed. Please run: pip install pyyaml")
    sys.exit(1)

try:
    from pipeline import SAGBuilder

except ImportError:
    print("[!] Could not find SAGBuilder. Make sure it's in pipeline.py or accessible in your PYTHONPATH.")
    sys.exit(1)


def get_user_choice(prompt: str, options: list) -> str:
    """A helper function to get a valid choice from the user."""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")

    while True:
        try:
            choice_num = int(input(f"Enter your choice (1-{len(options)}): "))
            if 1 <= choice_num <= len(options):
                return options[choice_num - 1]

            else:
                print("Invalid choice. Please try again.")

        except ValueError:
            print("Invalid input. Please enter a number.")


def get_text_input(prompt: str, default: str = None) -> str:
    """A helper function to get a text input from the user."""
    prompt_text = f"{prompt} (default: {default}): " if default else f"{prompt}: "
    user_input = input(prompt_text).strip()
    return user_input or default


def run_interactive_configuration():
    print("--- Welcome to the Interactive SAG Project Builder! ---")
    project_name = get_text_input("Enter a name for your SAG project", "MySAGApp")
    sag_config = {
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

    sag_config["vector_store"] = vs_config
    llm_engine = get_user_choice("Choose your LLM engine:", ["openai", "local", "ollama"])
    default_model = "gpt-4o-mini" if llm_engine == "openai" else "llama3"
    llm_model_name = get_text_input(f"Enter the model name for '{llm_engine}'", default_model)
    llm_config = {"engine": llm_engine, "model_name": llm_model_name}
    if llm_engine == "openai":
        llm_config["api_key"] = get_text_input("Enter your OpenAI API key (or press Enter to use env)", "")

    sag_config["llm"] = llm_config

    embedder_model = get_text_input("Enter the embedding model name", "all-MiniLM-L6-v2")
    sag_config["embedder"]["model_name"] = embedder_model
    print("\n--- Configuration Summary ---")
    print(json.dumps(sag_config, indent=2))
    confirm = get_text_input("Do you want to build the SAG project with this config? (yes/no)", "yes").lower()

    if confirm in ["yes", "y"]:
        try:
            builder = SAGBuilder(project_name=project_name, config=sag_config)
            final_zip_path = builder.build()
            config_save_path = os.path.join(builder.output_dir, "config.yaml")
            with open(config_save_path, "w") as f:
                yaml.dump(sag_config, f, indent=2, sort_keys=False)

            print(f"[✔] Configuration saved to: {config_save_path}")
            print("\n🎉 SAG project created successfully! 🎉")
            print(f"Zip file: {final_zip_path}")
            print(f"Project dir: ./{project_name}/")

        except Exception as e:
            print(f"[!] Error: {e}")

    else:
        print("Build cancelled.")