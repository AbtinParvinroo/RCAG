
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
    from pipeline import RAGBuilder

except ImportError:
    print("[!] Could not find RAGBuilder. Make sure it's in a file named main.py or accessible in your PYTHONPATH.")
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
    """Main interactive loop to build the RAG configuration."""
    print("--- Welcome to the Interactive RAG Project Builder! ---")

    project_name = get_text_input("Enter a name for your project", "MyRAGApp")

    rag_config = {
        "embedder": {},
        "vector_store": {},
        "llm": {}
    }

    vs_backend = get_user_choice("Choose your Vector Store backend:", ["faiss", "qdrant", "chroma"])
    vs_dim = int(get_text_input("Enter the vector dimension for your embeddings (e.g., 384, 768)", "384"))

    vs_config = {"backend": vs_backend, "dim": vs_dim}
    if vs_backend == "qdrant":
        vs_config["host"] = get_text_input("Enter Qdrant host", "localhost")
        vs_config["port"] = int(get_text_input("Enter Qdrant port", "6333"))

    rag_config["vector_store"] = vs_config
    llm_engine = get_user_choice("Choose your LLM engine:", ["openai", "local", "ollama"])
    default_model = "gpt-4o-mini" if llm_engine == "openai" else "llama3"
    llm_model_name = get_text_input(f"Enter the model name for '{llm_engine}'", default_model)
    llm_config = {"engine": llm_engine, "model_name": llm_model_name}
    if llm_engine == "openai":
        llm_config["api_key"] = get_text_input("Enter your OpenAI API key (or press Enter to use environment variable)", "")

    rag_config["llm"] = llm_config
    embedder_model = get_text_input("Enter the embedding model name from Hugging Face", "all-MiniLM-L6-v2")
    rag_config["embedder"]["model_name"] = embedder_model
    print("\n--- Configuration Summary ---")
    print(json.dumps(rag_config, indent=2))
    confirm = get_text_input("Do you want to build the project with this configuration? (yes/no)", "yes").lower()

    if confirm in ['yes', 'y']:
        print("\n--- Starting Project Build ---")
        try:
            builder = RAGBuilder(project_name=project_name, config=rag_config)
            final_zip_path = builder.build()
            config_save_path = os.path.join(builder.output_dir, 'config.yaml')
            with open(config_save_path, 'w') as f:
                yaml.dump(rag_config, f, indent=2, sort_keys=False)

            print(f"[✔] Configuration saved to: {config_save_path}")
            print("\n----------------------------------------------------")
            print("🎉 Your RAG application package has been created! 🎉")
            print(f"Zip file available at: {final_zip_path}")
            print(f"Project source code available in: ./{project_name}/")
            print("\nNext steps:")
            print(f"1. Unzip the file or go into the '{project_name}' directory.")
            print(f"2. (Optional) Review and edit the generated 'config.yaml'.")
            print(f"3. Install dependencies: pip install -r requirements.txt")
            print(f"4. Edit 'main.py' to add your data ingestion logic.")
            print("5. Run your pipeline: python main.py")
            print("----------------------------------------------------")

        except Exception as e:
            print(f"\n[!] An error occurred during the build process: {e}")
            print("[!] Please ensure all component files exist and are accessible.")

    else:
        print("Build cancelled.")