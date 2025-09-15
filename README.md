# Universal AI Pipeline Builder 🚀

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-production_ready-brightgreen)]()

An intelligent, modular platform for rapidly building, configuring, and deploying advanced AI pipelines. This toolkit empowers you to effortlessly create custom **RAG**, **SAG**, and **CAG** applications with just a few commands.

## ✨ Key Features

- **Modular Architecture:** All components (Embedder, LLM, Memory, etc.) are designed as independent, reusable modules.
- **Effortless Configuration:** Build your pipeline using a simple `config.yaml` file or the included Interactive Configuration Wizard.
- **Three Core Architectures:**
  - **RAG:** For building powerful Question-Answering systems over your documents.
  - **SAG:** For creating intelligent chatbots with persistent, scalable memory.
  - **CAG:** For deploying state-of-the-art assistants that combine conversational memory with external knowledge retrieval.
- **Deployable Output:** The final product is a clean, self-contained project folder with all necessary code and a `requirements.txt` file.
- **Highly Flexible:** Easily swap out LLMs, vector databases, and memory strategies to fit your specific needs.

---
## 🏛️ Supported Architectures

This platform allows you to generate three distinct types of AI assistants based on your requirements:

### 1. **RAG (Retrieval-Augmented Generation)**
> The Document Expert 📚

A powerful system designed to answer questions based on a provided set of external documents and knowledge bases. It has no conversational memory.

**Workflow:** `Query` -> `Retrieve Documents` -> `LLM` -> `Answer`

### 2. **SAG (Standard Agent Generation)**
> The Smart Conversationalist 💬

An intelligent chatbot that remembers past interactions and can maintain long, coherent conversations. It does not have access to external knowledge.

**Workflow:** `Query` + `Conversation History` -> `Compress Memory` -> `LLM` -> `Answer`

### 3. **CAG (Conversational Augmented Generation)**
> The All-in-One Assistant (RAG + SAG) 🧠

The most advanced agent, combining the best of both worlds. This assistant both remembers your conversation and can perform research in its knowledge base to answer your questions.

**Workflow:** `Query` + `History` -> `Rewrite Query` -> `Retrieve Documents` + `Compress History` -> `LLM` -> `Answer`

---
## 🚀 Quick Start

Get your first AI assistant up and running in minutes!

1.  **Clone the Project:**
    ```bash
    git clone [your-repo-url]
    cd [your-repo-folder]
    ```

2.  **Install Builder Dependencies:**
    * Ensure all your component source files (`chunker.py`, `llm.py`, etc.) are placed in their respective `rag_components`, `sag_components`, and `cag_components` directories.
    * Install the core dependencies for the builder tool itself:
    ```bash
    pip install pydantic pyyaml jinja2
    ```

3.  **Run the Configuration Wizard:**
    Execute the interactive script to build your pipeline step-by-step.
    ```bash
    python configure.py
    ```
    The wizard will ask for your desired pipeline type (RAG, SAG, or CAG) and guide you through the relevant settings.

4.  **Run Your Generated Pipeline:**
    * After the wizard finishes, a new project folder (e.g., `MyCAGApp`) and a `.zip` file will be created.
    * Navigate into your new project directory:
    ```bash
    cd MyCAGApp
    ```
    * Install the pipeline-specific dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    * Finally, run the application:
    ```bash
    python main.py
    ```

---
## 📦 Project Components

This platform is built from a suite of powerful, modular components:

| Component File        | Main Class          | Role & Responsibility                                              |
| --------------------- | ------------------- | ------------------------------------------------------------------ |
| `detector.py`         | `DataTypeDetector`  | Automatically detects the type of input files (PDF, TXT, CSV, etc.). |
| `chunker.py`          | `RAGChunker`        | Splits documents and multimedia files into small, manageable chunks. |
| `embedder.py`         | `SmartEmbedder`     | Converts text, images, and audio into numerical vectors (embeddings).|
| `vector_db.py`        | `SmartVectorStore`  | A unified interface for working with various vector databases.       |
| `retriever.py`        | `Retriever`         | An intelligent search engine for retrieving relevant information.      |
| `memory_layer.py`     | `MemoryLayer`       | Manages conversation history persistently using Redis.             |
| `compressor.py`       | `ContextCompressor` | Intelligently compresses and summarizes conversation history.        |
| `prompt_builder.py`   | `PromptFormatter`   | Assembles and formats the final prompt to be sent to the LLM.      |
| `post_processor.py`   | `PostProcessor`     | A Quality Control (QC) unit to clean and validate the LLM's final answer.|
| `llm.py`              | `LLMManager`        | The system's "brain" and a universal interface for various LLMs.   |
| **`pipeline.py`** | **`CAGBuilder`** | **The master factory that assembles all components into an application.** |
| **`configure.py`**| -                   | **The interactive wizard that makes the building process easy.** |

---
## ⚙️ Configuration (`config.yaml`)

The `configure.py` wizard automatically generates a `config.yaml` file inside your new project. This file allows you to easily review and modify your pipeline's settings.

**Example `config.yaml` for a full CAG pipeline:**
```yaml
pipeline_type: cag
embedder:
  model_name: all-MiniLM-L6-v2
vector_store:
  backend: faiss
  dim: 384
retriever: {}
memory:
  strategy: top_k
  redis:
    host: localhost
    port: 6379
llm:
  engine: openai
  model_name: gpt-4o-mini
post_processor:
  steps:
    - clean
    - relevance_check
