import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

core_deps = [
    "numpy",
    "pandas",
    "scikit-learn",
    "nltk",
    "requests",
    "redis",
    "openai>=1.0",
    "pydantic",
    "pyyaml",
    "jinja2",
]

extras = {
    "local_llm": ["torch", "transformers>=4.41.0", "accelerate", "bitsandbytes", "peft"],
    "multimodal": ["Pillow", "pydub", "opencv-python"],
    "vector_dbs": ["faiss-cpu", "qdrant-client", "chromadb", "weaviate-client", "pymilvus"],
    "nlp_heavy": ["spacy"],
    "all": [
        "torch", "transformers>=4.41.0", "accelerate", "bitsandbytes", "peft",
        "Pillow", "pydub", "opencv-python",
        "faiss-cpu", "qdrant-client", "chromadb", "weaviate-client", "pymilvus",
        "spacy"
    ]
}

setuptools.setup(
    name="universal-ai-builder",
    version="1.0.0",
    author="[Abtin Parvinroo]",
    author_email="[abtin83parvinroo@gmail.com]",
    description="A modular platform to build, configure, and deploy RAG, SAG, and CAG pipelines.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="[https://github.com/AbtinParvinroo/RCAG.git]",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    python_requires='>=3.9',
    install_requires=core_deps,
    extras_require=extras,
    entry_points={
        'console_scripts': [
            'build-ai-pipeline = universal_ai_builder.configure:run_interactive_configuration',
        ],
    },
)