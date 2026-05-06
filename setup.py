from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="isopro",
    version="0.3.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "opencv-python>=4.5.0",
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "ultralytics>=8.0.0",  # For YOLO
        "pillow>=9.0.0",       # Image processing
        "stable-baselines3>=2.0.0",
        "scikit-learn>=1.0.0",
        "transformers",
        "sentence-transformers",
        "langchain",
        "langchain_openai",
        "isozero>=0.1.0",      # For reasoning capabilities
        "iso-adverse",         # For adversarial testing
        "anthropic>=0.3.0",    # For Claude API
        "openai",              # For OpenAI API
        "nltk",
        "rouge",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.65.0", 
        "python-dotenv>=0.19.0",
        "pyyaml>=6.0.0",
        "rich>=13.0.0",        # Enhanced terminal output
        # API dependencies
        "flask>=2.0.0",        # API framework
        "flask-cors>=3.0.10",  # CORS support
        "gunicorn>=20.1.0",    # WSGI server
        "werkzeug>=2.0.0",     # WSGI utilities
    ],
    entry_points={
        "console_scripts": [
            "isopro=isopro.cli.main:run",
            "isopro-mcp=isopro.mcp_server:run",
        ],
    },
    extras_require={
        # Minimal install: CLI + API backends (no GPU, no local models)
        'cli': [
            "typer>=0.12.0",
            "rich>=13.7.0",
            "openai>=1.0.0",
            "anthropic>=0.20.0",
        ],
        # MCP server support
        'mcp': [
            "mcp>=1.0.0",
        ],
        # Local model training + inference (GPU recommended)
        'train': [
            "torch>=2.0.0",
            "transformers>=4.40.0",
            "accelerate>=0.27.0",
            "bitsandbytes>=0.42.0",
        ],
        # Everything
        'all': [
            "typer>=0.12.0",
            "rich>=13.7.0",
            "openai>=1.0.0",
            "anthropic>=0.20.0",
            "mcp>=1.0.0",
            "torch>=2.0.0",
            "transformers>=4.40.0",
            "accelerate>=0.27.0",
        ],
        'dev': [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
        'api': [
            "flask>=2.0.0",
            "flask-cors>=3.0.10",
            "gunicorn>=20.1.0",
            "werkzeug>=2.0.0",
        ],
    },
    author="Jazmia Henry",
    author_email="isojaz@isoai.co",
    description=(
        "ISOPro: a reference implementation of Grounded Continuous Evaluation (GCE) "
        "— simulation-based fine-tuning and evaluation for LLMs. Replaces the learned "
        "reward model with a deterministic verifier and updates LoRA adapters on CPU, "
        "eliminating reward hacking by construction in verifiable-reward domains."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iso-ai/isopro",
    project_urls={
        "Source": "https://github.com/iso-ai/isopro",
        "Bug Tracker": "https://github.com/iso-ai/isopro/issues",
        "Examples": "https://github.com/iso-ai/isopro/tree/public/examples",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.7',
    license="Apache License 2.0",
    keywords="LLM AI simulation reinforcement-learning adversarial-attacks NLP workflow-automation computer-vision",
    package_data={
        "isopro": ["py.typed"],
    },
)