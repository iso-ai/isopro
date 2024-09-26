from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="isopro",
    version="0.1.0",  # Updated version
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "transformers",
        "sentence-transformers",
        "scikit-learn",
        "anthropic",
        "openai",
        "gymnasium",  # Updated from gym
        "stable-baselines3",
        "nltk",
        "rouge",
        "tqdm",
        "matplotlib",
        "seaborn",
        "python-dotenv",
        "langchain",
        "langchain_openai",
    ],
    author="Jazmia Henry",
    author_email="isojaz@isoai.co",
    description="Intelligent Simulation Orchestration for Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/isopro",
    project_urls={
        "Bug Tracker": "https://github.com/iso-ai/isopro/tree/main/.github/ISSUE_TEMPLATE.md",
        "Documentation": "https://github.com/yourusername/isopro/wiki",
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
    keywords="LLM AI simulation reinforcement-learning adversarial-attacks NLP",
    package_data={
        "isopro": ["py.typed"],
    },
)