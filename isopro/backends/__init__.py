"""Unified model and agent backends for ISOPro.

All environments talk exclusively to ModelBackend. Use the factory
classmethods to construct any backend — never import subclasses directly.

Quick start:

    from isopro.backends import ModelBackend, GenerationConfig

    # Local HuggingFace model (with layer probing)
    backend = ModelBackend.from_huggingface("mistralai/Mistral-7B-Instruct-v0.3")

    # Claude via Anthropic API
    backend = ModelBackend.from_api("claude-3-5-sonnet-20241022", provider="anthropic")

    # GPT-4o via OpenAI
    backend = ModelBackend.from_api("gpt-4o", provider="openai")

    # Together AI (OpenAI-compatible)
    backend = ModelBackend.from_api(
        "meta-llama/Llama-3-70b-chat-hf",
        provider="openai",
        base_url="https://api.together.xyz/v1",
        api_key="...",
    )

    # DSPy program
    backend = ModelBackend.from_dspy(my_dspy_module)

    # LangChain chain or agent
    backend = ModelBackend.from_langchain(my_lcel_chain)

    # AutoGen agent
    backend = ModelBackend.from_autogen(my_autogen_agent)

    # Generate
    result = backend.generate("What is the porosity of the Wolfcamp formation?")
    print(result.text)
    print(f"Latency: {result.latency_ms:.0f}ms")

    # Layer probing (HuggingFace only)
    if result.model_info.supports_layer_probing:
        record = backend.get_layer_activations("...")
        print(record.top_layers)
"""

from .base import GenerationConfig, GenerationResult, ModelBackend, ModelInfo
from .huggingface import (
    GradientSensitivityRecord,
    HuggingFaceBackend,
    LayerActivationRecord,
)

__all__ = [
    # Core interface
    "ModelBackend",
    "ModelInfo",
    "GenerationConfig",
    "GenerationResult",
    # HuggingFace + probing types
    "HuggingFaceBackend",
    "LayerActivationRecord",
    "GradientSensitivityRecord",
]
