"""Abstract base for all model and agent backends.

Every environment in ISOPro talks exclusively to ModelBackend.
It never knows whether it's speaking to a local HuggingFace model,
a Claude API call, or a DSPy program — the interface is identical.

Factory classmethods on ModelBackend are the single entry point
for constructing any backend:

    backend = ModelBackend.from_huggingface("mistralai/Mistral-7B-Instruct-v0.3")
    backend = ModelBackend.from_api("claude-3-5-sonnet-20241022", provider="anthropic")
    backend = ModelBackend.from_dspy(my_program)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    """Static metadata about a backend model.

    Attributes:
        model_id: Model name or HuggingFace repo ID.
        provider: One of "huggingface", "openai", "anthropic",
            "gemini", "dspy", "langchain", "autogen".
        is_local: True when weights run on the local machine.
        supports_streaming: True when generate_stream yields tokens
            incrementally rather than all at once.
        supports_layer_probing: True only for HuggingFace backends
            where we have direct access to model internals.
        parameter_count: Approximate number of parameters, if known.
        context_length: Maximum token context window, if known.
    """

    model_id: str
    provider: str
    is_local: bool
    supports_streaming: bool
    supports_layer_probing: bool
    parameter_count: Optional[int] = None
    context_length: Optional[int] = None


@dataclass
class GenerationConfig:
    """Unified generation parameters passed to every backend.

    Backends translate these into their provider-specific equivalents.
    Parameters not supported by a given backend are silently ignored.

    Attributes:
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).
        top_p: Nucleus sampling probability mass.
        stop_sequences: List of strings that halt generation.
        system_prompt: System-level instruction prepended to the
            conversation. Handled natively by Anthropic and OpenAI;
            prepended to the prompt for HuggingFace models.
    """

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stop_sequences: list[str] = field(default_factory=list)
    system_prompt: str = ""


@dataclass
class GenerationResult:
    """Output from a single generate() call.

    Attributes:
        text: The generated text (decoded, stripped).
        tokens_used: Total tokens consumed (prompt + completion).
            None when the backend does not report token counts.
        latency_ms: Wall-clock time for the generation call in ms.
        model_info: Reference to the backend's ModelInfo.
        metadata: Provider-specific extras (finish_reason, logprobs, etc.).
    """

    text: str
    tokens_used: Optional[int]
    latency_ms: float
    model_info: ModelInfo
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ModelBackend(ABC):
    """Unified interface for all models and agents in ISOPro.

    Subclass this to add new providers. The only required methods
    are generate() and get_model_info(). generate_stream() defaults
    to yielding the full generate() result as a single chunk if not
    overridden.
    """

    # ------------------------------------------------------------------
    # Abstract interface — all backends must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate a response to a prompt.

        Args:
            prompt: The user-facing input text.
            config: Generation parameters. Uses backend defaults if None.

        Returns:
            GenerationResult with text, latency, and metadata.
        """

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Return static metadata about this backend.

        Returns:
            ModelInfo describing the model, provider, and capabilities.
        """

    # ------------------------------------------------------------------
    # Optional — backends override for true streaming
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Stream generation token by token (or chunk by chunk).

        Default implementation buffers the full generate() result and
        yields it as a single chunk. Override for real streaming.

        Args:
            prompt: The user-facing input text.
            config: Generation parameters. Uses backend defaults if None.

        Yields:
            String chunks of the generated response.
        """
        result = self.generate(prompt, config)
        yield result.text

    # ------------------------------------------------------------------
    # Factory classmethods — the public construction API
    # ------------------------------------------------------------------

    @classmethod
    def from_huggingface(
        cls,
        model_id: str,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        **kwargs,
    ) -> "ModelBackend":
        """Create a backend backed by a local HuggingFace model.

        Supports layer activation probing and gradient sensitivity
        analysis — capabilities not available in API backends.

        Args:
            model_id: HuggingFace model repo ID (e.g. "mistralai/Mistral-7B-Instruct-v0.3").
            load_in_4bit: Quantize to 4-bit via bitsandbytes (requires CUDA).
            device_map: HuggingFace device map strategy ("auto", "cpu", "cuda").
            **kwargs: Additional arguments forwarded to HuggingFaceBackend.

        Returns:
            A HuggingFaceBackend instance (model not loaded until first call).
        """
        from .huggingface import HuggingFaceBackend

        return HuggingFaceBackend(
            model_id=model_id,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
            **kwargs,
        )

    @classmethod
    def from_api(
        cls,
        model: str,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> "ModelBackend":
        """Create a backend that calls an external API.

        Supports OpenAI, Anthropic, Gemini, and any OpenAI-compatible
        endpoint (Together, Groq, Fireworks, local vLLM, etc.) via
        the base_url override.

        Args:
            model: Model name as the provider expects it.
            provider: One of "openai", "anthropic", "gemini".
                Use "openai" with base_url for OpenAI-compatible APIs.
            api_key: API key. Falls back to environment variables if None.
            base_url: Override for OpenAI-compatible providers
                (e.g. "https://api.together.xyz/v1").
            **kwargs: Additional arguments forwarded to the backend.

        Returns:
            The appropriate APIBackend subclass.

        Raises:
            ValueError: If provider is not recognized.
        """
        if provider == "anthropic":
            from .anthropic_backend import AnthropicBackend

            return AnthropicBackend(model=model, api_key=api_key, **kwargs)

        if provider == "gemini":
            from .gemini_backend import GeminiBackend

            return GeminiBackend(model=model, api_key=api_key, **kwargs)

        if provider == "openai":
            from .openai_backend import OpenAIBackend

            return OpenAIBackend(
                model=model, api_key=api_key, base_url=base_url, **kwargs
            )

        raise ValueError(
            f"Unknown provider '{provider}'. "
            "Use 'openai', 'anthropic', or 'gemini'. "
            "For OpenAI-compatible APIs (Together, Groq, vLLM), "
            "use provider='openai' with base_url set."
        )

    @classmethod
    def from_dspy(cls, program, **kwargs) -> "ModelBackend":
        """Wrap a DSPy Module as a ModelBackend.

        Args:
            program: An instantiated DSPy Module (e.g. dspy.Predict,
                dspy.ChainOfThought, or a custom Module).
            **kwargs: Additional arguments forwarded to DSPyBackend.

        Returns:
            A DSPyBackend instance.
        """
        from .agent_backends.dspy_backend import DSPyBackend

        return DSPyBackend(program=program, **kwargs)

    @classmethod
    def from_langchain(cls, chain, **kwargs) -> "ModelBackend":
        """Wrap a LangChain Runnable or Chain as a ModelBackend.

        Args:
            chain: Any LangChain object implementing .invoke() or .run(),
                including LCEL chains, agents, and retrievers.
            **kwargs: Additional arguments forwarded to LangChainBackend.

        Returns:
            A LangChainBackend instance.
        """
        from .agent_backends.langchain_backend import LangChainBackend

        return LangChainBackend(chain=chain, **kwargs)

    @classmethod
    def from_autogen(cls, agent, **kwargs) -> "ModelBackend":
        """Wrap an AutoGen ConversableAgent as a ModelBackend.

        Args:
            agent: An instantiated AutoGen agent (ConversableAgent
                or any subclass).
            **kwargs: Additional arguments forwarded to AutoGenBackend.

        Returns:
            An AutoGenBackend instance.
        """
        from .agent_backends.autogen_backend import AutoGenBackend

        return AutoGenBackend(agent=agent, **kwargs)

    # ------------------------------------------------------------------
    # Shared utility
    # ------------------------------------------------------------------

    @staticmethod
    def _timed_call(fn, *args, **kwargs) -> tuple:
        """Call fn(*args, **kwargs) and return (result, latency_ms).

        Args:
            fn: Callable to time.
            *args: Positional arguments for fn.
            **kwargs: Keyword arguments for fn.

        Returns:
            Tuple of (fn result, elapsed milliseconds as float).
        """
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return result, latency_ms
