"""OpenAI and OpenAI-compatible API backend.

Covers OpenAI natively and any provider that implements the OpenAI
chat completions API via base_url override:

    Provider         base_url
    --------         --------
    Together AI      https://api.together.xyz/v1
    Groq             https://api.groq.com/openai/v1
    Fireworks        https://api.fireworks.ai/inference/v1
    local vLLM       http://localhost:8000/v1
    local Ollama     http://localhost:11434/v1

API key falls back to the OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator, Optional

from .base import GenerationConfig, GenerationResult, ModelBackend, ModelInfo

logger = logging.getLogger(__name__)


class OpenAIBackend(ModelBackend):
    """Backend for OpenAI and OpenAI-compatible chat completion APIs.

    Args:
        model: Model identifier as the provider expects it
            (e.g. "gpt-4o", "meta-llama/Llama-3-70b-chat-hf").
        api_key: API key. Reads OPENAI_API_KEY from environment if None.
        base_url: Override for OpenAI-compatible providers.
            Leave None for standard OpenAI.
        default_config: Default GenerationConfig applied when callers
            do not supply one.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_config: Optional[GenerationConfig] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.default_config = default_config or GenerationConfig()
        self._client = None  # Lazy init.

    # ------------------------------------------------------------------
    # ModelBackend interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Call the chat completions endpoint and return the response.

        Args:
            prompt: User-facing input text.
            config: Generation parameters. Uses default_config if None.

        Returns:
            GenerationResult with text, token usage, and latency.
        """
        config = config or self.default_config
        client = self._get_client()
        messages = _build_messages(prompt, config.system_prompt)

        def _call():
            return client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop=config.stop_sequences or None,
            )

        response, latency_ms = self._timed_call(_call)
        text = response.choices[0].message.content or ""
        tokens_used = (
            response.usage.total_tokens if response.usage else None
        )

        return GenerationResult(
            text=text.strip(),
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            model_info=self.get_model_info(),
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "model": response.model,
            },
        )

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Stream chat completion tokens as they arrive.

        Args:
            prompt: User-facing input text.
            config: Generation parameters. Uses default_config if None.

        Yields:
            Text delta strings from each streaming chunk.
        """
        config = config or self.default_config
        client = self._get_client()
        messages = _build_messages(prompt, config.system_prompt)

        stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=config.stop_sequences or None,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def get_model_info(self) -> ModelInfo:
        """Return metadata for this OpenAI backend.

        Returns:
            ModelInfo indicating the provider and model ID.
        """
        # Infer a display-friendly provider name from base_url if set.
        provider = "openai"
        if self.base_url:
            if "together" in self.base_url:
                provider = "together_ai"
            elif "groq" in self.base_url:
                provider = "groq"
            elif "fireworks" in self.base_url:
                provider = "fireworks"
            elif "localhost" in self.base_url or "127.0.0.1" in self.base_url:
                provider = "local_openai_compatible"

        return ModelInfo(
            model_id=self.model,
            provider=provider,
            is_local="localhost" in (self.base_url or ""),
            supports_streaming=True,
            supports_layer_probing=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        """Return the cached OpenAI client, creating it on first call."""
        if self._client is None:
            from openai import OpenAI

            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _build_messages(prompt: str, system_prompt: str) -> list[dict]:
    """Build the OpenAI messages array from prompt and optional system prompt.

    Args:
        prompt: User turn content.
        system_prompt: Optional system instruction.

    Returns:
        List of message dicts in OpenAI chat format.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages
