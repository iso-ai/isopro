"""Anthropic Claude API backend.

API key falls back to the ANTHROPIC_API_KEY environment variable.
Supports all current Claude models (claude-3-5-sonnet, claude-3-opus, etc.)
and streaming via the Anthropic SDK's native streaming interface.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator, Optional

from .base import GenerationConfig, GenerationResult, ModelBackend, ModelInfo

logger = logging.getLogger(__name__)

# Anthropic requires a max_tokens value — use this when the caller
# does not provide a GenerationConfig.
_DEFAULT_MAX_TOKENS = 1024


class AnthropicBackend(ModelBackend):
    """Backend for Anthropic Claude models via the Anthropic SDK.

    Args:
        model: Claude model ID (e.g. "claude-3-5-sonnet-20241022").
        api_key: Anthropic API key. Reads ANTHROPIC_API_KEY if None.
        default_config: Default GenerationConfig for this backend.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        default_config: Optional[GenerationConfig] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.default_config = default_config or GenerationConfig(
            max_new_tokens=_DEFAULT_MAX_TOKENS
        )
        self._client = None  # Lazy init.

    # ------------------------------------------------------------------
    # ModelBackend interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Call the Anthropic messages API and return the response.

        Args:
            prompt: User-facing input text.
            config: Generation parameters. Uses default_config if None.

        Returns:
            GenerationResult with text, token usage, and latency.
        """
        config = config or self.default_config
        client = self._get_client()

        kwargs = dict(
            model=self.model,
            max_tokens=config.max_new_tokens,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            top_p=config.top_p,
        )
        if config.system_prompt:
            kwargs["system"] = config.system_prompt
        if config.stop_sequences:
            kwargs["stop_sequences"] = config.stop_sequences

        def _call():
            return client.messages.create(**kwargs)

        response, latency_ms = self._timed_call(_call)
        text = response.content[0].text if response.content else ""
        tokens_used = (
            response.usage.input_tokens + response.usage.output_tokens
            if response.usage
            else None
        )

        return GenerationResult(
            text=text.strip(),
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            model_info=self.get_model_info(),
            metadata={
                "stop_reason": response.stop_reason,
                "model": response.model,
            },
        )

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Stream Claude's response token by token.

        Args:
            prompt: User-facing input text.
            config: Generation parameters. Uses default_config if None.

        Yields:
            Text delta strings from each streaming event.
        """
        config = config or self.default_config
        client = self._get_client()

        kwargs = dict(
            model=self.model,
            max_tokens=config.max_new_tokens,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            top_p=config.top_p,
        )
        if config.system_prompt:
            kwargs["system"] = config.system_prompt
        if config.stop_sequences:
            kwargs["stop_sequences"] = config.stop_sequences

        with client.messages.stream(**kwargs) as stream:
            for text_chunk in stream.text_stream:
                yield text_chunk

    def get_model_info(self) -> ModelInfo:
        """Return metadata for this Anthropic backend.

        Returns:
            ModelInfo with provider set to "anthropic".
        """
        return ModelInfo(
            model_id=self.model,
            provider="anthropic",
            is_local=False,
            supports_streaming=True,
            supports_layer_probing=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        """Return the cached Anthropic client, creating it on first call."""
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.api_key)
        return self._client
