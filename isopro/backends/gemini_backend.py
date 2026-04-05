"""Google Gemini API backend.

API key falls back to the GOOGLE_API_KEY environment variable.
Supports all Gemini models (gemini-1.5-pro, gemini-1.5-flash, etc.)
and streaming via the google-generativeai SDK.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator, Optional

from .base import GenerationConfig, GenerationResult, ModelBackend, ModelInfo

logger = logging.getLogger(__name__)


class GeminiBackend(ModelBackend):
    """Backend for Google Gemini models via the google-generativeai SDK.

    Args:
        model: Gemini model ID (e.g. "gemini-1.5-pro", "gemini-1.5-flash").
        api_key: Google API key. Reads GOOGLE_API_KEY if None.
        default_config: Default GenerationConfig for this backend.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        default_config: Optional[GenerationConfig] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.default_config = default_config or GenerationConfig()
        self._genai_model = None  # Lazy init.

    # ------------------------------------------------------------------
    # ModelBackend interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Call the Gemini generate_content API and return the response.

        Args:
            prompt: User-facing input text.
            config: Generation parameters. Uses default_config if None.

        Returns:
            GenerationResult with text, latency, and metadata.
        """
        config = config or self.default_config
        genai_model = self._get_model(config)

        full_prompt = _build_prompt(prompt, config.system_prompt)

        def _call():
            return genai_model.generate_content(full_prompt)

        response, latency_ms = self._timed_call(_call)
        text = response.text if response.text else ""

        # Gemini does not always expose token counts in the basic SDK call.
        tokens_used = None
        try:
            tokens_used = response.usage_metadata.total_token_count
        except AttributeError:
            pass

        return GenerationResult(
            text=text.strip(),
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            model_info=self.get_model_info(),
            metadata={"finish_reason": str(response.candidates[0].finish_reason)
                      if response.candidates else ""},
        )

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Stream Gemini response chunks as they arrive.

        Args:
            prompt: User-facing input text.
            config: Generation parameters. Uses default_config if None.

        Yields:
            Text chunks from each streaming response part.
        """
        config = config or self.default_config
        genai_model = self._get_model(config)
        full_prompt = _build_prompt(prompt, config.system_prompt)

        for chunk in genai_model.generate_content(full_prompt, stream=True):
            if chunk.text:
                yield chunk.text

    def get_model_info(self) -> ModelInfo:
        """Return metadata for this Gemini backend.

        Returns:
            ModelInfo with provider set to "gemini".
        """
        return ModelInfo(
            model_id=self.model,
            provider="gemini",
            is_local=False,
            supports_streaming=True,
            supports_layer_probing=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self, config: GenerationConfig):
        """Return a configured Gemini GenerativeModel instance.

        Reconfigures generation settings from config on each call so
        temperature and max_tokens are respected per-request.

        Args:
            config: Generation parameters to apply.

        Returns:
            A google.generativeai.GenerativeModel instance.
        """
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig as GeminiGenConfig

        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(
            model_name=self.model,
            generation_config=GeminiGenConfig(
                max_output_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop_sequences=config.stop_sequences or None,
            ),
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _build_prompt(prompt: str, system_prompt: str) -> str:
    """Combine system prompt and user prompt for Gemini.

    Gemini's basic API does not have a dedicated system role in all
    versions, so we prepend it as context.

    Args:
        prompt: User-facing input.
        system_prompt: Optional system instruction.

    Returns:
        Combined prompt string.
    """
    if system_prompt:
        return f"{system_prompt}\n\n{prompt}"
    return prompt
