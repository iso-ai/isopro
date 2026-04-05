"""LangChain Runnable / Chain backend.

Wraps any LangChain object — LCEL chains, legacy chains, agents,
retrievers — as a ModelBackend. The wrapper tries .invoke() first
(LCEL standard), then .run() (legacy), then direct __call__.

For agents that return structured dicts, set output_key to the
key that holds the final answer string.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from ..base import GenerationConfig, GenerationResult, ModelBackend, ModelInfo

logger = logging.getLogger(__name__)


class LangChainBackend(ModelBackend):
    """Wraps a LangChain Runnable or Chain as a ModelBackend.

    Args:
        chain: Any LangChain object with .invoke(), .run(), or __call__.
        chain_id: Human-readable name for this chain.
            Defaults to the class name.
        input_key: Key to use when passing prompt to dict-style invoke.
            Set to None to pass the prompt as a plain string.
        output_key: Key to extract from dict output. Common values:
            "output", "answer", "result", "text". Falls back to
            str(output) if the key is not found.
    """

    def __init__(
        self,
        chain,
        chain_id: Optional[str] = None,
        input_key: str = "input",
        output_key: str = "output",
    ) -> None:
        self.chain = chain
        self.chain_id = chain_id or type(chain).__name__
        self.input_key = input_key
        self.output_key = output_key

    # ------------------------------------------------------------------
    # ModelBackend interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Run the LangChain chain on a prompt.

        Tries invoke() → run() → __call__() in order.

        Args:
            prompt: Input text forwarded to the chain.
            config: Recorded in metadata; not applied to the chain.

        Returns:
            GenerationResult with the chain's string output.
        """
        start = time.perf_counter()
        raw_output = self._invoke_chain(prompt)
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = self._extract_text(raw_output)
        return GenerationResult(
            text=text,
            tokens_used=None,
            latency_ms=latency_ms,
            model_info=self.get_model_info(),
            metadata={"chain_type": type(self.chain).__name__},
        )

    def get_model_info(self) -> ModelInfo:
        """Return metadata identifying this as a LangChain backend.

        Returns:
            ModelInfo with provider set to "langchain".
        """
        return ModelInfo(
            model_id=self.chain_id,
            provider="langchain",
            is_local=False,
            supports_streaming=False,
            supports_layer_probing=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invoke_chain(self, prompt: str):
        """Invoke the chain using the best available method.

        Args:
            prompt: Input text.

        Returns:
            Raw output from the chain (string or dict).

        Raises:
            RuntimeError: If no compatible invocation method is found.
        """
        # LCEL chains and modern agents use .invoke()
        if hasattr(self.chain, "invoke"):
            try:
                return self.chain.invoke({self.input_key: prompt})
            except TypeError:
                # Some chains expect a plain string, not a dict.
                return self.chain.invoke(prompt)

        # Legacy chains use .run()
        if hasattr(self.chain, "run"):
            return self.chain.run(prompt)

        # Last resort: direct call
        if callable(self.chain):
            return self.chain(prompt)

        raise RuntimeError(
            f"LangChain object of type '{type(self.chain).__name__}' "
            "has no invoke(), run(), or __call__ method."
        )

    def _extract_text(self, output) -> str:
        """Extract a string from a chain's raw output.

        Args:
            output: Raw chain output (string or dict).

        Returns:
            Cleaned string response.
        """
        if isinstance(output, str):
            return output.strip()

        if isinstance(output, dict):
            # Try the configured output_key first, then common alternatives.
            for key in (self.output_key, "output", "answer", "result", "text", "response"):
                if key in output:
                    return str(output[key]).strip()

        # AIMessage and similar objects from LCEL
        if hasattr(output, "content"):
            return str(output.content).strip()

        return str(output).strip()
