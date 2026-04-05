"""DSPy program backend.

Wraps any DSPy Module (Predict, ChainOfThought, ReAct, custom Module)
as a ModelBackend so it can operate inside any ISOPro environment
without modification.

DSPy programs receive a single 'prompt' keyword argument. If your
program expects different field names, subclass DSPyBackend and
override _call_program().
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from ..base import GenerationConfig, GenerationResult, ModelBackend, ModelInfo

logger = logging.getLogger(__name__)


class DSPyBackend(ModelBackend):
    """Wraps a DSPy Module as a ModelBackend.

    Args:
        program: An instantiated DSPy Module. Must be callable and
            return an object with a string representation or a
            'response' / 'answer' / 'output' attribute.
        program_id: Human-readable identifier for this program.
            Defaults to the class name of the program.
        output_field: Attribute name to read from the DSPy prediction
            object. Common values: "answer", "response", "output".
            Falls back to str() of the prediction if the attribute
            is not found.
    """

    def __init__(
        self,
        program,
        program_id: Optional[str] = None,
        output_field: str = "answer",
    ) -> None:
        self.program = program
        self.program_id = program_id or type(program).__name__
        self.output_field = output_field

    # ------------------------------------------------------------------
    # ModelBackend interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Run the DSPy program on a prompt and return the result.

        Note: DSPy programs manage their own LM configuration. The
        GenerationConfig is recorded in metadata but not forwarded —
        configure temperature and max_tokens on the DSPy LM directly.

        Args:
            prompt: Input passed to the DSPy program as the 'prompt' field.
            config: Recorded in metadata only; not applied to DSPy.

        Returns:
            GenerationResult with the program's string output.
        """
        start = time.perf_counter()
        prediction = self._call_program(prompt)
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = self._extract_text(prediction)
        return GenerationResult(
            text=text,
            tokens_used=None,  # DSPy does not expose token counts directly.
            latency_ms=latency_ms,
            model_info=self.get_model_info(),
            metadata={"program_type": type(self.program).__name__},
        )

    def get_model_info(self) -> ModelInfo:
        """Return metadata identifying this as a DSPy backend.

        Returns:
            ModelInfo with provider set to "dspy".
        """
        return ModelInfo(
            model_id=self.program_id,
            provider="dspy",
            is_local=False,  # Unknown — depends on DSPy's configured LM.
            supports_streaming=False,
            supports_layer_probing=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers — override in subclasses for custom field names
    # ------------------------------------------------------------------

    def _call_program(self, prompt: str):
        """Invoke the DSPy program with the prompt as a keyword argument.

        Args:
            prompt: Input text.

        Returns:
            DSPy Prediction object.
        """
        return self.program(prompt=prompt)

    def _extract_text(self, prediction) -> str:
        """Extract a string from a DSPy Prediction object.

        Tries self.output_field first, then common field names,
        then falls back to str(prediction).

        Args:
            prediction: DSPy Prediction object or any object.

        Returns:
            String representation of the program's output.
        """
        for field in (self.output_field, "answer", "response", "output", "text"):
            value = getattr(prediction, field, None)
            if value is not None:
                return str(value).strip()
        return str(prediction).strip()
