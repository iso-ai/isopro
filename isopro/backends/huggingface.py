"""HuggingFace local model backend with layer probing.

This is the most capable backend — it has direct access to model
internals, enabling two features unavailable in API backends:

  1. Layer activation tracking: records the L2 norm of each
     transformer layer's output during a forward pass. Shows
     which layers are most engaged on a given task.

  2. Gradient sensitivity analysis: runs a backward pass and
     records gradient norms per layer. Shows which weights would
     change most if the model were fine-tuned on this task.

Both are exposed through get_layer_activations() and
get_gradient_sensitivity() respectively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

from .base import GenerationConfig, GenerationResult, ModelBackend, ModelInfo

logger = logging.getLogger(__name__)

# Hooks capture activation norms for layer names matching these substrings.
# Covers most transformer architectures (Llama, Mistral, Qwen, Phi, Gemma).
_LAYER_NAME_PATTERNS = ("layers.", "h.", "blocks.", "transformer.h")


@dataclass
class LayerActivationRecord:
    """Per-layer activation norms from a single forward pass.

    Attributes:
        prompt: The input that produced these activations.
        activations: Mapping of layer name → L2 norm of output tensor.
        top_layers: The 5 most active layers, sorted descending by norm.
    """

    prompt: str
    activations: dict[str, float] = field(default_factory=dict)

    @property
    def top_layers(self) -> list[tuple[str, float]]:
        """Return the 5 most active layers as (name, norm) pairs."""
        return sorted(self.activations.items(), key=lambda x: x[1], reverse=True)[:5]


@dataclass
class GradientSensitivityRecord:
    """Per-layer gradient norms from a backward pass.

    Higher gradient norm → that layer's weights would change most
    during fine-tuning, meaning it's most sensitive to this task.

    Attributes:
        prompt: The input used to compute gradients.
        gradient_norms: Mapping of parameter name → gradient L2 norm.
        top_sensitive: The 5 highest-gradient parameters.
        recommendation: Human-readable fine-tuning suggestion.
    """

    prompt: str
    gradient_norms: dict[str, float] = field(default_factory=dict)

    @property
    def top_sensitive(self) -> list[tuple[str, float]]:
        """Return the 5 most gradient-sensitive parameters."""
        return sorted(
            self.gradient_norms.items(), key=lambda x: x[1], reverse=True
        )[:5]

    @property
    def recommendation(self) -> str:
        """Generate a plain-English fine-tuning recommendation."""
        if not self.top_sensitive:
            return "No gradient data available."
        top_names = [name for name, _ in self.top_sensitive[:3]]
        layer_nums = _extract_layer_numbers(top_names)
        if layer_nums:
            span = f"layers {min(layer_nums)}–{max(layer_nums)}"
        else:
            span = "the most sensitive parameters"
        return (
            f"Fine-tuning {span} is likely to improve performance "
            f"on this task. Consider LoRA targeting: {', '.join(top_names[:2])}."
        )


class HuggingFaceBackend(ModelBackend):
    """Local HuggingFace model backend.

    Model weights are loaded lazily on the first generate() call.
    Forward hooks are registered at load time and remain active
    for the session — activation records are overwritten each call.

    Args:
        model_id: HuggingFace repo ID or local path.
        load_in_4bit: Enable bitsandbytes 4-bit quantization (CUDA only).
        device_map: HuggingFace device placement strategy.
        torch_dtype: Torch dtype override (e.g. torch.float16).
            Inferred automatically if None.
        max_context_length: Override detected context length.
    """

    def __init__(
        self,
        model_id: str,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        torch_dtype=None,
        max_context_length: Optional[int] = None,
    ) -> None:
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_context_length = max_context_length

        # Populated by _ensure_loaded()
        self._model = None
        self._tokenizer = None

        # Activation records — overwritten on each forward pass.
        self._activation_records: dict[str, float] = {}
        self._hook_handles: list = []

    # ------------------------------------------------------------------
    # ModelBackend interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate a response using the local model.

        Args:
            prompt: Input text (system prompt is prepended if set in config).
            config: Generation parameters.

        Returns:
            GenerationResult including text, latency, and token count.
        """
        config = config or GenerationConfig()
        self._ensure_loaded()

        full_prompt = _build_prompt(prompt, config.system_prompt)
        inputs = self._tokenizer(full_prompt, return_tensors="pt").to(
            self._model.device
        )

        import torch

        # Clear previous activation records before this forward pass.
        self._activation_records.clear()

        gen_kwargs = dict(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.temperature > 0.0,
        )
        if config.stop_sequences:
            # HuggingFace uses eos_token_id for stop; we post-process instead.
            gen_kwargs["eos_token_id"] = self._tokenizer.eos_token_id

        def _run():
            with torch.no_grad():
                return self._model.generate(**inputs, **gen_kwargs)

        output_ids, latency_ms = self._timed_call(_run)

        # Decode only the newly generated tokens (not the prompt).
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # Truncate at any stop sequences.
        for stop in config.stop_sequences:
            if stop in text:
                text = text[: text.index(stop)]

        tokens_used = int(inputs["input_ids"].shape[1]) + int(new_ids.shape[0])

        return GenerationResult(
            text=text,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            model_info=self.get_model_info(),
        )

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Stream tokens using HuggingFace TextIteratorStreamer.

        Args:
            prompt: Input text.
            config: Generation parameters.

        Yields:
            Decoded token strings as they are generated.
        """
        import threading

        import torch
        from transformers import TextIteratorStreamer

        config = config or GenerationConfig()
        self._ensure_loaded()

        full_prompt = _build_prompt(prompt, config.system_prompt)
        inputs = self._tokenizer(full_prompt, return_tensors="pt").to(
            self._model.device
        )
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.temperature > 0.0,
        )

        # Generation runs in a background thread so we can yield from the streamer.
        thread = threading.Thread(
            target=lambda: self._model.generate(**gen_kwargs), daemon=True
        )
        thread.start()

        for chunk in streamer:
            yield chunk

        thread.join()

    def get_model_info(self) -> ModelInfo:
        """Return metadata about the loaded model.

        Returns:
            ModelInfo with parameter count and context length if available.
        """
        param_count = None
        context_length = self.max_context_length

        if self._model is not None:
            try:
                param_count = sum(
                    p.numel() for p in self._model.parameters()
                )
            except Exception:
                pass
            if context_length is None:
                context_length = getattr(
                    self._model.config, "max_position_embeddings", None
                )

        return ModelInfo(
            model_id=self.model_id,
            provider="huggingface",
            is_local=True,
            supports_streaming=True,
            supports_layer_probing=True,
            parameter_count=param_count,
            context_length=context_length,
        )

    # ------------------------------------------------------------------
    # Layer probing — HuggingFace-only capabilities
    # ------------------------------------------------------------------

    def get_layer_activations(self, prompt: str) -> LayerActivationRecord:
        """Run a forward pass and return per-layer activation norms.

        Hooks are registered at model load time. This method triggers
        a fresh forward pass (no generation) so activations reflect
        only the prompt encoding.

        Args:
            prompt: Input text to probe.

        Returns:
            LayerActivationRecord mapping layer names to L2 norms.
        """
        import torch

        self._ensure_loaded()
        self._activation_records.clear()

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            self._model(**inputs)

        return LayerActivationRecord(
            prompt=prompt,
            activations=dict(self._activation_records),
        )

    def get_gradient_sensitivity(
        self,
        prompt: str,
        target_text: str = "",
    ) -> GradientSensitivityRecord:
        """Compute gradient norms per parameter for a given prompt.

        Runs a forward pass with gradients enabled, computes cross-entropy
        loss against target_text (or the model's own top prediction if
        target_text is empty), backpropagates, and records gradient norms.

        Higher gradient norm → that parameter is most sensitive to this
        input and would change most under fine-tuning.

        Args:
            prompt: Input text.
            target_text: Target continuation. If empty, uses the model's
                own greedy prediction as the target (self-consistency probe).

        Returns:
            GradientSensitivityRecord with per-parameter gradient norms.
        """
        import torch
        import torch.nn.functional as F

        self._ensure_loaded()

        full_prompt = prompt + (target_text or "")
        inputs = self._tokenizer(full_prompt, return_tensors="pt").to(
            self._model.device
        )

        # Enable gradients for this pass only.
        for param in self._model.parameters():
            param.requires_grad_(True)

        outputs = self._model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()

        grad_norms: dict[str, float] = {}
        for name, param in self._model.named_parameters():
            if param.grad is not None and _is_layer_parameter(name):
                grad_norms[name] = float(param.grad.norm().item())

        # Clean up gradients immediately — don't leave them attached.
        self._model.zero_grad()
        for param in self._model.parameters():
            param.requires_grad_(False)

        return GradientSensitivityRecord(prompt=prompt, gradient_norms=grad_norms)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load model and tokenizer if not already loaded (lazy init)."""
        if self._model is not None:
            return

        logger.info("Loading HuggingFace model '%s'...", self.model_id)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        quant_config = None
        if self.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        dtype = self.torch_dtype
        if dtype is None and not self.load_in_4bit:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quant_config,
            device_map=self.device_map,
            torch_dtype=dtype,
        )
        self._model.eval()
        self._register_activation_hooks()
        logger.info("Model '%s' loaded. Activation hooks registered.", self.model_id)

    def _register_activation_hooks(self) -> None:
        """Attach forward hooks to all qualifying transformer layers.

        Hooks record the L2 norm of each layer's output tensor into
        self._activation_records. They fire on every forward pass
        (both generate() and get_layer_activations()).
        """
        # Remove any previously registered hooks.
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

        for name, module in self._model.named_modules():
            if _is_layer_parameter(name):
                handle = module.register_forward_hook(self._make_activation_hook(name))
                self._hook_handles.append(handle)

        logger.debug(
            "Registered activation hooks on %d modules.", len(self._hook_handles)
        )

    def _make_activation_hook(self, layer_name: str):
        """Return a forward hook closure that records activation norms.

        Args:
            layer_name: The name of the module being hooked.

        Returns:
            A hook function compatible with register_forward_hook.
        """
        import torch

        def hook(module, input, output):
            # Output may be a tensor or a tuple (e.g. attention layers).
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                self._activation_records[layer_name] = float(tensor.norm().item())

        return hook


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _build_prompt(prompt: str, system_prompt: str) -> str:
    """Prepend system prompt to user prompt for models without chat templates.

    Args:
        prompt: User-facing input.
        system_prompt: Optional system instruction.

    Returns:
        Combined prompt string.
    """
    if system_prompt:
        return f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
    return prompt


def _is_layer_parameter(name: str) -> bool:
    """Return True if the module/parameter name looks like a transformer layer.

    Args:
        name: Dotted module or parameter name from named_modules().

    Returns:
        True if the name matches a known transformer layer pattern.
    """
    return any(pattern in name for pattern in _LAYER_NAME_PATTERNS)


def _extract_layer_numbers(names: list[str]) -> list[int]:
    """Extract integer layer indices from a list of parameter names.

    Args:
        names: List of dotted parameter names.

    Returns:
        Sorted list of unique layer numbers found.
    """
    import re

    nums: set[int] = set()
    for name in names:
        for match in re.finditer(r"\.(\d+)\.", name):
            nums.add(int(match.group(1)))
    return sorted(nums)
