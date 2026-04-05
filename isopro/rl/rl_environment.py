"""RL environments for ISOPro.

Fixes over the original:
  - LLMRLEnvironment now uses ModelBackend instead of a hardcoded Anthropic client.
  - Observation space is a real 384-dim sentence embedding (via MiniLM) so the
    RL agent receives meaningful state — not random noise.
  - evaluate_persona_adherence() parses the score robustly with a regex fallback.
  - get_human_feedback() is replaced by a real response-quality signal derived
    from response length, coherence, and task relevance.
  - GymRLEnvironment is unchanged — it was already correct.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..backends.base import GenerationConfig, ModelBackend
from ..environments.base_env import BaseEnvironment, EpisodeResult

logger = logging.getLogger(__name__)

# Embedding dimension for MiniLM-L6-v2. If sentence-transformers is not
# available, we fall back to a handcrafted 16-dim feature vector.
_EMBED_DIM = 384
_FALLBACK_OBS_DIM = 16


class LLMRLEnvironment(BaseEnvironment, gym.Env):
    """RL environment driven by a language model via ModelBackend.

    At each step:
      1. The observation (encoded text embedding) is passed to _get_action().
      2. The action (a discrete index 0-4) is converted to a natural-language
         prompt fragment and sent to the backend.
      3. The response is embedded into the next observation.
      4. Reward is computed from response quality signals (persona adherence,
         coherence, length appropriateness) — not random noise.

    Args:
        backend: Any ModelBackend instance.
        agent_prompt: System prompt defining the agent's persona/task.
        max_steps: Episode length before truncation.
        config: GenerationConfig applied to every backend call.
    """

    # Maps discrete action index → natural language instruction fragment.
    _ACTION_LABELS = {
        0: "Respond directly and concisely.",
        1: "Ask a clarifying question.",
        2: "Provide a detailed explanation.",
        3: "Summarize the conversation so far.",
        4: "Acknowledge and redirect the conversation.",
    }

    def __init__(
        self,
        backend: ModelBackend,
        agent_prompt: str = "You are a helpful assistant.",
        max_steps: int = 10,
        config: Optional[GenerationConfig] = None,
    ) -> None:
        BaseEnvironment.__init__(self, backend=backend)

        self.agent_prompt = agent_prompt
        self.max_steps = max_steps
        self.config = config or GenerationConfig(
            max_new_tokens=256,
            system_prompt=agent_prompt,
        )

        self.action_space = spaces.Discrete(len(self._ACTION_LABELS))
        obs_dim = _get_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._current_step: int = 0
        self._conversation_history: list[dict] = []
        self._last_response: str = ""
        self._embedder = _load_embedder()

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None) -> tuple:
        """Reset conversation history and return the initial observation.

        Returns:
            Tuple of (initial_observation, info_dict).
        """
        if hasattr(super(), "reset"):
            try:
                gym.Env.reset(self, seed=seed)
            except TypeError:
                pass

        self._current_step = 0
        self._conversation_history = []
        self._last_response = ""
        obs = _encode_text(self.agent_prompt, self._embedder)
        return obs, {}

    def step(self, action: int) -> tuple:
        """Execute one conversation step.

        Args:
            action: Discrete action index (0-4) selecting a response strategy.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        self._current_step += 1
        action_instruction = self._ACTION_LABELS.get(int(action), self._ACTION_LABELS[0])

        # Build the prompt from history + action instruction.
        history_text = _format_history(self._conversation_history)
        prompt = (
            f"{history_text}\n\n"
            f"[Instruction: {action_instruction}]\n"
            f"Assistant:"
        )

        result = self.backend.generate(prompt, self.config)
        response = result.text
        self._last_response = response
        self._conversation_history.append(
            {"role": "assistant", "content": response}
        )

        reward = self._compute_reward(response, action_instruction)
        observation = _encode_text(response, self._embedder)
        terminated = self._current_step >= self.max_steps
        info = {"text_output": response, "action_label": action_instruction}

        logger.debug(
            "Step %d | action=%d | reward=%.3f | response_len=%d",
            self._current_step,
            action,
            reward,
            len(response),
        )
        return observation, reward, terminated, False, info

    # ------------------------------------------------------------------
    # BaseEnvironment interface
    # ------------------------------------------------------------------

    def _get_action(self, observation, config=None) -> int:
        """Default policy: cycle through actions round-robin.

        Override this in subclasses with a trained RL policy.

        Args:
            observation: Current text embedding (unused by default policy).
            config: Ignored; config is set at construction time.

        Returns:
            An integer action index.
        """
        return self._current_step % len(self._ACTION_LABELS)

    def compute_metrics(self) -> dict:
        """Compute summary metrics for the completed episode.

        Returns:
            Dict with step count, mean reward, and response quality stats.
        """
        rewards = [s.reward for s in self._step_results]
        response_lengths = [
            len(s.text_output or "") for s in self._step_results
        ]
        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "reward_std": float(np.std(rewards)) if rewards else 0.0,
            "mean_response_length": float(np.mean(response_lengths)) if response_lengths else 0.0,
            "steps_completed": len(self._step_results),
        }

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(self, response: str, action_instruction: str) -> float:
        """Compute a multi-component reward from the model's response.

        Components:
          - Persona adherence score (via a backend call — cheap 50-token eval)
          - Length appropriateness (penalize too-short or too-long responses)
          - Coherence with conversation history (cosine similarity of embeddings)

        Args:
            response: The model's text response this step.
            action_instruction: The instruction that prompted this response.

        Returns:
            Scalar reward in approximately [-1, 2].
        """
        adherence = self._evaluate_persona_adherence(response)
        length_score = _length_reward(response)
        coherence = self._evaluate_coherence(response)
        total = adherence + length_score * 0.3 + coherence * 0.3
        return float(total)

    def _evaluate_persona_adherence(self, response: str) -> float:
        """Ask the backend to score how well the response fits the persona.

        Uses a short, structured prompt to elicit a numeric score.
        Falls back to 0.5 if the response cannot be parsed.

        Args:
            response: The response to evaluate.

        Returns:
            Float in [0, 1].
        """
        eval_prompt = (
            f"Rate from 0.0 to 1.0 how well this response adheres to the persona: "
            f'"{self.agent_prompt}"\n\n'
            f'Response: "{response}"\n\n'
            f"Reply with only a number between 0.0 and 1.0."
        )
        try:
            eval_config = GenerationConfig(max_new_tokens=10, temperature=0.0)
            result = self.backend.generate(eval_prompt, eval_config)
            return _parse_score(result.text, default=0.5)
        except Exception as exc:
            logger.debug("Persona adherence eval failed: %s", exc)
            return 0.5

    def _evaluate_coherence(self, response: str) -> float:
        """Measure coherence as cosine similarity to the conversation history.

        Args:
            response: The response to evaluate.

        Returns:
            Float in [0, 1] — 1.0 means perfectly on-topic.
        """
        if not self._conversation_history or self._embedder is None:
            return 0.5

        history_text = _format_history(self._conversation_history[:-1])
        if not history_text.strip():
            return 0.5

        hist_emb = _encode_text(history_text, self._embedder)
        resp_emb = _encode_text(response, self._embedder)

        # Cosine similarity.
        denom = (np.linalg.norm(hist_emb) * np.linalg.norm(resp_emb))
        if denom == 0:
            return 0.5
        return float(np.dot(hist_emb, resp_emb) / denom)


class GymRLEnvironment(BaseEnvironment, gym.Env):
    """Wrapper for standard Gymnasium environments.

    Passes through all gym.Env calls unchanged. Adds run_episode()
    from BaseEnvironment so it fits the ISOPro evaluation protocol.

    Args:
        env_name: Gymnasium environment ID (e.g. "CartPole-v1").
    """

    def __init__(self, env_name: str) -> None:
        BaseEnvironment.__init__(self, backend=None)
        self._gym_env = gym.make(env_name)
        self.action_space = self._gym_env.action_space
        self.observation_space = self._gym_env.observation_space
        logger.info("Initialized GymRLEnvironment with '%s'", env_name)

    def reset(self, seed=None, options=None) -> tuple:
        """Reset the wrapped gym environment.

        Returns:
            Tuple of (observation, info).
        """
        return self._gym_env.reset(seed=seed, options=options)

    def step(self, action) -> tuple:
        """Step the wrapped gym environment.

        Args:
            action: Action in the environment's action space.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        return self._gym_env.step(action)

    def _get_action(self, observation, config=None):
        """Default policy: random action from the action space.

        Args:
            observation: Ignored by the random policy.
            config: Ignored.

        Returns:
            A random action sampled from the action space.
        """
        return self._gym_env.action_space.sample()

    def compute_metrics(self) -> dict:
        """Compute summary metrics for the completed gym episode.

        Returns:
            Dict with total and mean reward.
        """
        rewards = [s.reward for s in self._step_results]
        return {
            "total_reward": float(sum(rewards)),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        }

    def render(self, mode: str = "human"):
        """Render the wrapped gym environment."""
        return self._gym_env.render()

    def close(self):
        """Close the wrapped gym environment."""
        return self._gym_env.close()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _load_embedder():
    """Load the MiniLM sentence embedder if available.

    Returns:
        SentenceTransformer model, or None if not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading MiniLM-L6-v2 sentence embedder.")
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. Using fallback feature vector. "
            "Install with: pip install sentence-transformers"
        )
        return None


def _get_obs_dim() -> int:
    """Return the observation dimensionality based on available libraries.

    Returns:
        384 if sentence-transformers is available, else _FALLBACK_OBS_DIM.
    """
    try:
        import sentence_transformers  # noqa: F401
        return _EMBED_DIM
    except ImportError:
        return _FALLBACK_OBS_DIM


def _encode_text(text: str, embedder) -> np.ndarray:
    """Encode text into a fixed-dim numpy observation vector.

    Uses sentence-transformers if available; otherwise extracts a
    handcrafted 16-dim feature vector from text statistics.

    Args:
        text: Input text to encode.
        embedder: SentenceTransformer model or None.

    Returns:
        Float32 numpy array.
    """
    if embedder is not None:
        return embedder.encode(text, convert_to_numpy=True).astype(np.float32)
    return _fallback_features(text)


def _fallback_features(text: str) -> np.ndarray:
    """Extract a 16-dim feature vector from text without embeddings.

    Features: token count, char count, avg word length, sentence count,
    question mark count, exclamation count, digit fraction,
    uppercase fraction, and 8 character n-gram frequency features.

    Args:
        text: Input text.

    Returns:
        Float32 numpy array of shape (16,).
    """
    words = text.split()
    n = max(len(words), 1)
    chars = max(len(text), 1)
    sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)

    features = np.array([
        min(n / 100.0, 1.0),                           # normalized token count
        min(chars / 500.0, 1.0),                        # normalized char count
        sum(len(w) for w in words) / (n * 10.0),        # avg word length (norm)
        min(sentences / 10.0, 1.0),                     # sentence count (norm)
        text.count("?") / n,                            # question density
        text.count("!") / n,                            # exclamation density
        sum(c.isdigit() for c in text) / chars,         # digit fraction
        sum(c.isupper() for c in text) / chars,         # uppercase fraction
        # 8 char-level n-gram presence features (hashed)
        *[float((hash(text[i:i+2]) % 100) / 100.0)
          for i in range(0, min(len(text), 16), 2)],
    ], dtype=np.float32)

    # Pad or truncate to exactly _FALLBACK_OBS_DIM
    if len(features) < _FALLBACK_OBS_DIM:
        features = np.pad(features, (0, _FALLBACK_OBS_DIM - len(features)))
    return features[:_FALLBACK_OBS_DIM]


def _format_history(history: list[dict]) -> str:
    """Format conversation history as a readable string.

    Args:
        history: List of {"role": ..., "content": ...} dicts.

    Returns:
        Multi-line string with role prefixes.
    """
    lines = []
    for msg in history:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _parse_score(text: str, default: float = 0.5) -> float:
    """Robustly parse a float score from a model response.

    Tries direct float conversion, then regex for any decimal number.
    Clamps to [0, 1].

    Args:
        text: Raw model response text.
        default: Value to return if parsing fails.

    Returns:
        Float in [0.0, 1.0].
    """
    text = text.strip()
    try:
        return min(max(float(text), 0.0), 1.0)
    except ValueError:
        pass

    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        try:
            raw = float(match.group(1))
            # Handle "85" style percentages.
            return min(max(raw / 100.0 if raw > 1.0 else raw, 0.0), 1.0)
        except ValueError:
            pass

    return default


# ---------------------------------------------------------------------------
# Backwards compatibility alias
# ---------------------------------------------------------------------------

#: Alias retained so existing imports of BaseRLEnvironment continue to work.
#: New code should use BaseEnvironment directly.
BaseRLEnvironment = BaseEnvironment


def _length_reward(response: str) -> float:
    """Score response length on a bell curve centered at 100 chars.

    Too short (< 20 chars) or too long (> 500 chars) are penalized.

    Args:
        response: The model's response text.

    Returns:
        Float in [-1, 1].
    """
    n = len(response)
    if n < 20:
        return -1.0
    if n > 500:
        return max(-1.0, 1.0 - (n - 500) / 500.0)
    # Peak reward at ~100 chars, falls off symmetrically.
    return 1.0 - abs(n - 100) / 400.0
