"""Conversation simulation environment.

Upgraded from the original to:
  - Accept any ModelBackend instead of hardcoding Claude.
  - Track per-turn metrics (sentiment, response length, coherence).
  - Return structured EpisodeResult via run_episode().
  - Retain full backwards compatibility with set_ai_agent(model=...) callers.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..backends.base import GenerationConfig, ModelBackend
from ..environments.base_env import BaseEnvironment
from .user_personas import UserPersona

logger = logging.getLogger(__name__)


class ConversationEnvironment(BaseEnvironment):
    """Environment for simulating multi-turn conversations.

    Supports two construction patterns:

    Pattern 1 — ModelBackend (recommended):
        env = ConversationEnvironment(backend=ModelBackend.from_api("gpt-4o", provider="openai"))

    Pattern 2 — Legacy string model name (backwards compatible):
        env = ConversationEnvironment()
        env.set_ai_agent(model="claude-3-opus-20240229")

    Args:
        backend: A ModelBackend instance. If None, set_ai_agent() must
            be called before run_episode().
        ai_prompt: System prompt defining the AI's persona/role.
        max_turns: Number of conversation turns per episode.
    """

    def __init__(
        self,
        backend: Optional[ModelBackend] = None,
        ai_prompt: str = (
            "You are a helpful customer service agent. "
            "Respond politely and professionally."
        ),
        max_turns: int = 5,
    ) -> None:
        super().__init__(backend=backend)
        self.ai_prompt = ai_prompt
        self.max_turns = max_turns

        self._user_persona: Optional[UserPersona] = None
        self._conversation_history: list[dict] = []
        self._turn: int = 0
        self._per_turn_metrics: list[dict] = []

    # ------------------------------------------------------------------
    # Legacy API — backwards compatible with original callers
    # ------------------------------------------------------------------

    def set_ai_agent(self, model: str = "claude-3-opus-20240229") -> None:
        """Set up an Anthropic backend from a model name string.

        Kept for backwards compatibility. New code should pass a
        ModelBackend to the constructor instead.

        Args:
            model: Claude model ID string.
        """
        self.backend = ModelBackend.from_api(model, provider="anthropic")
        logger.info("Set AI agent via legacy API: model=%s", model)

    def set_user_persona(self, persona_type: str, **kwargs) -> None:
        """Set the user persona for the conversation.

        Args:
            persona_type: Persona type string (e.g. "upset", "human_request").
            **kwargs: Additional arguments forwarded to UserPersona.create().
        """
        self._user_persona = UserPersona.create(persona_type, **kwargs)
        logger.info("Set user persona: %s", persona_type)

    def run_conversation(self, num_turns: int = 5) -> list[dict]:
        """Run a conversation and return the history.

        Legacy entry point. New code should use run_episode() instead,
        which returns structured EpisodeResult with metrics.

        Args:
            num_turns: Number of turns to simulate.

        Returns:
            List of {"role": ..., "content": ...} dicts.
        """
        if not self.backend or not self._user_persona:
            raise ValueError(
                "Both a backend (or model via set_ai_agent) and a user persona "
                "must be set before running a conversation."
            )
        self.max_turns = num_turns
        result = self.run_episode(max_steps=num_turns)
        return self._conversation_history

    # ------------------------------------------------------------------
    # BaseEnvironment interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple:
        """Reset conversation state.

        Returns:
            Tuple of (empty observation dict, info dict).
        """
        self._conversation_history = []
        self._turn = 0
        self._per_turn_metrics = []
        return {}, {}

    def step(self, action: str) -> tuple:
        """Execute one conversation turn.

        The action is the user's message. The environment generates the
        AI response using the backend, records it, and computes turn metrics.

        Args:
            action: The user's message text for this turn.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if not self.backend:
            raise RuntimeError(
                "No backend configured. Pass a ModelBackend to the constructor "
                "or call set_ai_agent() first."
            )

        self._turn += 1
        self._conversation_history.append({"role": "user", "content": action})

        # Build prompt from full conversation history.
        config = GenerationConfig(
            max_new_tokens=512,
            system_prompt=self.ai_prompt,
        )
        prompt = _format_history_as_prompt(self._conversation_history)
        result = self.backend.generate(prompt, config)
        ai_response = result.text

        self._conversation_history.append(
            {"role": "assistant", "content": ai_response}
        )

        turn_metrics = _compute_turn_metrics(action, ai_response, self._conversation_history)
        self._per_turn_metrics.append(turn_metrics)
        reward = _turn_reward(turn_metrics)

        terminated = self._turn >= self.max_turns
        obs = {
            "turn": self._turn,
            "user_message": action,
            "ai_response": ai_response,
        }
        info = {
            "text_output": ai_response,
            "turn_metrics": turn_metrics,
            "latency_ms": result.latency_ms,
        }
        return obs, reward, terminated, False, info

    def _get_action(self, observation, config=None) -> str:
        """Generate the next user message from the configured persona.

        Args:
            observation: Current conversation state (partially used).
            config: Ignored — persona has its own generation logic.

        Returns:
            User message string from the persona.
        """
        if not self._user_persona:
            return "Hello, can you help me?"
        return self._user_persona.generate_message(self._conversation_history)

    def compute_metrics(self) -> dict:
        """Aggregate per-turn metrics across the full conversation.

        Returns:
            Dict with mean sentiment, mean coherence, and response length stats.
        """
        if not self._per_turn_metrics:
            return {}

        sentiments = [m.get("sentiment", 0.0) for m in self._per_turn_metrics]
        coherences = [m.get("coherence", 0.0) for m in self._per_turn_metrics]
        response_lengths = [m.get("response_length", 0) for m in self._per_turn_metrics]

        return {
            "turns_completed": self._turn,
            "mean_sentiment": float(np.mean(sentiments)),
            "mean_coherence": float(np.mean(coherences)),
            "mean_response_length": float(np.mean(response_lengths)),
            "min_response_length": int(min(response_lengths)) if response_lengths else 0,
            "max_response_length": int(max(response_lengths)) if response_lengths else 0,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _format_history_as_prompt(history: list[dict]) -> str:
    """Format conversation history as a string prompt for the backend.

    Args:
        history: List of {"role": ..., "content": ...} dicts.

    Returns:
        Multi-line prompt string ending with "Assistant:".
    """
    lines = []
    for msg in history[:-1]:  # Exclude the last user message (it's the prompt).
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")

    # The last message is always the current user turn.
    last = history[-1]
    lines.append(f"User: {last['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _compute_turn_metrics(
    user_message: str,
    ai_response: str,
    history: list[dict],
) -> dict:
    """Compute quality metrics for a single conversation turn.

    Args:
        user_message: The user's message this turn.
        ai_response: The AI's response this turn.
        history: Full conversation history including this turn.

    Returns:
        Dict with sentiment, coherence, and length metrics.
    """
    metrics: dict = {
        "response_length": len(ai_response),
        "user_message_length": len(user_message),
        "sentiment": 0.0,
        "coherence": 0.0,
    }

    # Sentiment via TextBlob if available.
    try:
        from textblob import TextBlob
        metrics["sentiment"] = float(TextBlob(ai_response).sentiment.polarity)
    except ImportError:
        pass

    # Coherence: TF-IDF cosine similarity between user message and AI response.
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        if len(history) >= 2:
            vectorizer = TfidfVectorizer()
            docs = [user_message, ai_response]
            matrix = vectorizer.fit_transform(docs)
            similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
            metrics["coherence"] = float(similarity)
    except ImportError:
        pass

    return metrics


def _turn_reward(metrics: dict) -> float:
    """Convert turn metrics into a scalar reward.

    Args:
        metrics: Output of _compute_turn_metrics().

    Returns:
        Scalar reward in approximately [-1, 1].
    """
    sentiment = metrics.get("sentiment", 0.0)    # [-1, 1]
    coherence = metrics.get("coherence", 0.5)    # [0, 1]
    length = metrics.get("response_length", 0)

    # Penalize extremely short or extremely long responses.
    length_score = 1.0 if 50 <= length <= 400 else max(0.0, 1.0 - abs(length - 200) / 400.0)

    return (sentiment * 0.4) + (coherence * 0.4) + (length_score * 0.2)
