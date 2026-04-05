"""AutoGen ConversableAgent backend.

Wraps an AutoGen agent as a ModelBackend by initiating a single-turn
conversation between the wrapped agent and a lightweight proxy sender.
Each generate() call starts a fresh conversation so there is no state
leakage between environment steps.

Supports AutoGen v0.2.x (pyautogen) and v0.4.x (autogen-agentchat).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from ..base import GenerationConfig, GenerationResult, ModelBackend, ModelInfo

logger = logging.getLogger(__name__)


class AutoGenBackend(ModelBackend):
    """Wraps an AutoGen ConversableAgent as a ModelBackend.

    Each generate() call initiates a fresh single-turn chat between a
    minimal proxy sender and the wrapped agent, then extracts the last
    assistant message as the response.

    Args:
        agent: An instantiated AutoGen ConversableAgent or AssistantAgent.
        agent_id: Human-readable name. Defaults to agent.name.
        max_turns: Number of conversation turns per generate() call.
            Keep at 1 for single-response environments; increase for
            multi-step reasoning agents.
    """

    def __init__(
        self,
        agent,
        agent_id: Optional[str] = None,
        max_turns: int = 1,
    ) -> None:
        self.agent = agent
        self.agent_id = agent_id or getattr(agent, "name", type(agent).__name__)
        self.max_turns = max_turns

    # ------------------------------------------------------------------
    # ModelBackend interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Send a prompt to the AutoGen agent and return its response.

        Creates a temporary proxy sender for each call to avoid
        conversation state accumulating across environment steps.

        Args:
            prompt: Input message sent to the agent.
            config: Recorded in metadata; AutoGen manages its own LLM config.

        Returns:
            GenerationResult containing the agent's last message.
        """
        start = time.perf_counter()
        text = self._run_single_turn(prompt)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return GenerationResult(
            text=text,
            tokens_used=None,
            latency_ms=latency_ms,
            model_info=self.get_model_info(),
            metadata={"agent_name": self.agent_id, "max_turns": self.max_turns},
        )

    def get_model_info(self) -> ModelInfo:
        """Return metadata identifying this as an AutoGen backend.

        Returns:
            ModelInfo with provider set to "autogen".
        """
        return ModelInfo(
            model_id=self.agent_id,
            provider="autogen",
            is_local=False,
            supports_streaming=False,
            supports_layer_probing=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_single_turn(self, prompt: str) -> str:
        """Initiate a single-turn AutoGen conversation and extract the reply.

        Attempts the AutoGen v0.2 API (initiate_chat) first, then falls
        back to the v0.4 agentchat API.

        Args:
            prompt: Message to send to the agent.

        Returns:
            The agent's response as a plain string.
        """
        try:
            return self._run_v2(prompt)
        except (AttributeError, ImportError):
            pass

        try:
            return self._run_v4(prompt)
        except (AttributeError, ImportError) as exc:
            raise RuntimeError(
                "Could not invoke AutoGen agent — ensure either pyautogen (v0.2) "
                "or autogen-agentchat (v0.4) is installed."
            ) from exc

    def _run_v2(self, prompt: str) -> str:
        """Invoke using AutoGen v0.2 (pyautogen) API.

        Args:
            prompt: Input message.

        Returns:
            Last assistant message text.
        """
        from autogen import UserProxyAgent

        # Silent proxy: human_input_mode="NEVER" prevents blocking on stdin.
        proxy = UserProxyAgent(
            name="isopro_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
        )
        proxy.initiate_chat(
            self.agent,
            message=prompt,
            max_turns=self.max_turns,
            silent=True,
        )
        return _extract_last_message(proxy.chat_messages.get(self.agent, []))

    def _run_v4(self, prompt: str) -> str:
        """Invoke using AutoGen v0.4 (autogen-agentchat) API.

        Args:
            prompt: Input message.

        Returns:
            Last assistant message text.
        """
        import asyncio

        from autogen_agentchat.messages import TextMessage
        from autogen_core import CancellationToken

        async def _async_run():
            response = await self.agent.on_messages(
                [TextMessage(content=prompt, source="user")],
                cancellation_token=CancellationToken(),
            )
            return response.chat_message.content

        return asyncio.run(_async_run())


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_last_message(messages: list[dict]) -> str:
    """Extract the last assistant message from an AutoGen chat history.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        Content of the last assistant message, or empty string if none.
    """
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            return content.strip() if isinstance(content, str) else str(content)
    return ""
