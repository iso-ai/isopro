# isopro/__init__.py

"""
isopro: Intelligent Simulation Orchestration for LLMs

This package provides tools for creating, managing, and analyzing simulations
involving Large Language Models (LLMs), including reinforcement learning,
conversation simulations, and adversarial testing.

Top-level imports are *soft* — submodules with optional heavy dependencies
(adversarial attacks, RL, orchestration, conversation backends) are imported
in try/except blocks so that consumers who only need a subset of the package
can `import isopro` (or any specific submodule) without installing every
optional dependency. Missing names are simply absent from the namespace.
"""

from __future__ import annotations

import warnings as _warnings

__version__ = "0.3.2"

__all__: list[str] = []


def _try_import(import_stmt: str, names: list[str]) -> None:
    """Execute ``from .X import Y, Z`` and append exposed names on success.

    Args:
        import_stmt: A ``from .module import Name, Other`` style statement.
        names: Names to add to ``__all__`` if the import succeeds.
    """
    try:
        exec(import_stmt, globals())
    except (ImportError, ModuleNotFoundError) as exc:
        _warnings.warn(
            f"isopro: optional submodule unavailable ({exc.__class__.__name__}: {exc}). "
            f"Skipping {names}.",
            stacklevel=2,
        )
        return
    __all__.extend(names)


_try_import(
    "from .environments.simulation_environment import SimulationEnvironment",
    ["SimulationEnvironment"],
)
_try_import(
    "from .environments.custom_environment import CustomEnvironment",
    ["CustomEnvironment"],
)
_try_import(
    "from .environments.llm_orchestrator import LLMOrchestrator",
    ["LLMOrchestrator"],
)
_try_import("from .agents.ai_agent import AI_Agent", ["AI_Agent"])
_try_import("from .base.base_component import BaseComponent", ["BaseComponent"])
_try_import(
    "from .wrappers.simulation_wrapper import SimulationWrapper",
    ["SimulationWrapper"],
)
_try_import("from .rl.rl_environment import BaseRLEnvironment", ["BaseRLEnvironment"])
_try_import("from .rl.rl_agent import RLAgent", ["RLAgent"])
_try_import(
    "from .conversation_simulation import "
    "ConversationSimulator, ConversationEnvironment, ConversationAgent",
    ["ConversationSimulator", "ConversationEnvironment", "ConversationAgent"],
)
_try_import(
    "from .adversarial_simulation import "
    "AdversarialSimulator, AdversarialEnvironment, AdversarialAgent",
    ["AdversarialSimulator", "AdversarialEnvironment", "AdversarialAgent"],
)
_try_import(
    "from .orchestration_simulation import "
    "LLaMAAgent, SubAgent, OrchestrationEnv, AI_AgentException, ComponentException",
    ["LLaMAAgent", "SubAgent", "OrchestrationEnv", "AI_AgentException", "ComponentException"],
)
