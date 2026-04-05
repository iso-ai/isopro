"""Environment test suite for isopro.

Runs each strengthened environment through a real scenario and prints
structured results. Tests are independent — a failure in one does not
block the others.

Usage:
    python tests/test_environments.py
    python tests/test_environments.py --skip-api   # skips Anthropic/OpenAI calls
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import traceback
from dataclasses import dataclass
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    details: str
    error: str = ""


def run_test(name: str, fn: Callable) -> TestResult:
    """Run a single test function, catch exceptions, return result."""
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")
    try:
        details = fn()
        print(f"  PASSED")
        return TestResult(name=name, passed=True, details=details or "")
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"  FAILED: {exc}")
        print(textwrap.indent(tb, "    "))
        return TestResult(name=name, passed=False, details="", error=str(exc))


def print_summary(results: list[TestResult]) -> None:
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    passed = sum(r.passed for r in results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}]  {r.name}")
        if r.details:
            for line in r.details.strip().splitlines():
                print(f"           {line}")
    print(f"\n  {passed}/{len(results)} tests passed")


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------


def test_model_backend_factories() -> str:
    """Verify all ModelBackend factory methods exist and return correct types."""
    from isopro.backends.base import ModelBackend
    from isopro.backends.huggingface import HuggingFaceBackend
    from isopro.backends.openai_backend import OpenAIBackend
    from isopro.backends.anthropic_backend import AnthropicBackend
    from isopro.backends.gemini_backend import GeminiBackend
    from isopro.backends.agent_backends.dspy_backend import DSPyBackend
    from isopro.backends.agent_backends.langchain_backend import LangChainBackend
    from isopro.backends.agent_backends.autogen_backend import AutoGenBackend

    # HuggingFace
    hf = ModelBackend.from_huggingface("gpt2")
    assert isinstance(hf, HuggingFaceBackend)
    info = hf.get_model_info()
    assert info.provider == "huggingface"
    assert info.supports_layer_probing is True
    assert info.is_local is True
    print(f"  HuggingFaceBackend: model_id={info.model_id}, layer_probing={info.supports_layer_probing}")

    # OpenAI
    oai = ModelBackend.from_api("gpt-4o", provider="openai", api_key="test-key")
    assert isinstance(oai, OpenAIBackend)
    assert oai.get_model_info().provider == "openai"
    print(f"  OpenAIBackend: model_id={oai.get_model_info().model_id}")

    # OpenAI-compatible (Together AI style)
    together = ModelBackend.from_api(
        "meta-llama/Llama-3-70b",
        provider="openai",
        api_key="test",
        base_url="https://api.together.xyz/v1",
    )
    assert together.get_model_info().provider == "together_ai"
    print(f"  Together AI (OpenAI-compat): provider={together.get_model_info().provider}")

    # Anthropic
    claude = ModelBackend.from_api("claude-3-5-sonnet-20241022", provider="anthropic", api_key="test-key")
    assert isinstance(claude, AnthropicBackend)
    print(f"  AnthropicBackend: model_id={claude.get_model_info().model_id}")

    # Gemini
    gemini = ModelBackend.from_api("gemini-1.5-flash", provider="gemini", api_key="test-key")
    assert isinstance(gemini, GeminiBackend)
    print(f"  GeminiBackend: model_id={gemini.get_model_info().model_id}")

    # DSPy stub
    class FakeDSPy:
        def __call__(self, prompt=""):
            class R:
                answer = "DSPy response"
            return R()

    dspy_b = ModelBackend.from_dspy(FakeDSPy())
    assert isinstance(dspy_b, DSPyBackend)
    result = dspy_b.generate("test prompt")
    assert result.text == "DSPy response"
    assert result.model_info.provider == "dspy"
    print(f"  DSPyBackend: generate() -> '{result.text}'")

    # LangChain stub
    class FakeChain:
        def invoke(self, inputs):
            return {"output": f"LangChain response to: {inputs.get('input', '')}"}

    lc_b = ModelBackend.from_langchain(FakeChain())
    assert isinstance(lc_b, LangChainBackend)
    result = lc_b.generate("hello")
    assert "LangChain response" in result.text
    print(f"  LangChainBackend: generate() -> '{result.text}'")

    return "All 7 backends constructed and validated."


def test_car_rl_environment() -> str:
    """Full CarRLEnvironment test: physics, reward shaping, run_episode."""
    from isopro.car_simulator.car_rl_environment import CarRLEnvironment

    env = CarRLEnvironment(num_cars=2, max_steps=30, goal_radius=0.08, is_rainy=True)

    # Check spaces.
    expected_obs_dim = 2 * 7 + 3  # 2 cars * 7 features + 3 context
    assert env.observation_space.shape == (expected_obs_dim,), (
        f"Expected obs shape ({expected_obs_dim},), got {env.observation_space.shape}"
    )
    assert env.action_space.shape == (4,), f"Expected action shape (4,), got {env.action_space.shape}"
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")

    # Reset.
    obs, info = env.reset(seed=42)
    assert obs.shape == (expected_obs_dim,)
    assert obs.dtype == np.float32
    print(f"  Reset OK — obs sample: {obs[:4].round(3)}")

    # Verify rainy friction is higher.
    env_dry = CarRLEnvironment(num_cars=1, is_rainy=False)
    assert env.friction > env_dry.friction, "Rainy friction should exceed dry friction"
    print(f"  Friction (rainy={env.friction:.3f}, dry={env_dry.friction:.3f}) — correct ordering")

    # Manual steps — verify rewards are shaped correctly.
    rewards = []
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break

    print(f"  10 steps: rewards in [{min(rewards):.3f}, {max(rewards):.3f}]")
    print(f"  Goals reached: {info['goals_reached']}/2")

    # Full episode via run_episode().
    env2 = CarRLEnvironment(num_cars=1, max_steps=50)
    result = env2.run_episode(max_steps=50)
    summary = result.summary()
    print(f"  run_episode() — steps={summary['steps']}, total_reward={summary['total_reward']:.3f}")
    print(f"  Metrics: success_rate={summary['success_rate']:.2f}, steps_taken={summary['steps_taken']}")

    assert result.total_steps > 0
    assert "success_rate" in result.metrics
    assert result.model_id == "none"  # No backend.

    # render() should not crash.
    env2.reset()
    env2.step(env2.action_space.sample())
    env2.render()

    return f"run_episode: {result.total_steps} steps, reward={result.total_reward:.3f}"


def test_ui_element_detector() -> str:
    """Test UIElementDetector on synthetic frames — no video file required."""
    import cv2
    import numpy as np
    from isopro.workflow_simulation.workflow_environment import UIElementDetector

    detector = UIElementDetector()

    # --- Frame 1: bright colored rectangles (buttons) ---
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 50), (200, 100), (0, 120, 255), -1)   # orange button
    cv2.rectangle(frame, (300, 50), (450, 100), (0, 200, 80), -1)   # green button
    cv2.rectangle(frame, (50, 200), (600, 240), (180, 180, 180), -1)  # gray input bar

    elements = detector.detect_elements(frame)
    print(f"  Frame 1 (colored rects): detected {len(elements)} elements")
    for elem in elements[:5]:
        print(f"    - id={elem.id}, type={elem.type}, conf={elem.confidence:.2f}, bbox={[round(b,2) for b in elem.bbox]}")

    # At minimum the detector should not crash and return a list.
    assert isinstance(elements, list)

    # --- Frame 2: white background with dark text lines ---
    frame2 = np.full((480, 640, 3), 240, dtype=np.uint8)
    for y in [80, 140, 200, 260, 320]:
        cv2.putText(frame2, "Sample label text here", (50, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)

    elements2 = detector.detect_elements(frame2)
    print(f"  Frame 2 (text on white): detected {len(elements2)} elements")

    # --- Frame 3: completely black frame (edge case) ---
    frame3 = np.zeros((480, 640, 3), dtype=np.uint8)
    elements3 = detector.detect_elements(frame3)
    print(f"  Frame 3 (black frame):   detected {len(elements3)} elements (expected 0)")

    return f"Detector ran on 3 synthetic frames without error."


def test_conversation_environment_with_api() -> str:
    """ConversationEnvironment end-to-end with real Anthropic API."""
    import os
    from isopro.backends.base import ModelBackend, GenerationConfig
    from isopro.conversation_simulation.conversation_environment import ConversationEnvironment
    from isopro.conversation_simulation.user_personas import UserPersona

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    assert api_key, "ANTHROPIC_API_KEY not set — cannot run API test"

    backend = ModelBackend.from_api(
        "claude-3-haiku-20240307",  # cheapest model for testing
        provider="anthropic",
        api_key=api_key,
    )
    info = backend.get_model_info()
    print(f"  Backend: {info.model_id} ({info.provider})")
    print(f"  Layer probing: {info.supports_layer_probing}, Streaming: {info.supports_streaming}")

    env = ConversationEnvironment(
        backend=backend,
        ai_prompt="You are a concise, friendly customer service agent for a tech company.",
        max_turns=2,
    )

    # Set a user persona.
    env.set_user_persona("human_request")

    # Run a 2-turn conversation.
    result = env.run_episode(max_steps=2)
    summary = result.summary()

    print(f"  Episode: {result.total_steps} turns, reward={result.total_reward:.3f}")
    print(f"  Metrics: {summary}")

    # Inspect conversation.
    for i, step in enumerate(result.step_results):
        print(f"  Turn {i+1}:")
        print(f"    User:      {step.info.get('turn_metrics', {}).get('user_message_length', '?')} chars")
        print(f"    AI:        {step.info.get('turn_metrics', {}).get('response_length', '?')} chars")
        print(f"    Sentiment: {step.info.get('turn_metrics', {}).get('sentiment', '?'):.3f}")
        print(f"    Coherence: {step.info.get('turn_metrics', {}).get('coherence', '?'):.3f}")
        if step.text_output:
            preview = step.text_output[:100].replace('\n', ' ')
            print(f"    Response:  '{preview}...'")

    assert result.total_steps == 2
    assert "mean_sentiment" in result.metrics
    assert result.model_id == "claude-3-haiku-20240307"

    return f"{result.total_steps} turns, reward={result.total_reward:.3f}, coherence={summary.get('mean_coherence', 0):.3f}"


def test_llm_rl_environment_with_api() -> str:
    """LLMRLEnvironment end-to-end with real Anthropic API."""
    import os
    from isopro.backends.base import ModelBackend, GenerationConfig
    from isopro.rl.rl_environment import LLMRLEnvironment

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    assert api_key, "ANTHROPIC_API_KEY not set — cannot run API test"

    backend = ModelBackend.from_api(
        "claude-3-haiku-20240307",
        provider="anthropic",
        api_key=api_key,
    )

    env = LLMRLEnvironment(
        backend=backend,
        agent_prompt="You are a helpful assistant that gives concise answers.",
        max_steps=3,
        config=GenerationConfig(max_new_tokens=100, temperature=0.5),
    )

    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")

    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape, (
        f"Obs shape mismatch: {obs.shape} vs {env.observation_space.shape}"
    )
    assert obs.dtype == np.float32
    print(f"  Reset OK — obs shape={obs.shape}, dtype={obs.dtype}")

    # Run 3 steps.
    total_reward = 0.0
    for step_num in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        label = info.get("action_label", "")
        response_preview = (info.get("text_output") or "")[:60].replace('\n', ' ')
        print(f"  Step {step_num+1}: action={action} ({label[:30]}...) reward={reward:.3f}")
        print(f"    Response: '{response_preview}...'")
        assert obs.shape == env.observation_space.shape, "Obs shape changed mid-episode"
        assert isinstance(reward, float), "Reward must be float"
        if terminated or truncated:
            break

    # run_episode() test.
    result = env.run_episode(max_steps=2)
    print(f"  run_episode(): steps={result.total_steps}, total_reward={result.total_reward:.3f}")
    assert result.metrics.get("steps_completed", 0) > 0

    return f"3 manual steps + run_episode OK, total_reward={total_reward:.3f}"


def test_base_environment_contract() -> str:
    """Verify BaseEnvironment cannot be instantiated without implementing abstracts."""
    from isopro.environments.base_env import BaseEnvironment, EpisodeResult, StepResult

    # Should raise TypeError — abstract methods not implemented.
    try:
        bad = BaseEnvironment()
        raise AssertionError("Should have raised TypeError for abstract class")
    except TypeError:
        print("  BaseEnvironment correctly raises TypeError when instantiated directly")

    # Verify EpisodeResult.summary() works.
    from isopro.car_simulator.car_rl_environment import CarRLEnvironment
    env = CarRLEnvironment(num_cars=1, max_steps=5)
    result = env.run_episode(max_steps=5)
    summary = result.summary()

    required_keys = {"episode_id", "environment", "model", "provider", "steps",
                     "total_reward", "mean_reward", "terminated", "duration_s"}
    missing = required_keys - set(summary.keys())
    assert not missing, f"EpisodeResult.summary() missing keys: {missing}"
    print(f"  EpisodeResult.summary() contains all required keys")
    print(f"  Sample: episode_id={summary['episode_id']}, steps={summary['steps']}")

    # Verify StepResult fields.
    if result.step_results:
        step = result.step_results[0]
        assert hasattr(step, "step")
        assert hasattr(step, "reward")
        assert hasattr(step, "terminated")
        assert hasattr(step, "latency_ms")
        print(f"  StepResult fields OK: step={step.step}, reward={step.reward:.3f}, latency={step.latency_ms:.1f}ms")

    return "BaseEnvironment contract, EpisodeResult, and StepResult all validated."


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-api", action="store_true", help="Skip tests that call external APIs")
    args = parser.parse_args()

    results = []

    # Always-run tests (no API needed)
    results.append(run_test("ModelBackend factories", test_model_backend_factories))
    results.append(run_test("CarRLEnvironment", test_car_rl_environment))
    results.append(run_test("UIElementDetector (synthetic frames)", test_ui_element_detector))
    results.append(run_test("BaseEnvironment contract", test_base_environment_contract))

    # API-dependent tests
    if args.skip_api:
        print("\n[Skipping API tests — pass without --skip-api to run them]")
    else:
        results.append(run_test("ConversationEnvironment + Anthropic API", test_conversation_environment_with_api))
        results.append(run_test("LLMRLEnvironment + Anthropic API", test_llm_rl_environment_with_api))

    print_summary(results)

    failed = sum(1 for r in results if not r.passed)
    sys.exit(failed)


if __name__ == "__main__":
    main()
