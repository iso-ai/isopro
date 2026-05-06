"""Deterministic verifier for MBPP code-creation responses.

Two-stage verification, mirroring the scheduling verifier:

  Stage 1 — Extraction: pull a Python code block out of the model's free-text
            response. Tries fenced ```python``` first, then bare ``` blocks,
            then a defensive whole-response fallback for models that emit
            raw code with no fence.

  Stage 2 — Execution: write the code + the MBPP unit tests to a temp file
            and run it in a fresh ``python3`` subprocess with a wall-clock
            timeout, no inherited environment, and stderr/stdout captured.
            Pass iff the subprocess exits with returncode 0.

No partial credit. Any uncaught exception or any failed assertion rejects the
entire response. This is the architectural promise of GCE in the code domain:
the verifier is the reward signal.

Security note:
    The subprocess is *not* a hardened sandbox — it cannot stop adversarial
    code from making network calls, reading the filesystem, etc. It is
    suitable for verifying outputs from a cooperating LLM in a research
    setting. For untrusted code review, use a real sandbox (firejail,
    containers, or the BigCode sandbox).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------------------


@dataclass
class CodeVerificationResult:
    """Detailed outcome of running a generated solution against MBPP tests.

    Attributes:
        passed: True iff every assertion passed.
        extracted_code: The code block lifted from the model's response,
            or None if extraction failed.
        extraction_strategy: Which extraction strategy succeeded.
        failures: Human-readable failure reasons (parse error, syntax
            error, assertion failure, timeout, etc.).
        stdout: Captured stdout of the test run (truncated).
        stderr: Captured stderr of the test run (truncated).
        timed_out: True if the subprocess hit the wall-clock limit.
    """

    passed: bool
    extracted_code: str | None = None
    extraction_strategy: str | None = None
    failures: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


# ---------------------------------------------------------------------------
# Stage 1 — Extraction
# ---------------------------------------------------------------------------


_FENCED_PYTHON_RE = re.compile(
    r"```(?:python|py|python3)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL
)
_FENCED_BARE_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
_DEF_RE = re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE)


def _extract_fenced_python(text: str) -> str | None:
    """Pull the contents of the first ```python ... ``` block.

    Args:
        text: The model's raw response.

    Returns:
        Inner code if found, else None.
    """
    match = _FENCED_PYTHON_RE.search(text)
    return match.group(1) if match else None


def _extract_fenced_bare(text: str) -> str | None:
    """Pull the contents of the first untyped ``` ... ``` block.

    Args:
        text: The model's raw response.

    Returns:
        Inner code if found, else None.
    """
    match = _FENCED_BARE_RE.search(text)
    return match.group(1) if match else None


def _extract_bare_def(text: str) -> str | None:
    """Fallback: assume the response is raw code if it contains a ``def``.

    Strips a trailing ``Note:``-style commentary if a model rambles after
    the function definition.

    Args:
        text: The model's raw response.

    Returns:
        Stripped raw text if it contains a top-level ``def``, else None.
    """
    if not _DEF_RE.search(text):
        return None
    return text.strip()


_EXTRACT_STRATEGIES: list[tuple[str, Any]] = [
    ("fenced_python", _extract_fenced_python),
    ("fenced_bare", _extract_fenced_bare),
    ("bare_def", _extract_bare_def),
]


def extract_code(text: str) -> tuple[str | None, str | None]:
    """Lift a Python code block out of a free-text response.

    Tries each extraction strategy in order; returns the first that hits.

    Args:
        text: The model's raw response.

    Returns:
        (code, strategy_name). Both None if no strategy succeeded.
    """
    for name, strategy in _EXTRACT_STRATEGIES:
        code = strategy(text)
        if code and code.strip():
            return code, name
    return None, None


# ---------------------------------------------------------------------------
# Stage 2 — Execution
# ---------------------------------------------------------------------------


def _build_test_script(
    code: str,
    test_imports: list[str],
    tests: list[str],
) -> str:
    """Assemble the script that will be executed in the sandbox.

    Args:
        code: Extracted candidate solution.
        test_imports: Imports the assertions need.
        tests: List of assertion strings.

    Returns:
        Complete Python source as a single string.
    """
    blocks = []
    for imp in test_imports:
        blocks.append(imp)
    blocks.append(code)
    blocks.append("# --- MBPP assertions ---")
    blocks.extend(tests)
    return "\n".join(blocks)


def _run_in_sandbox(
    script: str,
    timeout_s: float,
) -> tuple[bool, bool, str, str]:
    """Execute the test script in a fresh subprocess.

    Args:
        script: Full source to run.
        timeout_s: Wall-clock seconds before SIGKILL.

    Returns:
        (success, timed_out, stdout, stderr).
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"},
            check=False,
        )
        return proc.returncode == 0, False, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        return (
            False,
            True,
            exc.stdout.decode("utf-8", errors="replace") if exc.stdout else "",
            exc.stderr.decode("utf-8", errors="replace") if exc.stderr else "",
        )
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def verify_code(
    response: str,
    test_imports: list[str],
    tests: list[str],
    timeout_s: float = 5.0,
    max_log_chars: int = 800,
) -> CodeVerificationResult:
    """End-to-end verification: extract, execute, classify.

    Args:
        response: Raw model response text.
        test_imports: Imports the MBPP assertions need.
        tests: List of assertion strings from MBPP's ``test_list``.
        timeout_s: Subprocess wall-clock limit. MBPP problems should run in
            milliseconds; the default 5s catches accidental infinite loops.
        max_log_chars: Truncate captured stdout/stderr at this length so a
            misbehaving response can't blow up logs.

    Returns:
        A CodeVerificationResult with full diagnostics.
    """
    code, strategy = extract_code(response)
    if code is None:
        return CodeVerificationResult(
            passed=False,
            extraction_strategy=None,
            failures=["No Python code block extracted from response."],
        )

    script = _build_test_script(code, test_imports, tests)
    success, timed_out, stdout, stderr = _run_in_sandbox(script, timeout_s)

    failures: list[str] = []
    if timed_out:
        failures.append(f"Execution timed out after {timeout_s:.1f}s.")
    elif not success:
        # Try to surface the most informative line from stderr.
        last_lines = [l for l in stderr.strip().split("\n") if l][-3:]
        failures.append("Test failed: " + " | ".join(last_lines) if last_lines else "Test failed (no stderr).")

    return CodeVerificationResult(
        passed=success,
        extracted_code=code,
        extraction_strategy=strategy,
        failures=failures,
        stdout=stdout[:max_log_chars],
        stderr=stderr[:max_log_chars],
        timed_out=timed_out,
    )


# ---------------------------------------------------------------------------
# ISOPro environment-compatible scoring entrypoint
# ---------------------------------------------------------------------------


def score_mbpp_task(task: Any, response: str) -> tuple[float, dict]:
    """Score one MBPP response. Binary: 1.0 or 0.0.

    Compatible with the ISOPro env.score() interface. Reads the test list
    and required imports from ``task.metadata``.

    Args:
        task: A Task whose metadata contains 'test_list' and 'test_imports'.
        response: The model's raw text response.

    Returns:
        (reward, detail). reward is 1.0 iff every assertion passes.
    """
    meta = task.metadata
    result = verify_code(
        response=response,
        test_imports=list(meta.get("test_imports", [])),
        tests=list(meta["test_list"]),
        timeout_s=float(meta.get("timeout_s", 5.0)),
    )

    detail = {
        "passed": result.passed,
        "extraction_strategy": result.extraction_strategy,
        "failures": result.failures,
        "timed_out": result.timed_out,
        "stderr_tail": result.stderr[-300:],
    }
    return (1.0 if result.passed else 0.0), detail
