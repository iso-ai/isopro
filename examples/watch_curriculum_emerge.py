"""Visualize the implicit curriculum forming in ISOPro's replay buffer.

The GCE paper's central training claim is that ISOPro's replay buffer
*becomes* a curriculum without anyone designing one. Easy wins accumulate
first. Harder problems enter the buffer only as model capability develops.
By iteration N, the model trains on everything it has solved correctly in
iterations 1..N — compounding correct signal without compounding error.

This script visualizes that emergence directly from a saved training log
(no model required). It reproduces the data that Figure 3 of the paper plots,
right in your terminal.

Run:
    python examples/watch_curriculum_emerge.py
    python examples/watch_curriculum_emerge.py --log path/to/your_run.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_DEFAULT_LOG = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "scheduling_experiment_qwen"
    / "experiment_results.json"
)


_TIER_LABELS = {
    "tier0_warmup": "T0 warmup",
    "tier1_sequencing": "T1 sequencing",
    "tier2_resource_alloc": "T2 resource",
    "tier3_deadline_pressure": "T3 deadline",
    "tier4_pairwise": "T4 pairwise",
    "tier5_full_composition": "T5 full",
}


def _bar(count: int, total: int, width: int = 30) -> str:
    """Render a horizontal bar with `count` of `total` units filled.

    Args:
        count: Filled units.
        total: Maximum value the bar can represent.
        width: Total bar width in characters.

    Returns:
        Bar string of length `width`.
    """
    if total <= 0:
        return " " * width
    fill = round(width * count / total)
    return "█" * fill + "·" * (width - fill)


def _load_history(log_path: Path) -> list[dict]:
    """Extract per-iteration history from a saved experiment_results.json.

    Args:
        log_path: Path to a JSON file produced by run_scheduling_experiment.py.

    Returns:
        Ordered list of per-iteration dicts.
    """
    data = json.loads(log_path.read_text())
    if "isopro_loop" in data and "history" in data["isopro_loop"]:
        return data["isopro_loop"]["history"]
    if "history" in data:
        return data["history"]
    raise ValueError(
        f"Could not find an iteration history in {log_path}. Expected either "
        f"'isopro_loop.history' or 'history' at the top level."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=_DEFAULT_LOG)
    args = parser.parse_args()

    if not args.log.exists():
        print(f"Log file not found: {args.log}", file=sys.stderr)
        print(
            "Run examples/run_scheduling_experiment.py first, or pass --log "
            "to point at any RSTrainingLog JSON.",
            file=sys.stderr,
        )
        sys.exit(1)

    history = _load_history(args.log)

    # ------------------------------------------------------------------
    # Print the buffer's composition each iteration
    # ------------------------------------------------------------------
    print("=" * 78)
    print("REPLAY BUFFER COMPOSITION OVER TIME — the implicit curriculum")
    print("=" * 78)

    buffer_max = max(
        (sum(it.get("buffer_composition", {}).values()) for it in history),
        default=1,
    )

    for iteration in history:
        i = iteration["iteration"]
        comp = iteration.get("buffer_composition", {})
        total = sum(comp.values())
        hits = iteration.get("n_correct", 0)
        attempts = iteration.get("n_rollouts", 0)
        loss = iteration.get("train_loss")
        loss_str = f"loss={loss:.3f}" if loss is not None else "loss=—"
        hit_rate = (hits / attempts * 100) if attempts else 0.0

        print(
            f"\n── Iteration {i}  "
            f"hit_rate={hit_rate:5.1f}%  ({hits}/{attempts})  "
            f"buffer={total}  {loss_str}"
        )
        for tier_id, label in _TIER_LABELS.items():
            n = comp.get(tier_id, 0)
            print(f"  {label:14s} {_bar(n, buffer_max)} {n}")

    # ------------------------------------------------------------------
    # Capability emergence: when did each tier first produce a correct trace?
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("CAPABILITY EMERGENCE — first iteration each tier appeared in buffer")
    print("=" * 78)
    first_seen: dict[str, int | None] = {tier: None for tier in _TIER_LABELS}
    for iteration in history:
        for tier in _TIER_LABELS:
            if first_seen[tier] is None and iteration.get(
                "buffer_composition", {}
            ).get(tier, 0) > 0:
                first_seen[tier] = iteration["iteration"]

    for tier, label in _TIER_LABELS.items():
        when = first_seen[tier]
        marker = f"iteration {when}" if when else "never reached"
        print(f"  {label:14s}  →  {marker}")

    # ------------------------------------------------------------------
    # Per-tier eval accuracy at the final iteration
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("FINAL PER-TIER ACCURACY")
    print("=" * 78)
    final = history[-1].get("tier_accuracies", {})
    for tier, label in _TIER_LABELS.items():
        acc = final.get(tier, 0.0)
        bar = _bar(int(acc * 30), 30, width=30)
        print(f"  {label:14s} {bar} {acc*100:5.1f}%")

    print(
        """
The two architectural takeaways:

  1. The buffer's composition was never designed — it is a direct readout of
     what the model could solve at each iteration. T0 warmup dominates early.
     Harder tiers enter only after capability emerges. This is the implicit
     curriculum.

  2. Tiers that *never* appear in the buffer are honest capability boundaries.
     The framework reports them rather than papering over them with synthetic
     examples or oracle distillation.
"""
    )


if __name__ == "__main__":
    main()
