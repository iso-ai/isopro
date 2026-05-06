"""Generate analysis tables and charts for the ISOPro vs GRPO-LoRA comparison.

Reads every ``data/{isopro,grpo_lora}_*/experiment_results.json`` produced by
the experiments/ Modal runners + local M1 ISOPro runs, plus the paper's
published Qwen scheduling baseline, and emits:

    docs/analysis/results_table.md          full 12-cell results + per-tier
    docs/analysis/head_to_head.md           5 or 6 head-to-head margins
    docs/analysis/buffer_balance.md         entropy vs ISOPro-margin scatter
    docs/analysis/comparison_bars.png       grouped bar chart (mean acc)
    docs/analysis/per_tier_heatmap.png      tier × cell heatmap
    docs/analysis/trajectory_lines.png      per-iteration mean acc curves
    docs/analysis/buffer_composition.png    final-iter buffer bars
    docs/analysis/buffer_balance_scatter.png  hypothesis test plot

Designed to be re-runnable: drop in a new experiment_results.json, re-run.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT = REPO / "docs" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Cell registry — every (method, model, domain) → JSON path or None
# ---------------------------------------------------------------------------


# Paper's published Qwen scheduling ISOPro result (no JSON on disk; hardcoded).
PAPER_QWEN_SCHEDULING = {
    "method":            "isopro",
    "model":             "qwen-2.5-3b",
    "domain":            "scheduling",
    "source":            "paper",
    "baseline":          {"tier0_warmup": 0.80, "tier1_sequencing": 0.0,
                          "tier2_resource_alloc": 0.0,
                          "tier3_deadline_pressure": 0.0,
                          "tier4_pairwise": 0.0,
                          "tier5_full_composition": 0.0},
    "final":             {"tier0_warmup": 1.00, "tier1_sequencing": 0.667,
                          "tier2_resource_alloc": 0.0,
                          "tier3_deadline_pressure": 0.667,
                          "tier4_pairwise": 0.0,
                          "tier5_full_composition": 0.0},
    "history":           None,                                # not available
}


CELLS: list[dict[str, Any]] = [
    {"method": "isopro",    "model": "qwen-2.5-3b",  "domain": "scheduling",
     "source": "paper",     "data": PAPER_QWEN_SCHEDULING},
    {"method": "isopro",    "model": "llama-3.2-3b", "domain": "scheduling",
     "source": "m1",        "path": DATA / "scheduling_llama32_3b" / "experiment_results.json"},
    {"method": "isopro",    "model": "gemma-2-2b",   "domain": "scheduling",
     "source": "m1",        "path": DATA / "scheduling_gemma2_2b" / "isopro_results.json"},
    {"method": "isopro",    "model": "qwen-2.5-3b",  "domain": "mbpp",
     "source": "modal",     "path": DATA / "isopro_Qwen2.5-3B-Instruct_mbpp" / "experiment_results.json"},
    {"method": "isopro",    "model": "llama-3.2-3b", "domain": "mbpp",
     "source": "modal",     "path": DATA / "isopro_Llama-3.2-3B-Instruct_mbpp" / "experiment_results.json"},
    {"method": "isopro",    "model": "gemma-2-2b",   "domain": "mbpp",
     "source": "modal",     "path": DATA / "isopro_gemma-2-2b-it_mbpp" / "experiment_results.json"},
    {"method": "grpo_lora", "model": "qwen-2.5-3b",  "domain": "scheduling",
     "source": "modal",     "path": DATA / "grpo_lora_Qwen2.5-3B-Instruct_scheduling" / "experiment_results.json"},
    {"method": "grpo_lora", "model": "llama-3.2-3b", "domain": "scheduling",
     "source": "modal",     "path": DATA / "grpo_lora_Llama-3.2-3B-Instruct_scheduling" / "experiment_results.json"},
    {"method": "grpo_lora", "model": "gemma-2-2b",   "domain": "scheduling",
     "source": "modal",     "path": DATA / "grpo_lora_gemma-2-2b-it_scheduling" / "experiment_results.json"},
    {"method": "grpo_lora", "model": "qwen-2.5-3b",  "domain": "mbpp",
     "source": "modal",     "path": DATA / "grpo_lora_Qwen2.5-3B-Instruct_mbpp" / "experiment_results.json"},
    {"method": "grpo_lora", "model": "llama-3.2-3b", "domain": "mbpp",
     "source": "modal",     "path": DATA / "grpo_lora_Llama-3.2-3B-Instruct_mbpp" / "experiment_results.json"},
    {"method": "grpo_lora", "model": "gemma-2-2b",   "domain": "mbpp",
     "source": "modal",     "path": DATA / "grpo_lora_gemma-2-2b-it_mbpp" / "experiment_results.json"},
]


def _load_cell(cell: dict[str, Any]) -> dict[str, Any] | None:
    """Load a cell's results JSON or return its inline data.

    Args:
        cell: A CELLS registry entry.

    Returns:
        Dict with keys ``baseline``, ``final``, ``history``, ``mean``, or
        None if the file isn't there yet.
    """
    if cell.get("data"):
        d = cell["data"]
        return {
            "baseline":         d.get("baseline"),
            "final":            d["final"],
            "history":          d.get("history"),
            "mean":             sum(d["final"].values()) / max(len(d["final"]), 1),
            "tier_count":       len(d["final"]),
        }
    p: Path = cell["path"]
    if not p.exists():
        return None
    payload = json.loads(p.read_text())
    history = payload.get("isopro_loop", {}).get("history") or payload.get("history") or []
    final = payload.get("final") or (history[-1]["tier_accuracies"] if history else {})
    baseline = payload.get("baseline")
    return {
        "baseline":   baseline,
        "final":      final,
        "history":    history,
        "mean":       sum(final.values()) / max(len(final), 1),
        "tier_count": len(final),
        "wall_clock": payload.get("total_wall_clock_s"),
    }


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def emit_main_table(rows: list[dict]) -> None:
    """Write the full 12-cell results table to docs/analysis/results_table.md."""
    out = ["# Full Results: ISOPro vs GRPO-LoRA across 3 models × 2 domains\n"]
    out.append("Cells with — are not yet available.\n\n")
    out.append("## Mean accuracy\n\n")
    out.append("| Model | Scheduling — ISOPro | Scheduling — GRPO | MBPP — ISOPro | MBPP — GRPO |\n")
    out.append("|---|---:|---:|---:|---:|\n")
    by_key = {(r["method"], r["model"], r["domain"]): r for r in rows}
    for model in ("qwen-2.5-3b", "llama-3.2-3b", "gemma-2-2b"):
        cells = []
        for method, domain in [
            ("isopro", "scheduling"), ("grpo_lora", "scheduling"),
            ("isopro", "mbpp"),       ("grpo_lora", "mbpp"),
        ]:
            r = by_key.get((method, model, domain))
            if r and r.get("mean") is not None:
                cells.append(f"{r['mean']*100:.1f}%")
            else:
                cells.append("—")
        out.append(f"| {model} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |\n")

    out.append("\n## Per-tier breakdown\n")
    for r in rows:
        if not r.get("final"):
            continue
        out.append(f"\n### {r['method']} / {r['model']} / {r['domain']} "
                   f"(source: {r['source']}; mean: {r['mean']*100:.1f}%)\n\n")
        out.append("| Tier | Accuracy |\n|---|---:|\n")
        for tier, acc in r["final"].items():
            out.append(f"| {tier} | {acc*100:.1f}% |\n")

    (OUT / "results_table.md").write_text("".join(out))
    print(f"wrote {OUT / 'results_table.md'}")


def emit_head_to_head(rows: list[dict]) -> None:
    """Write the head-to-head margin table to docs/analysis/head_to_head.md."""
    out = ["# ISOPro vs GRPO-LoRA — Head-to-head margins\n\n"]
    out.append("| Model | Domain | ISOPro | GRPO | Winner | Margin |\n")
    out.append("|---|---|---:|---:|---|---:|\n")
    by_key = {(r["method"], r["model"], r["domain"]): r for r in rows}
    iso_wins = grp_wins = 0
    iso_margin = grp_margin = 0.0
    for model in ("qwen-2.5-3b", "llama-3.2-3b", "gemma-2-2b"):
        for domain in ("scheduling", "mbpp"):
            iso = by_key.get(("isopro", model, domain))
            grp = by_key.get(("grpo_lora", model, domain))
            if not (iso and grp and iso.get("mean") is not None and grp.get("mean") is not None):
                iso_str = f"{iso['mean']*100:.1f}%" if (iso and iso.get("mean") is not None) else "—"
                grp_str = f"{grp['mean']*100:.1f}%" if (grp and grp.get("mean") is not None) else "—"
                out.append(f"| {model} | {domain} | {iso_str} | {grp_str} | pending | — |\n")
                continue
            margin = (iso["mean"] - grp["mean"]) * 100
            if margin > 0:
                winner, abs_m = "ISOPro", margin
                iso_wins += 1
                iso_margin += margin
            elif margin < 0:
                winner, abs_m = "GRPO", -margin
                grp_wins += 1
                grp_margin += -margin
            else:
                winner, abs_m = "tie", 0.0
            out.append(f"| {model} | {domain} | {iso['mean']*100:.1f}% | "
                       f"{grp['mean']*100:.1f}% | **{winner}** | "
                       f"{abs_m:+.1f}pp |\n")

    out.append(f"\n**Score:** ISOPro {iso_wins} ({iso_margin:.1f}pp avg margin), "
               f"GRPO {grp_wins} ({grp_margin:.1f}pp avg margin).\n")
    (OUT / "head_to_head.md").write_text("".join(out))
    print(f"wrote {OUT / 'head_to_head.md'}")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


_MODEL_LABELS = {
    "qwen-2.5-3b":  "Qwen 2.5 3B",
    "llama-3.2-3b": "Llama 3.2 3B",
    "gemma-2-2b":   "Gemma 2 2B",
}


def chart_comparison_bars(rows: list[dict]) -> None:
    """Grouped bar chart: mean accuracy by (model, domain), method side-by-side."""
    by_key = {(r["method"], r["model"], r["domain"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    width = 0.35
    for ax, domain in zip(axes, ("scheduling", "mbpp")):
        models = list(_MODEL_LABELS.keys())
        x = np.arange(len(models))
        iso = [by_key.get(("isopro", m, domain), {}).get("mean") for m in models]
        grp = [by_key.get(("grpo_lora", m, domain), {}).get("mean") for m in models]
        iso_ = [v * 100 if v is not None else 0 for v in iso]
        grp_ = [v * 100 if v is not None else 0 for v in grp]
        ax.bar(x - width / 2, iso_, width, label="ISOPro",     color="#3a86ff")
        ax.bar(x + width / 2, grp_, width, label="GRPO-LoRA",  color="#ff006e")
        ax.set_xticks(x)
        ax.set_xticklabels([_MODEL_LABELS[m] for m in models], rotation=15)
        ax.set_title(domain.capitalize())
        ax.set_ylabel("Mean accuracy (%)")
        ax.set_ylim(0, 60)
        ax.grid(axis="y", alpha=0.3)
        # Annotate bars with values
        for xi, (vi, vg) in enumerate(zip(iso_, grp_)):
            if vi:  ax.text(xi - width / 2, vi + 1, f"{vi:.1f}", ha="center", fontsize=8)
            if vg:  ax.text(xi + width / 2, vg + 1, f"{vg:.1f}", ha="center", fontsize=8)
    axes[0].legend(loc="upper left")
    fig.suptitle("ISOPro vs GRPO-LoRA — mean accuracy across 3 models × 2 domains", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "comparison_bars.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT / 'comparison_bars.png'}")


def chart_per_tier_heatmap(rows: list[dict]) -> None:
    """Heatmap: per-tier final accuracy across all 12 cells."""
    sched_tiers = [
        "tier0_warmup", "tier1_sequencing", "tier2_resource_alloc",
        "tier3_deadline_pressure", "tier4_pairwise", "tier5_full_composition",
    ]
    mbpp_tiers = [
        "tier0_warmup", "tier1_short", "tier2_medium",
        "tier3_long", "tier4_held_out",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, domain, tiers in [
        (axes[0], "scheduling", sched_tiers),
        (axes[1], "mbpp",       mbpp_tiers),
    ]:
        labels, mat = [], []
        for method in ("isopro", "grpo_lora"):
            for model in _MODEL_LABELS:
                r = next((r for r in rows
                          if r["method"] == method and r["model"] == model
                          and r["domain"] == domain), None)
                if not (r and r.get("final")):
                    continue
                row = [r["final"].get(t, np.nan) for t in tiers]
                mat.append(row)
                m_short = "ISO" if method == "isopro" else "GRPO"
                labels.append(f"{m_short} {_MODEL_LABELS[model]}")
        if not mat:
            continue
        arr = np.array(mat)
        im = ax.imshow(arr, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels([t.replace("tier", "T").replace("_", " ") for t in tiers],
                           rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(domain.capitalize())
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if not np.isnan(arr[i, j]):
                    ax.text(j, i, f"{arr[i, j]*100:.0f}",
                            ha="center", va="center",
                            color="white" if arr[i, j] < 0.5 else "black",
                            fontsize=7)
        plt.colorbar(im, ax=ax, label="accuracy", shrink=0.85)

    fig.suptitle("Per-tier accuracy across all (method, model, domain) cells", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "per_tier_heatmap.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT / 'per_tier_heatmap.png'}")


def chart_trajectories(rows: list[dict]) -> None:
    """Per-iteration mean accuracy lines, one panel per (model, domain)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True)
    domains = ("scheduling", "mbpp")
    for di, domain in enumerate(domains):
        for mi, model in enumerate(_MODEL_LABELS):
            ax = axes[di, mi]
            ax.set_title(f"{_MODEL_LABELS[model]} — {domain}", fontsize=10)
            ax.set_xlabel("iteration")
            if mi == 0:
                ax.set_ylabel("mean accuracy (%)")
            ax.set_ylim(0, 60)
            ax.grid(alpha=0.3)
            for method, color in [("isopro", "#3a86ff"), ("grpo_lora", "#ff006e")]:
                r = next((r for r in rows
                          if r["method"] == method and r["model"] == model
                          and r["domain"] == domain), None)
                if not r or not r.get("history"):
                    continue
                xs = [h["iteration"] for h in r["history"]]
                ys = []
                for h in r["history"]:
                    accs = h.get("tier_accuracies") or {}
                    ys.append(sum(accs.values()) / len(accs) * 100 if accs else 0)
                m_short = "ISOPro" if method == "isopro" else "GRPO"
                ax.plot(xs, ys, marker="o", label=m_short, color=color)
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Mean accuracy trajectory per iteration", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "trajectory_lines.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT / 'trajectory_lines.png'}")


def chart_buffer_balance_scatter(rows: list[dict]) -> None:
    """Scatter: buffer entropy at final iter vs ISOPro - GRPO margin (pp).

    Tests the hypothesis: when ISOPro's replay buffer is *more balanced*
    across tiers (higher entropy), ISOPro tends to win by a larger margin.
    """
    pts: list[tuple[float, float, str, str]] = []
    by_key = {(r["method"], r["model"], r["domain"]): r for r in rows}

    for model in _MODEL_LABELS:
        for domain in ("scheduling", "mbpp"):
            iso = by_key.get(("isopro", model, domain))
            grp = by_key.get(("grpo_lora", model, domain))
            if not (iso and grp and iso.get("mean") is not None and grp.get("mean") is not None):
                continue
            history = iso.get("history") or []
            if not history:
                continue
            comp = history[-1].get("buffer_composition") or {}
            total = sum(comp.values())
            if total == 0:
                continue
            ps = [v / total for v in comp.values() if v > 0]
            entropy = -sum(p * math.log2(p) for p in ps) if ps else 0.0
            margin = (iso["mean"] - grp["mean"]) * 100
            pts.append((entropy, margin, model, domain))

    if not pts:
        print("(skipping buffer-balance scatter; not enough cells with history)")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))
    color_by_domain = {"scheduling": "#fb8500", "mbpp": "#219ebc"}
    for ent, mar, model, dom in pts:
        ax.scatter(ent, mar, s=80, color=color_by_domain[dom], edgecolor="black",
                   label=dom if dom not in ax.get_legend_handles_labels()[1] else "")
        ax.annotate(f"{_MODEL_LABELS[model].split()[0]} ({dom[0]})",
                    (ent, mar), textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Replay-buffer entropy (bits) — higher = more tier-balanced")
    ax.set_ylabel("ISOPro − GRPO margin (percentage points)")
    ax.set_title("Buffer-balance hypothesis: balanced buffers favor ISOPro")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "buffer_balance_scatter.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT / 'buffer_balance_scatter.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    rows: list[dict] = []
    missing: list[str] = []
    for cell in CELLS:
        loaded = _load_cell(cell)
        if loaded is None:
            missing.append(f"{cell['method']}/{cell['model']}/{cell['domain']}")
            continue
        rows.append({**cell, **loaded})

    if missing:
        print("Missing cells:")
        for m in missing:
            print(f"  - {m}")
        print()

    print(f"Loaded {len(rows)}/{len(CELLS)} cells.")
    emit_main_table(rows)
    emit_head_to_head(rows)
    chart_comparison_bars(rows)
    chart_per_tier_heatmap(rows)
    chart_trajectories(rows)
    chart_buffer_balance_scatter(rows)


if __name__ == "__main__":
    main()
