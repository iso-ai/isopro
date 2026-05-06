"""Generate analysis tables and charts for the ISOPro vs GRPO-LoRA comparison.

Reads every ``data/{isopro,grpo_lora}_*/experiment_results.json`` produced by
the experiments/ Modal runners + local M1 ISOPro runs, plus the paper's
published Qwen scheduling baseline, and emits:

    docs/analysis/results_table.md          full 12-cell results + per-tier
    docs/analysis/head_to_head.md           winner margins + net learning Δ
    docs/analysis/comparison_bars.png       grouped bar chart (mean acc)
    docs/analysis/per_tier_heatmap.png      tier × cell heatmap
    docs/analysis/trajectory_lines.png      per-iteration mean acc curves
    docs/analysis/delta_from_baseline.png   signed Δ-from-baseline bars

Designed to be re-runnable: drop in a new experiment_results.json, re-run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT = REPO / "docs" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Cell registry
# ---------------------------------------------------------------------------


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
    "history":           None,
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


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _mean(d: dict | None) -> float | None:
    """Mean of dict values, or None if dict is empty/None."""
    if not d:
        return None
    return sum(d.values()) / len(d)


def _load_cell(cell: dict[str, Any]) -> dict[str, Any] | None:
    """Load a cell's results JSON or return its inline data.

    Returns dict with: baseline, final, history, mean, baseline_mean,
    delta_pp (final_mean − baseline_mean, expressed in percentage points).

    For GRPO cells (which don't log a true zero-shot baseline), iter-1 eval
    is used as a proxy baseline and tagged so callers can disclose it.
    """
    if cell.get("data"):
        d = cell["data"]
        baseline = d.get("baseline")
        final = d["final"]
        return {
            "baseline":       baseline,
            "baseline_mean":  _mean(baseline),
            "baseline_proxy": False,
            "final":          final,
            "history":        d.get("history"),
            "mean":           _mean(final),
            "delta_pp":       ((_mean(final) - _mean(baseline)) * 100
                               if baseline and final else None),
            "tier_count":     len(final),
        }
    p: Path = cell["path"]
    if not p.exists():
        return None
    payload = json.loads(p.read_text())
    history = payload.get("isopro_loop", {}).get("history") or payload.get("history") or []
    final = payload.get("final") or (history[-1]["tier_accuracies"] if history else {})
    baseline = payload.get("baseline")
    baseline_proxy = False
    if not baseline and history:
        baseline = history[0].get("tier_accuracies")
        baseline_proxy = True
    return {
        "baseline":       baseline,
        "baseline_mean":  _mean(baseline),
        "baseline_proxy": baseline_proxy,
        "final":          final,
        "history":        history,
        "mean":           _mean(final),
        "delta_pp":       ((_mean(final) - _mean(baseline)) * 100
                           if baseline and final else None),
        "tier_count":     len(final),
        "wall_clock":     payload.get("total_wall_clock_s"),
    }


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def emit_main_table(rows: list[dict]) -> None:
    """Write the full 12-cell results table to docs/analysis/results_table.md."""
    out = ["# Full Results: ISOPro vs GRPO-LoRA across 3 models × 2 domains\n\n"]
    out.append("Eval set: 5 problems per tier. Single seed (42).\n\n")

    out.append("## Final mean accuracy\n\n")
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
            cells.append(f"{r['mean']*100:.1f}%" if r and r.get("mean") is not None else "—")
        out.append(f"| {model} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |\n")

    out.append("\n## Net learning Δ (final − baseline)\n\n")
    out.append("Positive = the method moved the model up from its zero-shot baseline. "
               "Negative = regression. GRPO baselines marked † use the iter-1 eval "
               "(after 14 training steps) as a proxy because the GRPO runner does "
               "not probe the model pre-step-0.\n\n")
    out.append("| Model | Domain | Method | Baseline | Final | Δ |\n")
    out.append("|---|---|---|---:|---:|---:|\n")
    for model in ("qwen-2.5-3b", "llama-3.2-3b", "gemma-2-2b"):
        for domain in ("scheduling", "mbpp"):
            for method in ("isopro", "grpo_lora"):
                r = by_key.get((method, model, domain))
                if not r:
                    continue
                b = r["baseline_mean"]
                f = r["mean"]
                d = r["delta_pp"]
                proxy = "†" if r["baseline_proxy"] else ""
                b_str = f"{b*100:.1f}%{proxy}" if b is not None else "—"
                f_str = f"{f*100:.1f}%" if f is not None else "—"
                d_str = f"{d:+.1f}pp" if d is not None else "—"
                m_short = "ISOPro" if method == "isopro" else "GRPO"
                out.append(f"| {model} | {domain} | {m_short} | {b_str} | {f_str} | **{d_str}** |\n")

    out.append("\n## Per-tier breakdown\n")
    for r in rows:
        if not r.get("final"):
            continue
        out.append(f"\n### {r['method']} / {r['model']} / {r['domain']} "
                   f"(source: {r['source']}; mean: {r['mean']*100:.1f}%, "
                   f"Δ: {r['delta_pp']:+.1f}pp)\n\n")
        out.append("| Tier | Baseline | Final | Δ |\n|---|---:|---:|---:|\n")
        baseline = r.get("baseline") or {}
        for tier, acc in r["final"].items():
            b = baseline.get(tier)
            b_str = f"{b*100:.1f}%" if b is not None else "—"
            d_str = f"{(acc - b) * 100:+.1f}pp" if b is not None else "—"
            out.append(f"| {tier} | {b_str} | {acc*100:.1f}% | {d_str} |\n")

    (OUT / "results_table.md").write_text("".join(out))
    print(f"wrote {OUT / 'results_table.md'}")


def emit_head_to_head(rows: list[dict]) -> None:
    """Write the head-to-head margin table to docs/analysis/head_to_head.md."""
    out = ["# ISOPro vs GRPO-LoRA — Head-to-head margins\n\n"]
    out.append("Eval set: 5 problems per tier. Single seed (42). "
               "GRPO baseline marked † uses iter-1 eval as proxy.\n\n")
    out.append("## Final accuracy comparison\n\n")
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

    out.append(f"\n**Score:** ISOPro {iso_wins} ({iso_margin/max(iso_wins,1):.1f}pp avg "
               f"winning margin), GRPO {grp_wins} "
               f"({grp_margin/max(grp_wins,1):.1f}pp avg winning margin).\n\n")

    out.append("## Net learning Δ (final − baseline)\n\n")
    out.append("This separates 'method moved the model' from 'method ended at a "
               "high number because the baseline was high.' Both methods produce "
               "mixed Δ at consumer-budget hyperparameters.\n\n")
    out.append("| Model | Domain | ISOPro Δ | GRPO Δ |\n|---|---|---:|---:|\n")
    iso_delta_total = grp_delta_total = 0.0
    iso_n = grp_n = 0
    for model in ("qwen-2.5-3b", "llama-3.2-3b", "gemma-2-2b"):
        for domain in ("scheduling", "mbpp"):
            iso = by_key.get(("isopro", model, domain))
            grp = by_key.get(("grpo_lora", model, domain))
            iso_d = (f"{iso['delta_pp']:+.1f}pp"
                     if iso and iso.get("delta_pp") is not None else "—")
            grp_d = (f"{grp['delta_pp']:+.1f}pp" + ("†" if grp and grp["baseline_proxy"] else "")
                     if grp and grp.get("delta_pp") is not None else "—")
            out.append(f"| {model} | {domain} | {iso_d} | {grp_d} |\n")
            if iso and iso.get("delta_pp") is not None:
                iso_delta_total += iso["delta_pp"]; iso_n += 1
            if grp and grp.get("delta_pp") is not None:
                grp_delta_total += grp["delta_pp"]; grp_n += 1

    out.append(f"\n**Mean Δ across cells:** ISOPro {iso_delta_total/max(iso_n,1):+.1f}pp, "
               f"GRPO {grp_delta_total/max(grp_n,1):+.1f}pp.\n")

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
        for xi, (vi, vg) in enumerate(zip(iso_, grp_)):
            if vi:  ax.text(xi - width / 2, vi + 1, f"{vi:.1f}", ha="center", fontsize=8)
            if vg:  ax.text(xi + width / 2, vg + 1, f"{vg:.1f}", ha="center", fontsize=8)
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "ISOPro vs GRPO-LoRA — final mean accuracy  "
        "(eval N=5 per tier; single seed)",
        fontsize=11,
    )
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

    fig.suptitle(
        "Per-tier accuracy across all (method, model, domain) cells  "
        "(eval N=5 per tier; single seed)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "per_tier_heatmap.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT / 'per_tier_heatmap.png'}")


def chart_trajectories(rows: list[dict]) -> None:
    """Per-iteration mean accuracy lines, one panel per (model, domain).

    Paper-published cells (Qwen scheduling ISOPro) lack per-iteration history;
    we render their baseline and final as endpoints instead of dropping them.
    """
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
                if not r:
                    continue
                m_short = "ISOPro" if method == "isopro" else "GRPO"
                if r.get("history"):
                    xs = [h["iteration"] for h in r["history"]]
                    ys = [(_mean(h.get("tier_accuracies") or {}) or 0) * 100
                          for h in r["history"]]
                    ax.plot(xs, ys, marker="o", label=m_short, color=color)
                elif r.get("baseline_mean") is not None and r.get("mean") is not None:
                    ax.plot([0, 6],
                            [r["baseline_mean"] * 100, r["mean"] * 100],
                            marker="s", markersize=10, linestyle=":",
                            label=f"{m_short} (endpoints only)", color=color)
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Mean accuracy trajectory per iteration  "
        "(eval N=5 per tier; single seed; dotted = endpoints only)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "trajectory_lines.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT / 'trajectory_lines.png'}")


def chart_delta_from_baseline(rows: list[dict]) -> None:
    """Signed Δ-from-baseline bars: 6 cells × 2 methods.

    Replaces the buffer-balance scatter (n=5 was too few for any single-axis
    hypothesis claim). The story this chart tells: both methods produce mixed
    Δ at consumer-budget hyperparameters; ISOPro has the largest learning
    gains; GRPO has the largest regressions; some cells regress under both.
    """
    by_key = {(r["method"], r["model"], r["domain"]): r for r in rows}
    cells: list[tuple[str, str]] = []
    for model in _MODEL_LABELS:
        for domain in ("scheduling", "mbpp"):
            cells.append((model, domain))

    iso_d, grp_d, labels = [], [], []
    for model, domain in cells:
        iso = by_key.get(("isopro", model, domain))
        grp = by_key.get(("grpo_lora", model, domain))
        labels.append(f"{_MODEL_LABELS[model].split()[0]}\n{domain}")
        iso_d.append(iso["delta_pp"] if iso and iso.get("delta_pp") is not None else 0.0)
        grp_d.append(grp["delta_pp"] if grp and grp.get("delta_pp") is not None else 0.0)

    x = np.arange(len(cells))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    iso_bars = ax.bar(x - width / 2, iso_d, width, color="#3a86ff", label="ISOPro")
    grp_bars = ax.bar(x + width / 2, grp_d, width, color="#ff006e", label="GRPO-LoRA")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Δ from baseline (percentage points)")
    ax.set_title(
        "Net learning gain: final − baseline accuracy across 6 cells\n"
        "(ISOPro = blue, GRPO-LoRA = red; eval N=5 per tier; single seed; "
        "GRPO baseline is iter-1 eval as proxy)"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")

    for bars in (iso_bars, grp_bars):
        for b in bars:
            v = b.get_height()
            if v == 0:
                continue
            ax.text(b.get_x() + b.get_width() / 2, v + (1 if v >= 0 else -3),
                    f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "delta_from_baseline.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT / 'delta_from_baseline.png'}")


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
    chart_delta_from_baseline(rows)

    # Drop the previous buffer-balance scatter — n=5 was too few for any
    # single-axis hypothesis claim, and the data we have shows two distinct
    # failure modes (Llama scheduling buffer-skew + Gemma MBPP small-model
    # drift) that one axis can't disambiguate.
    stale = OUT / "buffer_balance_scatter.png"
    if stale.exists():
        stale.unlink()
        print(f"removed stale {stale}")


if __name__ == "__main__":
    main()
