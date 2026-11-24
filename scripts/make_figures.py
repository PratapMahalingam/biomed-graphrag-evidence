"""Figures from results/<run>/metrics.json.

python scripts/make_figures.py --results results/scifact --out figures/scifact
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8, "figure.dpi": 150})


def _label(name: str) -> str:
    return {"hybrid": "Hybrid", "dense_medcpt": "MedCPT dense", "dense_lsa": "LSA dense"}.get(name, name)


def budget_curve(m: dict, out: Path):
    rows = m["budget_curve"]
    B = [r["budget"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    first = _label(m["system_names"]["first_stage"])
    ax.plot(B, [r["hybrid_hit"] for r in rows], "-o", color=BLUE, lw=2, ms=6, label=f"{first} only (B slots)")
    ax.plot(B, [r["graph_hit"] for r in rows], "-o", color=ORANGE, lw=2, ms=6, label=f"{first} + entity graph (graph fills B//2 slots)")
    nm = [r["rewired_hit"]["null_mean"] for r in rows]
    lo = [r["rewired_hit"]["null_mean"] - 2 * r["rewired_hit"]["null_sd"] for r in rows]
    hi = [r["rewired_hit"]["null_mean"] + 2 * r["rewired_hit"]["null_sd"] for r in rows]
    ax.fill_between(B, lo, hi, color=GRAY, alpha=0.25, lw=0)
    ax.plot(B, nm, "--", color=GRAY, lw=2, label=f"{first} + rewired graph (mean ± 2 SD, {rows[0]['rewired_hit']['n_null']} nulls)")
    ax.set_xlabel("Retrieval budget B (abstracts passed to the reader)")
    ax.set_ylabel("Claims with a gold abstract in the budget")
    ax.set_xticks(B)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Same budget: does the graph find evidence the first stage misses?", fontsize=10.5, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "budget_curve.png")
    plt.close(fig)


def pipeline(m: dict, out: Path):
    rr = m["system_names"]["rerank"]
    order = [("bm25", "BM25"), ("dense_bge", "bge-small (general)"), ("dense_medcpt", "MedCPT dense"),
             ("hybrid", "Hybrid (RRF)"), ("hybrid+graph", "Hybrid + graph (RRF)"),
             (rr, _label(m["system_names"]["first_stage"]) + " → cross-encoder")]
    order = [(k, n) for k, n in order if k in m["doc_level"]]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    for ax, key, title in zip(axes, ["ndcg@10", "recall@1"], ["nDCG@10", "Recall@1"]):
        vals = [m["doc_level"][k][key] for k, _ in order]
        y = range(len(order))
        ax.barh(list(y), vals, color=[GRAY] * (len(order) - 1) + [BLUE], height=0.6)
        for i, v in zip(y, vals):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(list(y), [n for _, n in order])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.08)
        ax.set_title(title, fontsize=10.5, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"SciFact dev claims with evidence (n = {m['n_claims']}), corpus of {m['n_docs']:,} abstracts",
                 fontsize=9, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "pipeline.png")
    plt.close(fig)


def stance(m: dict, out: Path):
    systems = [("bm25", "BM25"), ("dense_medcpt", "MedCPT dense"), ("hybrid", "Hybrid"),
               (m["system_names"]["rerank"], _label(m["system_names"]["first_stage"]) + " → CE")]
    systems = [(k, n) for k, n in systems if f"{k} recall@1" in m["stance"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    w = 0.36
    for j, (lab, col) in enumerate([("SUPPORT", AQUA), ("CONTRADICT", ORANGE)]):
        vals = [m["stance"][f"{k} recall@1"][lab] for k, _ in systems]
        xs = [i + (j - 0.5) * w for i in range(len(systems))]
        ax.bar(xs, vals, width=w - 0.04, color=col, label=lab.title() + "ed claims" if lab == "SUPPORT" else "Contradicted claims")
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(systems)), [n for _, n in systems])
    ax.set_ylabel("Recall@1 (gold abstract ranked first)")
    ax.set_ylim(0, 1.18)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    ax.set_title("Recall@1 by claim label (95% CIs of the gaps all include 0; see README)", fontsize=10.5, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "stance.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = json.loads((Path(a.results) / "metrics.json").read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    budget_curve(m, out)
    pipeline(m, out)
    stance(m, out)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
