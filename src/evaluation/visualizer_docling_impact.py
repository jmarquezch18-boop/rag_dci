"""
visualizer_docling_impact.py
============================
Figura de análisis de impacto de Docling en el agente DCI.

Compara RAG 3 (corpus plain text) vs RAG 3.2 (corpus Docling) en 4 paneles:
  A. Scatter MRR por query (diagonal = paridad)
  B. Secciones disponibles por corpus (plain vs Docling)
  C. B.cov (oracle coverage) por query — distribución
  D. Delta MRR (RAG 3.2 - RAG 3) ordenado — queries que mejoró/empeoró
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Colores ──────────────────────────────────────────────────────────────────
_C_RAG3   = "#EF5350"   # rojo — RAG 3 plain
_C_RAG32  = "#7E57C2"   # violeta — RAG 3.2 Docling
_C_ABOVE  = "#26C6DA"   # cian — mejoró con Docling
_C_BELOW  = "#FF8A65"   # naranja — empeoró con Docling
_C_GRAY   = "#90A4AE"

# Secciones disponibles por corpus (datos empíricos del análisis)
_PLAIN_SECTIONS = {
    "abstract": 70, "introduction": 62, "results": 56,
    "methodology": 55, "discussion": 38, "limitations": 2,
}
_DOCLING_SECTIONS = {
    "abstract": 70, "introduction": 62, "results": 56,
    "conclusion": 56, "methodology": 55, "discussion": 38,
    "related_work": 36, "limitations": 2, "unknown": 70,
}


def _load_per_query(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return {q["query_id"]: q for q in data["per_query_results"]}


def plot_docling_impact(results_base: Path, out_path: Path) -> None:
    rag3_path  = results_base / "rag3"  / "metrics_dci.json"
    rag32_path = results_base / "rag3_2" / "metrics_dci.json"

    if not rag3_path.exists() or not rag32_path.exists():
        raise FileNotFoundError(
            f"Se necesitan {rag3_path} y {rag32_path}"
        )

    q3  = _load_per_query(rag3_path)
    q32 = _load_per_query(rag32_path)
    common = sorted(set(q3) & set(q32))

    mrr3   = np.array([q3[qid]["point_a_pre_curation"]["mrr"]  for qid in common])
    mrr32  = np.array([q32[qid]["point_a_pre_curation"]["mrr"] for qid in common])
    bcov3  = np.array([q3[qid]["point_b_post_curation"]["coverage"]  for qid in common])
    bcov32 = np.array([q32[qid]["point_b_post_curation"]["coverage"] for qid in common])
    delta  = mrr32 - mrr3

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Impacto de Docling en el Agente DCI\n"
        "RAG 3 (corpus plain text) vs RAG 3.2 (corpus Docling + DeepSeek-R1 14B)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── Panel A: Scatter MRR ─────────────────────────────────────────────────
    ax = axes[0, 0]
    colors = [_C_ABOVE if d > 0.001 else (_C_BELOW if d < -0.001 else _C_GRAY) for d in delta]
    ax.scatter(mrr3, mrr32, c=colors, alpha=0.7, s=30, zorder=3)
    lim = max(mrr3.max(), mrr32.max()) * 1.1
    ax.plot([0, lim], [0, lim], "--", color="#546E7A", lw=1.2, label="Paridad")
    ax.set_xlabel("MRR — RAG 3 (plain text)", fontsize=10)
    ax.set_ylabel("MRR — RAG 3.2 (Docling)", fontsize=10)
    ax.set_title("A. MRR por query: plain vs Docling", fontweight="bold")
    ax.set_xlim(-0.01, lim)
    ax.set_ylim(-0.01, lim)

    above = int((delta > 0.001).sum())
    below = int((delta < -0.001).sum())
    equal = len(delta) - above - below
    legend_handles = [
        mpatches.Patch(color=_C_ABOVE, label=f"Docling mejor ({above})"),
        mpatches.Patch(color=_C_BELOW, label=f"Docling peor ({below})"),
        mpatches.Patch(color=_C_GRAY,  label=f"Sin cambio ({equal})"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.97, 0.05,
        f"n={len(common)} queries",
        transform=ax.transAxes, ha="right", fontsize=8, color="#546E7A",
    )

    # ── Panel B: Secciones por corpus ────────────────────────────────────────
    ax = axes[0, 1]
    sections_order = [
        "abstract", "introduction", "methodology", "results",
        "discussion", "conclusion", "related_work", "limitations", "unknown",
    ]
    plain_vals   = [_PLAIN_SECTIONS.get(s, 0)   for s in sections_order]
    docling_vals = [_DOCLING_SECTIONS.get(s, 0) for s in sections_order]

    x = np.arange(len(sections_order))
    width = 0.38
    bars1 = ax.bar(x - width/2, plain_vals,   width, color=_C_RAG3,  alpha=0.85, label="Plain text")
    bars2 = ax.bar(x + width/2, docling_vals, width, color=_C_RAG32, alpha=0.85, label="Docling")
    ax.set_xticks(x)
    ax.set_xticklabels(sections_order, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("N.º de archivos de sección", fontsize=10)
    ax.set_title("B. Secciones disponibles por corpus", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    # Anotar totales
    ax.text(
        0.98, 0.96,
        f"Plain: 283 archivos\nDocling: 445 archivos (+57%)",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # ── Panel C: Distribución B.cov ─────────────────────────────────────────
    ax = axes[1, 0]
    bins = np.linspace(0, 1, 21)
    ax.hist(bcov3,  bins=bins, alpha=0.75, color=_C_RAG3,  label=f"RAG 3 plain   (B.cov=0.000)")
    ax.hist(bcov32, bins=bins, alpha=0.75, color=_C_RAG32, label=f"RAG 3.2 Docling (B.cov=0.149)")
    ax.set_xlabel("Oracle Coverage (B.cov) por query", fontsize=10)
    ax.set_ylabel("N.º de queries", fontsize=10)
    ax.set_title("C. Distribución de cobertura oracle (B.cov)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    n_nonzero32 = int((bcov32 > 0).sum())
    ax.text(
        0.98, 0.96,
        f"RAG 3.2 cubre {n_nonzero32}/{len(common)} queries\nRAG 3 cubre 0/{len(common)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # ── Panel D: Delta MRR ordenado ─────────────────────────────────────────
    ax = axes[1, 1]
    delta_sorted = np.sort(delta)
    x_idx = np.arange(len(delta_sorted))
    bar_colors = [_C_ABOVE if d > 0 else (_C_BELOW if d < 0 else _C_GRAY) for d in delta_sorted]
    ax.bar(x_idx, delta_sorted, color=bar_colors, alpha=0.8, width=1.0)
    ax.axhline(0, color="#546E7A", lw=1.5, linestyle="--")
    ax.axhline(delta.mean(), color="#FDD835", lw=1.5, linestyle="-",
               label=f"Δ medio = {delta.mean():+.4f}")
    ax.set_xlabel("Queries ordenadas por ΔMRR", fontsize=10)
    ax.set_ylabel("ΔMRR (RAG 3.2 − RAG 3)", fontsize=10)
    ax.set_title("D. Delta MRR por query (Docling vs plain)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(
        0.02, 0.96,
        f"Mejora: {above} queries\nEmpeora: {below} queries",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # ── Nota al pie ──────────────────────────────────────────────────────────
    fig.text(
        0.5, -0.01,
        "Nota: B.cov usa containment recall (threshold=0.5) para RAG 3.2 y Jaccard (threshold=0.75) para RAG 3.\n"
        "RAG 3 B.cov=0 se debe a incompatibilidad de Jaccard con chunks DCI grandes (~300 tokens) vs GT pequeño (~50 tokens).",
        ha="center", fontsize=8, color="#546E7A", style="italic",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SUCCESS | Figura guardada en {out_path}")


if __name__ == "__main__":
    base = Path("results")
    out  = base / "docling_impact_analysis.png"
    plot_docling_impact(base, out)
