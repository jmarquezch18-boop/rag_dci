"""
visualizer_rag3.py
==================
Genera las figuras de análisis de RAG 3 y la comparación de 3 vías
RAG 1 vs RAG 3 vs RAG 2.

Figuras:
  results/rag3/figures/
    1. three_point_metrics.png   — puntos A/B/C lado a lado
    2. retrieval_metrics.png     — Punto A: MRR, Recall@k, NDCG@5
    3. curation_impact.png       — Punto A vs Punto B (efecto de la curación)
    4. generation_metrics.png    — Punto C: ROUGE-L, BERTScore, faithfulness cosine
    5. coverage_distribution.png — histograma de coverage Punto B

  results/comparison_rag1_rag3_rag2.png — comparación 3 vías (si existen los 3 JSONs)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


class RAG3Visualizer:
    """Lee results/rag3/metrics_post_llm.json y genera todas las figuras."""

    def __init__(self, results_base: str):
        self.rag3_dir = Path(results_base) / "rag3"
        self.figures_dir = self.rag3_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.rag1_dir = Path(results_base) / "rag1"
        self.rag2_dir = Path(results_base) / "rag2" / "round_n"
        self.results_base = Path(results_base)
        plt.style.use("seaborn-v0_8-whitegrid")

    def load_results(self) -> dict:
        json_path = self.rag3_dir / "metrics_post_llm.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"No se encontró {json_path}.\n"
                "Corre primero: python main.py --pipeline rag3 --mode evaluate"
            )
        return json.loads(json_path.read_text(encoding="utf-8"))

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = self.figures_dir / name
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Figura guardada: {}", name)

    def visualize_all(self) -> None:
        data = self.load_results()
        agg = data["aggregate"]
        pq = data["per_query_results"]

        self._plot_three_point(agg)
        self._plot_retrieval(agg)
        self._plot_curation_impact(agg)
        self._plot_generation(agg)
        self._plot_coverage_dist(pq)

        logger.success("Figuras RAG 3 guardadas en {}", self.figures_dir)

        self._plot_three_way_comparison()

    # ── Individual figures ────────────────────────────────────────────────────

    def _plot_three_point(self, agg: dict) -> None:
        pa = agg["point_a"]
        pb = agg["point_b"]
        pc = agg["point_c"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        color = "#FF8A65"

        def bar(ax, metrics, title):
            keys = list(metrics.keys())
            vals = [metrics[k] or 0 for k in keys]
            ax.barh(keys, vals, color=color, edgecolor="white")
            for i, v in enumerate(vals):
                ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
            ax.set_xlim(0, 1.15)
            ax.set_title(title, fontweight="bold")

        bar(axes[0], {k: pa[k] for k in ["mrr", "recall_at_5", "ndcg_at_5", "precision_at_5"]},
            "Point A — Pre-curation\n(Retrieval Quality)")
        bar(axes[1], {k: pb[k] for k in ["mrr", "recall_at_5", "coverage", "ndcg_at_5"]},
            "Point B — Post-curation\n(Curation Effect)")
        bar(axes[2], {k: pc[k] for k in ["rouge_l", "bert_score_f1", "faithfulness"]},
            "Point C — Post-LLM\n(Answer Quality)")

        fig.suptitle("RAG 3 — Curation Only (No Reranker)\nThree-Point Evaluation",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "three_point_metrics.png")

    def _plot_retrieval(self, agg: dict) -> None:
        pa = agg["point_a"]
        k_metrics = {k: v for k, v in pa.items() if k.startswith("recall_at_")}
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        color = "#FF8A65"

        axes[0].bar(["MRR", "NDCG@5", "P@5"],
                    [pa["mrr"], pa["ndcg_at_5"], pa["precision_at_5"]],
                    color=color, edgecolor="white")
        axes[0].set_ylim(0, 1.0)
        axes[0].set_title("Ranking Metrics (Point A)", fontweight="bold")

        labels = [k.replace("recall_at_", "R@") for k in k_metrics]
        axes[1].plot(labels, list(k_metrics.values()), "o-", color=color, linewidth=2)
        axes[1].set_ylim(0, 1.0)
        axes[1].set_title("Recall@k Curve (Point A)", fontweight="bold")
        axes[1].set_xlabel("k")

        fig.suptitle("RAG 3 — Retrieval Quality (Pre-Curation)", fontweight="bold")
        plt.tight_layout()
        self._save(fig, "retrieval_metrics.png")

    def _plot_curation_impact(self, agg: dict) -> None:
        pa = agg["point_a"]
        pb = agg["point_b"]
        metrics = ["mrr", "recall_at_5", "ndcg_at_5"]
        labels = ["MRR", "Recall@5", "NDCG@5"]
        x = np.arange(len(metrics))
        w = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - w / 2, [pa[m] for m in metrics], w, label="Pre-curation (A)",
               color="#90CAF9", edgecolor="white")
        ax.bar(x + w / 2, [pb.get(m, 0) for m in metrics], w, label="Post-curation (B)",
               color="#FF8A65", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.set_title("RAG 3 — Curation Impact (A vs B)", fontweight="bold")
        ax.set_ylabel("Score")

        # Coverage annotation
        cov = pb.get("coverage", 0)
        ax.text(0.98, 0.95, f"Coverage: {cov:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout()
        self._save(fig, "curation_impact.png")

    def _plot_generation(self, agg: dict) -> None:
        pc = agg["point_c"]
        metrics = {"ROUGE-L": pc["rouge_l"],
                   "BERTScore-F1": pc.get("bert_score_f1") or 0,
                   "Faithfulness\n(cosine)": pc.get("faithfulness") or 0}
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(metrics.keys(), metrics.values(), color="#FF8A65", edgecolor="white")
        for bar, val in zip(bars, metrics.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.set_title("RAG 3 — Generation Metrics (Point C)", fontweight="bold")
        ax.set_ylabel("Score")
        plt.tight_layout()
        self._save(fig, "generation_metrics.png")

    def _plot_coverage_dist(self, pq: list[dict]) -> None:
        coverages = [r.get("point_b_post_curation", {}).get("coverage", 0) or 0 for r in pq]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(coverages, bins=20, color="#FF8A65", edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(coverages), color="#BF360C", linestyle="--",
                   label=f"Mean: {np.mean(coverages):.3f}")
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Count")
        ax.set_title("RAG 3 — Coverage Distribution (Point B)", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        self._save(fig, "coverage_distribution.png")

    # ── 3-way comparison ──────────────────────────────────────────────────────

    def _plot_three_way_comparison(self) -> None:
        rag1_path = self.rag1_dir / "metrics_post_llm.json"
        rag3_path = self.rag3_dir / "metrics_post_llm.json"
        rag2_path = self.rag2_dir / "metrics_post_llm.json"

        if not all(p.exists() for p in [rag1_path, rag3_path, rag2_path]):
            logger.info("Comparación 3 vías omitida — faltan JSONs de RAG 1, 2 o 3")
            return

        rag1 = json.loads(rag1_path.read_text(encoding="utf-8"))
        rag3 = json.loads(rag3_path.read_text(encoding="utf-8"))
        rag2 = json.loads(rag2_path.read_text(encoding="utf-8"))

        def _get(data, *keys, default=0):
            v = data["aggregate"]
            for k in keys:
                v = v.get(k, {}) if isinstance(v, dict) else {}
            return v if isinstance(v, (int, float)) and v is not None else default

        # RAG 1 structure: point_a, point_b (no point_c with faithfulness)
        r1_mrr      = _get(rag1, "point_a", "mrr")
        r1_rec5     = _get(rag1, "point_a", "recall_at_5")
        r1_rouge    = _get(rag1, "point_b", "rouge_l")
        r1_bert     = _get(rag1, "point_b", "bert_score_f1")

        r3_mrr      = _get(rag3, "point_a", "mrr")
        r3_rec5     = _get(rag3, "point_a", "recall_at_5")
        r3_cov      = _get(rag3, "point_b", "coverage")
        r3_rouge    = _get(rag3, "point_c", "rouge_l")
        r3_bert     = _get(rag3, "point_c", "bert_score_f1")

        r2_mrr      = _get(rag2, "point_a", "mrr")
        r2_rec5     = _get(rag2, "point_a", "recall_at_5")
        r2_cov      = _get(rag2, "point_b", "coverage")
        r2_rouge    = _get(rag2, "point_c", "rouge_l")
        r2_bert     = _get(rag2, "point_c", "bert_score_f1")

        labels = ["RAG 1\n(Baseline)", "RAG 3\n(Curation only)", "RAG 2\n(Reranker+Curation)"]
        colors = ["#90CAF9", "#FF8A65", "#66BB6A"]
        x = np.arange(len(labels))
        w = 0.25

        metrics_groups = [
            ("MRR (Point A)",       [r1_mrr,   r3_mrr,   r2_mrr]),
            ("Recall@5 (Point A)",  [r1_rec5,  r3_rec5,  r2_rec5]),
            ("ROUGE-L (Point C)",   [r1_rouge, r3_rouge, r2_rouge]),
            ("BERTScore-F1 (C)",    [r1_bert,  r3_bert,  r2_bert]),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        for ax, (title, vals) in zip(axes, metrics_groups):
            bars = ax.bar(x, vals, color=colors, edgecolor="white", width=0.6)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylim(0, 1.15)
            ax.set_title(title, fontweight="bold")

        fig.suptitle(
            "RAG 1 vs RAG 3 vs RAG 2 — Ablation Study\n"
            "(Curation impact isolated from reranker)",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        out = self.results_base / "comparison_rag1_rag3_rag2.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.success("Comparación 3 vías guardada en {}", out)
