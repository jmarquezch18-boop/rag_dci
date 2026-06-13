"""
visualizer_rag2.py
==================
Genera las figuras de análisis de RAG 2 y la comparación RAG 1 vs RAG 2.

Posición en el pipeline: modo --mode visualize --pipeline rag2.
Decisión clave: NUNCA importa torch ni ningún modelo. Solo JSON → PNG.

Figuras generadas (todas 300 dpi):
  results/rag2/round_0/figures/
    1. three_point_metrics.png      — A/B/C lado a lado (figura más importante)
    2. retrieval_metrics.png        — Punto A: MRR, Recall@k, NDCG@5
    3. curation_impact.png          — Comparación Punto A vs Punto B
    4. generation_metrics.png       — Punto C: ROUGE-L, BERTScore, faithfulness
    5. section_filter_heatmap.png   — Recall@5 por sección filtrada
    6. coverage_distribution.png    — histograma de coverage Punto B
  results/rag2/round_n/figures/
    1-6. mismas que round_0
    7. active_learning_curve.png    — MRR del reranker por ronda

  results/comparison_rag1_vs_rag2.png — si existen ambos JSONs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger


class RAG2Visualizer:
    """Lee los JSONs de RAG 2 y genera todas las figuras."""

    def __init__(self, results_base: str, round_dir: str = "round_0"):
        self.rag2_dir = Path(results_base) / "rag2" / round_dir
        self.figures_dir = self.rag2_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.rag1_dir = Path(results_base) / "rag1"
        self.results_base = Path(results_base)
        self.round_dir = round_dir
        plt.style.use("seaborn-v0_8-whitegrid")

    def load_results(self) -> dict:
        """
        Carga metrics_post_llm.json de RAG 2.

        Raises:
            FileNotFoundError: Si el JSON no existe.
        """
        json_path = self.rag2_dir / "metrics_post_llm.json"
        if not json_path.exists():
            mode = "evaluate_r0" if self.round_dir == "round_0" else "evaluate_rn"
            raise FileNotFoundError(
                f"No se encontró {json_path}.\n"
                f"Corre primero: python main.py --pipeline rag2 --mode {mode}"
            )
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)

    def visualize_all(self) -> None:
        """Genera todas las figuras del round actual. Punto de entrada del modo visualize."""
        logger.info("Cargando resultados RAG 2 ({}) para visualización", self.round_dir)
        data = self.load_results()
        agg = data["aggregate"]
        per_query = data["per_query_results"]

        self._plot_three_point_metrics(agg)
        self._plot_retrieval_metrics(agg["point_a"])
        self._plot_curation_impact(agg)
        self._plot_generation_metrics(agg["point_c"])
        self._plot_section_filter_heatmap(per_query)
        self._plot_coverage_distribution(per_query)

        if self.round_dir == "round_n":
            curve_path = self.rag2_dir / "active_learning_curve.json"
            if curve_path.exists():
                with open(curve_path, encoding="utf-8") as f:
                    curve_data = json.load(f)
                self._plot_active_learning_curve(curve_data)

        logger.success("Figuras RAG 2 guardadas en {}", self.figures_dir)

        # Comparación si existen ambos RAGs
        self._plot_comparison_if_available()

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = self.figures_dir / name
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Figura guardada: {}", path.name)

    def _plot_three_point_metrics(self, agg: dict) -> None:
        """
        Gráfica de barras agrupadas mostrando métricas en los 3 puntos A/B/C.

        Esta es la figura central del estudio SAIL: muestra en una sola vista
        cómo la curación (A→B) y la generación (B→C) transforman la calidad.
        """
        metrics_ab = ["mrr", "recall_at_5", "precision_at_5", "ndcg_at_5"]
        labels = ["MRR", "Recall@5", "Precision@5", "NDCG@5"]

        a_vals = [agg["point_a"].get(m, 0.0) for m in metrics_ab]
        b_vals = [agg["point_b"].get(m, 0.0) for m in metrics_ab]

        # Punto C tiene métricas diferentes → gráfica separada por tipo
        c_metrics = ["rouge_l", "bert_score_f1", "faithfulness"]
        c_labels = ["ROUGE-L", "BERTScore-F1", "Faithfulness"]
        c_vals = [agg["point_c"].get(m) or 0.0 for m in c_metrics]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        x = np.arange(len(labels))
        width = 0.35

        # Panel izquierdo: A vs B (métricas de recuperación)
        ax = axes[0]
        bars_a = ax.bar(x - width / 2, a_vals, width, label="Punto A (pre-curación)",
                        color="#2196F3", edgecolor="white", alpha=0.9)
        bars_b = ax.bar(x + width / 2, b_vals, width, label="Punto B (post-curación)",
                        color="#4CAF50", edgecolor="white", alpha=0.9)
        for bar, val in zip(list(bars_a) + list(bars_b), a_vals + b_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Score")
        ax.set_title("Recuperación: Punto A vs B (Impacto Curación)", fontweight="bold")
        ax.legend(fontsize=9)

        # Panel derecho: Punto C (métricas de generación)
        ax = axes[1]
        colors = ["#FF9800", "#9C27B0", "#F44336"]
        bars_c = ax.bar(np.arange(len(c_labels)), c_vals, 0.5, color=colors, edgecolor="white")
        for bar, val in zip(bars_c, c_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(np.arange(len(c_labels)))
        ax.set_xticklabels(c_labels, fontsize=10)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Score")
        ax.set_title("Generación: Punto C (post-LLM)", fontweight="bold")

        fig.suptitle("SAIL-RAG 2 — Los 3 Puntos de Evaluación", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "three_point_metrics.png")

    def _plot_retrieval_metrics(self, point_a: dict) -> None:
        """Barras horizontales para métricas de Punto A incluyendo NDCG@5."""
        ordered_keys = (
            ["mrr"]
            + [f"recall_at_{k}" for k in sorted(
                int(k.split("_")[-1]) for k in point_a if k.startswith("recall_at_")
            )]
            + ["precision_at_5", "ndcg_at_5"]
        )
        labels = [k.replace("_", " ").upper() for k in ordered_keys]
        values = [point_a.get(k, 0.0) for k in ordered_keys]

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(labels)))
        bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Score", fontsize=11)
        ax.set_title("RAG 2 — Métricas de Recuperación Punto A (pre-curación)",
                     fontweight="bold", fontsize=12)
        plt.tight_layout()
        self._save(fig, "retrieval_metrics.png")

    def _plot_curation_impact(self, agg: dict) -> None:
        """Barras agrupadas comparando Punto A vs Punto B para medir impacto de curación."""
        shared_keys = ["mrr", "recall_at_5", "precision_at_5"]
        labels = ["MRR", "Recall@5", "Precision@5"]
        a_vals = [agg["point_a"].get(k, 0.0) for k in shared_keys]
        b_vals = [agg["point_b"].get(k, 0.0) for k in shared_keys]

        # Agregar coverage que es exclusivo de Punto B
        labels.append("Coverage")
        a_vals.append(0.0)  # Coverage no aplica en Punto A
        b_vals.append(agg["point_b"].get("coverage", 0.0))

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 5))
        bars_a = ax.bar(x - width / 2, a_vals, width, label="Punto A (pre-curación)",
                        color="#2196F3", edgecolor="white")
        bars_b = ax.bar(x + width / 2, b_vals, width, label="Punto B (post-curación)",
                        color="#4CAF50", edgecolor="white")

        for bars, vals in [(bars_a, a_vals), (bars_b, b_vals)]:
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Impacto de Curación: Punto A → Punto B", fontweight="bold", fontsize=12)
        ax.legend(fontsize=10)
        plt.tight_layout()
        self._save(fig, "curation_impact.png")

    def _plot_generation_metrics(self, point_c: dict) -> None:
        """Barras para ROUGE-L, BERTScore-F1 y faithfulness coseno del Punto C."""
        metrics = ["rouge_l", "bert_score_f1", "faithfulness"]
        labels = ["ROUGE-L", "BERTScore-F1", "Faithfulness\n(coseno)"]
        values = [point_c.get(m) or 0.0 for m in metrics]
        colors = ["#FF9800", "#9C27B0", "#F44336"]

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.45)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("RAG 2 — Métricas de Generación (Punto C)", fontweight="bold", fontsize=12)
        plt.tight_layout()
        self._save(fig, "generation_metrics.png")

    def _plot_section_filter_heatmap(self, per_query: list[dict]) -> None:
        """
        Heatmap de Recall@5 por tipo de sección filtrada.

        Agrupa queries por la sección usada como filtro (o "sin filtro")
        y muestra cómo varía Recall@5 por sección. Permite identificar
        qué secciones son más informativas para distintos tipos de queries.
        """
        section_recalls: dict[str, list[float]] = {}
        for r in per_query:
            sec = r.get("section_filter") or "sin_filtro"
            recall = r.get("point_a_pre_curation", {}).get("recall_at_5", 0.0)
            section_recalls.setdefault(sec, []).append(recall)

        if len(section_recalls) < 2:
            logger.info("Pocas secciones distintas para heatmap ({}) — omitiendo", len(section_recalls))
            return

        sections = sorted(section_recalls.keys())
        mean_recalls = [np.mean(section_recalls[s]) for s in sections]
        counts = [len(section_recalls[s]) for s in sections]

        fig, ax = plt.subplots(figsize=(max(6, len(sections) * 1.2), 4))
        data_matrix = np.array(mean_recalls).reshape(1, -1)
        im = ax.imshow(data_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Recall@5 promedio")
        ax.set_xticks(range(len(sections)))
        ax.set_xticklabels(
            [f"{s}\n(n={counts[i]})" for i, s in enumerate(sections)],
            fontsize=9, rotation=30, ha="right",
        )
        ax.set_yticks([])
        for j, val in enumerate(mean_recalls):
            ax.text(j, 0, f"{val:.3f}", ha="center", va="center", fontsize=10, fontweight="bold",
                    color="black")
        ax.set_title("Recall@5 por Sección Filtrada", fontweight="bold", fontsize=12)
        plt.tight_layout()
        self._save(fig, "section_filter_heatmap.png")

    def _plot_coverage_distribution(self, per_query: list[dict]) -> None:
        """Histograma de coverage del Punto B por query."""
        coverages = [
            r.get("point_b_post_curation", {}).get("coverage", 0.0)
            for r in per_query
        ]
        if not coverages:
            return

        coverages_arr = np.array(coverages)
        mean_cov = float(np.mean(coverages_arr))

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(coverages_arr, bins=20, color="#4CAF50", edgecolor="white", alpha=0.85)
        ax.axvline(mean_cov, color="red", linestyle="--", linewidth=1.8,
                   label=f"Media: {mean_cov:.3f}")
        ax.set_xlabel("Coverage (fracción ground truth capturada)", fontsize=11)
        ax.set_ylabel("Frecuencia", fontsize=11)
        ax.set_title("Distribución de Coverage Punto B por Query", fontweight="bold", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        plt.tight_layout()
        self._save(fig, "coverage_distribution.png")

    def _plot_active_learning_curve(self, curve_data: list[dict]) -> None:
        """Línea de MRR del reranker por ronda de fine-tuning (solo round_n)."""
        if not curve_data:
            return

        rounds = [r["round"] for r in curve_data]
        mrr_vals = [r.get("mrr", 0.0) for r in curve_data]
        losses = [r.get("final_loss", 0.0) for r in curve_data]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color_mrr = "steelblue"
        ax1.plot(rounds, mrr_vals, marker="o", color=color_mrr, linewidth=2.5,
                 markersize=8, label="MRR post-reranker")
        ax1.fill_between(rounds, mrr_vals, alpha=0.15, color=color_mrr)
        ax1.set_xlabel("Ronda de Fine-tuning", fontsize=11)
        ax1.set_ylabel("MRR", color=color_mrr, fontsize=11)
        ax1.tick_params(axis="y", labelcolor=color_mrr)
        ax1.set_ylim(0, max(max(mrr_vals) * 1.15, 0.1))

        ax2 = ax1.twinx()
        color_loss = "#E53935"
        ax2.plot(rounds, losses, marker="s", color=color_loss, linewidth=1.8,
                 linestyle="--", markersize=6, alpha=0.8, label="Loss fine-tuning")
        ax2.set_ylabel("Loss contrastivo", color=color_loss, fontsize=11)
        ax2.tick_params(axis="y", labelcolor=color_loss)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right")

        plt.title("Curva de Aprendizaje Activo — MRR por Ronda", fontweight="bold", fontsize=12)
        plt.tight_layout()
        self._save(fig, "active_learning_curve.png")

    def _plot_comparison_if_available(self) -> None:
        """
        Genera comparación RAG 1 vs RAG 2 solo si existen ambos JSONs.

        Lee results/rag1/metrics_post_llm.json y results/rag2/round_0/
        y results/rag2/round_n/ (si existe). Compara MRR pre-LLM y
        métricas de generación lado a lado.
        """
        rag1_path = self.rag1_dir / "metrics_post_llm.json"
        rag2_r0_path = self.results_base / "rag2" / "round_0" / "metrics_post_llm.json"
        rag2_rn_path = self.results_base / "rag2" / "round_n" / "metrics_post_llm.json"

        if not rag1_path.exists() or not rag2_r0_path.exists():
            logger.info(
                "Comparación RAG1 vs RAG2 omitida: faltan JSONs ({} o {})",
                rag1_path.name, rag2_r0_path.name,
            )
            return

        with open(rag1_path, encoding="utf-8") as f:
            rag1 = json.load(f)
        with open(rag2_r0_path, encoding="utf-8") as f:
            rag2_r0 = json.load(f)

        rag2_rn = None
        if rag2_rn_path.exists():
            with open(rag2_rn_path, encoding="utf-8") as f:
                rag2_rn = json.load(f)

        self._plot_comparison(rag1, rag2_r0, rag2_rn)

    def _plot_comparison(self, rag1: dict, rag2_r0: dict, rag2_rn: Optional[dict]) -> None:
        """Genera results/comparison_rag1_vs_rag2.png."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        def get_safe(d: dict, *keys, default=0.0):
            v = d
            for k in keys:
                v = v.get(k, {}) if isinstance(v, dict) else {}
            return v if isinstance(v, (int, float)) and v is not None else default

        # MRR pre-LLM
        ax = axes[0]
        conditions = ["RAG 1\nPunto A"]
        mrr_vals = [get_safe(rag1, "aggregate", "point_a", "mrr")]
        conditions.append("RAG 2 R0\nPunto A")
        mrr_vals.append(get_safe(rag2_r0, "aggregate", "point_a", "mrr"))
        conditions.append("RAG 2 R0\nPunto B")
        mrr_vals.append(get_safe(rag2_r0, "aggregate", "point_b", "mrr"))
        if rag2_rn:
            conditions.append("RAG 2 RN\nPunto A")
            mrr_vals.append(get_safe(rag2_rn, "aggregate", "point_a", "mrr"))
            conditions.append("RAG 2 RN\nPunto B")
            mrr_vals.append(get_safe(rag2_rn, "aggregate", "point_b", "mrr"))

        colors = ["#90CAF9", "#42A5F5", "#1565C0"] + (["#66BB6A", "#2E7D32"] if rag2_rn else [])
        bars = ax.bar(range(len(conditions)), mrr_vals, color=colors[:len(conditions)],
                      edgecolor="white")
        for bar, val in zip(bars, mrr_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title("MRR pre-LLM", fontweight="bold")
        ax.set_ylabel("MRR")

        # ROUGE-L
        ax = axes[1]
        r_conditions = ["RAG 1"]
        r_vals = [get_safe(rag1, "aggregate", "point_b", "rouge_l")]
        r_conditions.append("RAG 2 R0")
        r_vals.append(get_safe(rag2_r0, "aggregate", "point_c", "rouge_l"))
        if rag2_rn:
            r_conditions.append("RAG 2 RN")
            r_vals.append(get_safe(rag2_rn, "aggregate", "point_c", "rouge_l"))

        r_colors = ["#90CAF9", "#42A5F5"] + (["#66BB6A"] if rag2_rn else [])
        bars = ax.bar(range(len(r_conditions)), r_vals, color=r_colors,
                      edgecolor="white", width=0.5)
        for bar, val in zip(bars, r_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(r_conditions)))
        ax.set_xticklabels(r_conditions, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_title("ROUGE-L post-LLM", fontweight="bold")
        ax.set_ylabel("ROUGE-L")

        # BERTScore
        ax = axes[2]
        bs_conditions = list(r_conditions)
        bs_rag1 = get_safe(rag1, "aggregate", "point_b", "bert_score_f1")
        bs_r0 = get_safe(rag2_r0, "aggregate", "point_c", "bert_score_f1")
        bs_vals = [bs_rag1, bs_r0]
        if rag2_rn:
            bs_vals.append(get_safe(rag2_rn, "aggregate", "point_c", "bert_score_f1"))

        bs_colors = ["#90CAF9", "#42A5F5"] + (["#66BB6A"] if rag2_rn else [])
        bars = ax.bar(range(len(bs_conditions)), bs_vals, color=bs_colors,
                      edgecolor="white", width=0.5)
        for bar, val in zip(bars, bs_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(bs_conditions)))
        ax.set_xticklabels(bs_conditions, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_title("BERTScore-F1 post-LLM", fontweight="bold")
        ax.set_ylabel("BERTScore-F1")

        fig.suptitle("Comparación RAG 1 vs RAG 2", fontsize=14, fontweight="bold")
        plt.tight_layout()
        comp_path = self.results_base / "comparison_rag1_vs_rag2.png"
        fig.savefig(comp_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.success("Comparación guardada en {}", comp_path)
