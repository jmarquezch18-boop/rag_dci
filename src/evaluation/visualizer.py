"""
visualizer.py
=============
Genera las 5 figuras de análisis de RAG 1 leyendo solo el JSON de resultados.

Posición en el pipeline: paso final del modo --mode visualize.
Decisión clave: este módulo NUNCA importa torch, sentence-transformers
ni ningún modelo. Solo lee JSON y escribe PNG. Esto garantiza que se
pueden regenerar todas las figuras sin cargar ningún modelo en VRAM.

Figuras generadas (todas en results/rag1/figures/, 300 dpi):
  1. retrieval_metrics.png  — barras horizontales Punto A
  2. generation_metrics.png — barras ROUGE-L y BERTScore-F1
  3. recall_k_curve.png     — curva Recall@k
  4. score_distribution.png — histograma scores coseno top-20
  5. query_length_vs_mrr.png — scatter longitud query vs MRR
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # backend sin ventana — compatible con entornos sin display
import matplotlib.pyplot as plt
from loguru import logger


class RAG1Visualizer:
    """Lee metrics_post_llm.json y genera todas las figuras de RAG 1."""

    def __init__(self, results_base: str):
        self.rag1_dir = Path(results_base) / "rag1"
        self.figures_dir = self.rag1_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid")

    def load_results(self) -> dict:
        """
        Carga el JSON de resultados del pipeline RAG 1.

        Raises:
            FileNotFoundError: Si metrics_post_llm.json no existe.
        """
        json_path = self.rag1_dir / "metrics_post_llm.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"No se encontró {json_path}.\n"
                "Corre primero: python main.py --pipeline rag1 --mode evaluate"
            )
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)

    def visualize_all(self) -> None:
        """Genera las 5 figuras. Punto de entrada del modo visualize."""
        logger.info("Cargando resultados para visualización")
        data = self.load_results()
        agg = data["aggregate"]
        per_query = data["per_query_results"]

        self._plot_retrieval_metrics(agg["point_a"])
        self._plot_generation_metrics(agg["point_b"])
        self._plot_recall_k_curve(agg["point_a"])
        self._plot_score_distribution(per_query)
        self._plot_query_length_vs_mrr(per_query)

        logger.success("5 figuras guardadas en {}", self.figures_dir)

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = self.figures_dir / name
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Figura guardada: {}", path.name)

    def _plot_retrieval_metrics(self, point_a: dict) -> None:
        """Barras horizontales para todas las métricas de Punto A."""
        ordered_keys = (
            ["mrr"]
            + [f"recall_at_{k}" for k in sorted(
                int(k.split("_")[-1]) for k in point_a if k.startswith("recall_at_")
            )]
            + ["precision_at_5"]
        )
        labels = [k.replace("_", " ").upper() for k in ordered_keys]
        values = [point_a.get(k, 0.0) for k in ordered_keys]

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(labels)))
        bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=9,
            )

        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Score", fontsize=11)
        ax.set_title("RAG 1 Baseline — Métricas de Recuperación (Punto A)", fontweight="bold", fontsize=12)
        plt.tight_layout()
        self._save(fig, "retrieval_metrics.png")

    def _plot_generation_metrics(self, point_b: dict) -> None:
        """Barras para ROUGE-L y BERTScore-F1."""
        labels = ["ROUGE-L", "BERTScore-F1"]
        values = [
            point_b.get("rouge_l") or 0.0,
            point_b.get("bert_score_f1") or 0.0,
        ]
        colors = ["#2196F3", "#4CAF50"]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.4)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("RAG 1 Baseline — Métricas de Generación (Punto B)", fontweight="bold", fontsize=12)
        plt.tight_layout()
        self._save(fig, "generation_metrics.png")

    def _plot_recall_k_curve(self, point_a: dict) -> None:
        """Línea Recall@k para todos los valores de k."""
        k_vals = sorted(
            int(k.split("_")[-1]) for k in point_a if k.startswith("recall_at_")
        )
        recall_vals = [point_a.get(f"recall_at_{k}", 0.0) for k in k_vals]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(k_vals, recall_vals, marker="o", color="steelblue",
                linewidth=2.5, markersize=8, label="Recall@k")
        ax.fill_between(k_vals, recall_vals, alpha=0.15, color="steelblue")

        for k, r in zip(k_vals, recall_vals):
            ax.annotate(
                f"{r:.3f}", (k, r),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9,
            )

        ax.set_xlabel("k", fontsize=11)
        ax.set_ylabel("Recall@k", fontsize=11)
        ax.set_title("RAG 1 Baseline — Curva Recall@k", fontweight="bold", fontsize=12)
        ax.set_xticks(k_vals)
        ax.set_ylim(-0.02, 1.08)
        ax.legend(fontsize=10)
        plt.tight_layout()
        self._save(fig, "recall_k_curve.png")

    def _plot_score_distribution(self, per_query: list[dict]) -> None:
        """Histograma de scores de similitud coseno de los top-20 recuperados."""
        all_scores = [
            chunk["score"]
            for q in per_query
            for chunk in q.get("retrieved_chunks", [])
            if "score" in chunk
        ]

        if not all_scores:
            logger.warning("No hay scores de similitud para graficar")
            return

        scores_arr = np.array(all_scores)
        mean_score = float(np.mean(scores_arr))

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(scores_arr, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(mean_score, color="red", linestyle="--", linewidth=1.8,
                   label=f"Media: {mean_score:.3f}")

        ax.set_xlabel("Score de Similitud Coseno", fontsize=11)
        ax.set_ylabel("Frecuencia", fontsize=11)
        ax.set_title("RAG 1 Baseline — Distribución de Scores (Top-20)", fontweight="bold", fontsize=12)
        ax.legend(fontsize=10)
        plt.tight_layout()
        self._save(fig, "score_distribution.png")

    def _plot_query_length_vs_mrr(self, per_query: list[dict]) -> None:
        """Scatter: longitud de query (palabras) vs MRR individual."""
        lengths = [len(q["query"].split()) for q in per_query]
        mrrs = [q["point_a_pre_llm"].get("mrr", 0.0) for q in per_query]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(lengths, mrrs, color="steelblue", alpha=0.55,
                   edgecolors="white", linewidths=0.5, s=55)

        # Línea de tendencia solo si hay variación en x
        if len(set(lengths)) > 2:
            z = np.polyfit(lengths, mrrs, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(lengths), max(lengths), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=1.8, label="Tendencia lineal")
            ax.legend(fontsize=10)

        ax.set_xlabel("Longitud de Query (palabras)", fontsize=11)
        ax.set_ylabel("MRR", fontsize=11)
        ax.set_title("RAG 1 Baseline — Longitud de Query vs MRR", fontweight="bold", fontsize=12)
        ax.set_ylim(-0.05, 1.08)
        plt.tight_layout()
        self._save(fig, "query_length_vs_mrr.png")
