"""
visualizer_rag3_dci.py
======================
Genera las figuras de análisis del pipeline RAG 3 DCI + SAIL-RAG.

Lee results/rag3/metrics_dci.json (generado por rag3_dci_sail.py).
Genera figuras PNG 300dpi en results/rag3/figures/ y la comparación
final entre los 3 RAGs en results/final_comparison_rag1_rag2_rag3.png.

Figuras por-RAG3:
  1. three_point_comparison.png      — MRR, Recall@5, coverage para A/B/C
  2. agent_iterations_distribution.png — histograma de iteraciones por query
  3. tool_usage.png                  — frecuencia de uso de cada herramienta
  4. think_vs_mrr.png               — scatter: longitud de think vs MRR Punto A
  5. ragas_scores.png               — métricas RAGAS (placeholders en v1)
  6. deepeval_scores.png            — métricas DeepEval (placeholders en v1)

Figura comparativa (si existen JSONs de los 3 RAGs):
  results/final_comparison_rag1_rag2_rag3.png
    MRR pre-curación, Recall@5, ROUGE-L, RAGAS faithfulness
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

_COLOR_DCI = "#AB47BC"   # morado — RAG 3 DCI
_COLOR_RAG1 = "#42A5F5"  # azul — RAG 1
_COLOR_RAG2 = "#66BB6A"  # verde — RAG 2


class RAG3DCIVisualizer:
    """Lee metrics_dci.json y genera todas las figuras del pipeline DCI."""

    def __init__(self, results_base: str):
        """
        Args:
            results_base: Directorio raíz de resultados (config["paths"]["results"]).
        """
        self.results_base = Path(results_base)
        self.rag3_dir = self.results_base / "rag3"
        self.figures_dir = self.rag3_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.rag1_dir = self.results_base / "rag1"
        self.rag2_dir = self.results_base / "rag2" / "round_n"

        plt.style.use("seaborn-v0_8-whitegrid")

    # ── Carga de datos ────────────────────────────────────────────────────────

    def _load_dci(self) -> dict:
        path = self.rag3_dir / "metrics_dci.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró {path}.\n"
                "Corre: python main.py --pipeline rag3 --mode evaluate"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = self.figures_dir / name
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Figura guardada: {}", name)

    # ── Punto de entrada ──────────────────────────────────────────────────────

    def visualize_all(self) -> None:
        """Genera todas las figuras de RAG 3 DCI."""
        data = self._load_dci()
        agg = data["aggregate"]
        pq = data["per_query_results"]
        agent_stats = data.get("agent_stats", {})

        self._plot_three_point_comparison(agg)
        self._plot_agent_iterations(pq)
        self._plot_tool_usage(agent_stats)
        self._plot_think_vs_mrr(pq)
        self._plot_ragas_scores(agg)
        self._plot_deepeval_scores(agg)

        logger.success("Figuras RAG 3 DCI guardadas en {}", self.figures_dir)
        self._plot_final_comparison()

    # ── Figuras individuales ──────────────────────────────────────────────────

    def _plot_three_point_comparison(self, agg: dict) -> None:
        """Puntos A/B/C: MRR, Recall@5, coverage."""
        pa = agg.get("point_a", {})
        pb = agg.get("point_b", {})
        pc = agg.get("point_c", {})

        metrics_a = {"MRR": pa.get("mrr", 0), "Recall@5": pa.get("recall_at_5", 0), "NDCG@5": pa.get("ndcg_at_5", 0)}
        metrics_b = {"MRR": pb.get("mrr", 0), "Recall@5": pb.get("recall_at_5", 0), "Coverage": pb.get("coverage", 0)}
        metrics_c = {"ROUGE-L": pc.get("rouge_l", 0), "BERTScore": pc.get("bert_score_f1") or 0, "Faithfulness": pc.get("faithfulness") or 0}

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (title, metrics) in zip(
            axes,
            [("Point A — Pre-curation\n(DCI Retrieval)", metrics_a),
             ("Point B — Post-curation\n(SAIL-RAG Oracle)", metrics_b),
             ("Point C — Post-LLM\n(Answer Quality)", metrics_c)],
        ):
            keys, vals = list(metrics.keys()), list(metrics.values())
            bars = ax.barh(keys, vals, color=_COLOR_DCI, edgecolor="white")
            for bar, val in zip(bars, vals):
                ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=9)
            ax.set_xlim(0, 1.15)
            ax.set_title(title, fontweight="bold")

        fig.suptitle("RAG 3 — DCI + SAIL-RAG\nThree-Point Evaluation", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "three_point_comparison.png")

    def _plot_agent_iterations(self, pq: list[dict]) -> None:
        """Histograma de iteraciones del agente por query."""
        iters = [r.get("agent_iterations", 0) for r in pq]
        fig, ax = plt.subplots(figsize=(7, 4))
        max_iter = max(iters) if iters else 5
        bins = range(0, max_iter + 2)
        ax.hist(iters, bins=bins, color=_COLOR_DCI, edgecolor="white", alpha=0.85, rwidth=0.8)
        mean_iter = np.mean(iters) if iters else 0
        ax.axvline(mean_iter, color="#6A1B9A", linestyle="--", label=f"Media: {mean_iter:.2f}")
        ax.set_xlabel("Iteraciones del agente")
        ax.set_ylabel("Número de queries")
        ax.set_title("RAG 3 DCI — Distribución de Iteraciones por Query", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        self._save(fig, "agent_iterations_distribution.png")

    def _plot_tool_usage(self, agent_stats: dict) -> None:
        """Barras de frecuencia de uso de cada herramienta DCI."""
        tool_counts = agent_stats.get("tool_usage", {})
        if not tool_counts:
            tool_counts = {"(sin datos)": 0}

        tools = list(tool_counts.keys())
        counts = [tool_counts[t] for t in tools]
        x = np.arange(len(tools))

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(x, counts, color=_COLOR_DCI, edgecolor="white", alpha=0.9)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(cnt), ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tools, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Llamadas totales")
        ax.set_title("RAG 3 DCI — Frecuencia de Uso de Herramientas", fontweight="bold")
        plt.tight_layout()
        self._save(fig, "tool_usage.png")

    def _plot_think_vs_mrr(self, pq: list[dict]) -> None:
        """Scatter: longitud total del bloque think (palabras) vs MRR Punto A."""
        think_lens = []
        mrr_vals = []

        for r in pq:
            total_words = sum(
                len(tb.get("think", "").split())
                for tb in r.get("think_blocks", [])
            )
            mrr = r.get("point_a_pre_curation", {}).get("mrr", 0.0)
            think_lens.append(total_words)
            mrr_vals.append(mrr)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(think_lens, mrr_vals, c=_COLOR_DCI, alpha=0.6, edgecolors="white", s=40)

        # Solo trazar tendencia si hay varianza en X (evita SVD degenerado si todos son 0)
        if len(think_lens) > 2 and len(set(think_lens)) > 1:
            try:
                m, b = np.polyfit(think_lens, mrr_vals, 1)
                x_line = np.linspace(min(think_lens), max(think_lens), 100)
                ax.plot(x_line, m * x_line + b, "r--", alpha=0.7, label=f"Tendencia (slope={m:.4f})")
                ax.legend()
            except np.linalg.LinAlgError:
                pass

        no_think = sum(1 for v in think_lens if v == 0)
        if no_think == len(think_lens):
            ax.text(0.5, 0.5, "Modelo sin bloques <think>\n(Ollama suprime reasoning tokens)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10,
                    color="gray", style="italic")

        ax.set_xlabel("Palabras totales en bloques <think>")
        ax.set_ylabel("MRR (Punto A)")
        ax.set_title("RAG 3 DCI — Razonamiento vs Calidad de Retrieval", fontweight="bold")
        plt.tight_layout()
        self._save(fig, "think_vs_mrr.png")

    def _plot_ragas_scores(self, agg: dict) -> None:
        """5 métricas RAGAS (placeholders 0.0 en v1, diseñadas para llenarse)."""
        pc = agg.get("point_c", {})
        pb = agg.get("point_b", {})
        ragas_metrics = {
            "Context\nPrecision": pb.get("ragas_context_precision", 0.0),
            "Context\nRecall": pb.get("ragas_context_recall", 0.0),
            "Faithfulness": pc.get("ragas_faithfulness", 0.0),
            "Answer\nRelevancy": pc.get("ragas_answer_relevancy", 0.0),
            "Noise\nSensitivity": pc.get("ragas_noise_sensitivity", 0.0),
        }

        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(ragas_metrics))
        bars = ax.bar(x, list(ragas_metrics.values()), color=_COLOR_DCI, edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, ragas_metrics.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(list(ragas_metrics.keys()), fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("RAG 3 DCI — Métricas RAGAS", fontweight="bold")
        plt.tight_layout()
        self._save(fig, "ragas_scores.png")

    def _plot_deepeval_scores(self, agg: dict) -> None:
        """Métricas DeepEval por punto de evaluación."""
        pa = agg.get("point_a", {})
        pb = agg.get("point_b", {})
        pc = agg.get("point_c", {})
        de_metrics = {
            "Contextual\nRelevancy (A)": pa.get("deepeval_contextual_relevancy", 0.0),
            "Contextual\nRelevancy (B)": pb.get("deepeval_contextual_relevancy", 0.0),
            "Hallucination\n(C)": pc.get("deepeval_hallucination", 0.0),
            "Answer\nRelevancy (C)": pc.get("deepeval_answer_relevancy", 0.0),
        }

        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(de_metrics))
        bars = ax.bar(x, list(de_metrics.values()), color=_COLOR_DCI, edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, de_metrics.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(list(de_metrics.keys()), fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("RAG 3 DCI — Métricas DeepEval por Punto", fontweight="bold")
        plt.tight_layout()
        self._save(fig, "deepeval_scores.png")

    # ── Comparación final 4 sistemas ─────────────────────────────────────────

    def _plot_final_comparison(self) -> None:
        """
        Genera final_comparison_rag1_rag2_rag3.png con todos los sistemas disponibles.

        Orden de ablación:
          RAG 1   — BGE-M3 solo (baseline)
          RAG 2.1 — BGE-M3 + oracle (sin reranker)             [opcional]
          RAG 2.3 — BGE-M3 + reranker (sin oracle)             [opcional]
          RAG 2.2 — BGE-M3 + reranker + oracle
          RAG 3   — DCI DeepSeek-R1 + plain text + oracle Jaccard
          RAG 3.1 — DCI Granite 3.1 + Docling + oracle Contain [opcional]
          RAG 3.2 — DCI DeepSeek-R1 + Docling + oracle Contain [opcional]

        Todos los sistemas opcionales se incluyen si su JSON existe.
        Nota: Point A de RAG 3.x usa containment recall (métrica distinta a Jaccard de RAG 2.x).
        """
        rag1_path  = self.rag1_dir / "metrics_post_llm.json"
        rag21_path = self.results_base / "rag2_1" / "metrics_post_llm.json"
        rag23_path = self.results_base / "rag2_3" / "metrics_post_llm.json"
        rag2_path  = self.rag2_dir / "metrics_post_llm.json"
        rag3_path  = self.rag3_dir / "metrics_dci.json"
        rag31_path = self.results_base / "rag3_1" / "metrics_dci.json"
        rag32_path = self.results_base / "rag3_2" / "metrics_dci.json"

        missing = [p for p in [rag1_path, rag2_path, rag3_path] if not p.exists()]
        if missing:
            logger.info("Comparación final omitida — faltan: {}", [str(p) for p in missing])
            return

        def _load(p):
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

        rag1  = _load(rag1_path)
        rag2  = _load(rag2_path)
        rag3  = _load(rag3_path)
        rag21 = _load(rag21_path)
        rag23 = _load(rag23_path)
        rag31 = _load(rag31_path)
        rag32 = _load(rag32_path)

        def _get(data: dict, *keys, default: float = 0.0) -> float:
            if data is None:
                return default
            v = data.get("aggregate", {})
            for k in keys:
                v = v.get(k, {}) if isinstance(v, dict) else {}
            return float(v) if isinstance(v, (int, float)) and v is not None else default

        # ── Colores por sistema ───────────────────────────────────────────────
        _COLOR_RAG21 = "#FF9800"   # naranja
        _COLOR_RAG23 = "#EF5350"   # rojo coral
        _COLOR_RAG31 = "#26C6DA"   # cian — Granite+Docling
        _COLOR_RAG32 = "#7E57C2"   # violeta — DeepSeek+Docling

        # ── Construir listas en orden canónico de ablación ────────────────────
        systems = []  # (label, color, mrr, rec5, rouge, faith, has_faith)

        systems.append(("RAG 1\n(Baseline)",
                         _COLOR_RAG1,
                         _get(rag1, "point_a", "mrr"),
                         _get(rag1, "point_a", "recall_at_5"),
                         _get(rag1, "point_b", "rouge_l"),
                         None, False))

        if rag21:
            systems.append(("RAG 2.1\n(Oracle only)",
                             _COLOR_RAG21,
                             _get(rag21, "point_a", "mrr"),
                             _get(rag21, "point_a", "recall_at_5"),
                             _get(rag21, "point_c", "rouge_l"),
                             _get(rag21, "point_c", "faithfulness"), True))

        if rag23:
            systems.append(("RAG 2.3\n(Reranker only)",
                             _COLOR_RAG23,
                             _get(rag23, "point_a", "mrr"),
                             _get(rag23, "point_a", "recall_at_5"),
                             _get(rag23, "point_c", "rouge_l"),
                             _get(rag23, "point_c", "faithfulness"), True))

        systems.append(("RAG 2.2\n(Reranker+Oracle)",
                         _COLOR_RAG2,
                         _get(rag2, "point_a", "mrr"),
                         _get(rag2, "point_a", "recall_at_5"),
                         _get(rag2, "point_c", "rouge_l"),
                         _get(rag2, "point_c", "faithfulness"), True))

        systems.append(("RAG 3\n(DCI+Plain)",
                         _COLOR_DCI,
                         _get(rag3, "point_a", "mrr"),
                         _get(rag3, "point_a", "recall_at_5"),
                         _get(rag3, "point_c", "rouge_l"),
                         _get(rag3, "point_c", "faithfulness"), True))

        if rag31:
            systems.append(("RAG 3.1\n(Granite+Docling)",
                             _COLOR_RAG31,
                             _get(rag31, "point_a", "mrr"),
                             _get(rag31, "point_a", "recall_at_5"),
                             _get(rag31, "point_c", "rouge_l"),
                             _get(rag31, "point_c", "faithfulness"), True))

        if rag32:
            systems.append(("RAG 3.2\n(DeepSeek+Docling)",
                             _COLOR_RAG32,
                             _get(rag32, "point_a", "mrr"),
                             _get(rag32, "point_a", "recall_at_5"),
                             _get(rag32, "point_c", "rouge_l"),
                             _get(rag32, "point_c", "faithfulness"), True))

        labels_all = [s[0] for s in systems]
        colors_all = [s[1] for s in systems]
        vals_mrr   = [s[2] for s in systems]
        vals_rec   = [s[3] for s in systems]
        vals_rouge = [s[4] for s in systems]

        faith_systems = [(s[0], s[1], s[5]) for s in systems if s[6] and s[5] is not None]
        labels_faith  = [f[0] for f in faith_systems]
        colors_faith  = [f[1] for f in faith_systems]
        vals_faith    = [f[2] for f in faith_systems]

        x_all   = np.arange(len(labels_all))
        x_faith = np.arange(len(labels_faith))

        bar_w = max(0.45, min(0.65, 4.0 / len(labels_all)))
        fig_w = max(24, len(labels_all) * 3.2)
        fig, axes = plt.subplots(1, 4, figsize=(fig_w, 6))

        def _grouped_bar(ax, x, vals, colors, labels_list, title, note=""):
            bars = ax.bar(x, vals, color=colors, edgecolor="white", width=bar_w)
            ymax = max(vals) if vals else 0.1
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + ymax * 0.03,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels_list, fontsize=7, rotation=0)
            ax.set_ylim(0, ymax * 1.35 + 0.04)
            full_title = title + (f"\n{note}" if note else "")
            ax.set_title(full_title, fontweight="bold", fontsize=9.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.yaxis.grid(True, alpha=0.3, linestyle="--")
            ax.set_axisbelow(True)

        _grouped_bar(axes[0], x_all,   vals_mrr,   colors_all,   labels_all,   "MRR (Punto A)",
                     note="*RAG 3.x: containment recall")
        _grouped_bar(axes[1], x_all,   vals_rec,   colors_all,   labels_all,   "Recall@5 (Punto A)",
                     note="*RAG 3.x: containment recall")
        _grouped_bar(axes[2], x_all,   vals_rouge, colors_all,   labels_all,   "ROUGE-L (Punto C)")
        _grouped_bar(axes[3], x_faith, vals_faith, colors_faith, labels_faith,
                     "Faithfulness coseno (Punto C)", note="RAG 1 excluido (sin curación)")

        n = len(systems)
        fig.suptitle(
            f"Ablation Study Completo — {n} Sistemas RAG\n"
            "RAG 1 (baseline) → Ablation vectorial (2.x) → DCI Agéntico (3.x)",
            fontsize=12, fontweight="bold", y=1.01,
        )
        plt.tight_layout()

        out = self.results_base / "final_comparison_rag1_rag2_rag3.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.success("Comparación final ({} sistemas) guardada en {}", n, out)
