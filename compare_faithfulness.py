"""
compare_faithfulness.py
=======================
Runs RAGAS faithfulness on RAG2's saved responses for the same 50-query
judge subset used by RAG1, producing an apples-to-apples comparison.

No need to re-run the pipeline — responses are already in the result JSONs.

Usage:
    python compare_faithfulness.py

Output:
    results/faithfulness_comparison.json
    results/faithfulness_comparison.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
# .env lives one level up (juancho/.env), same as main.py expects
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()

from src.evaluation.judge import compute_ragas_batch
from loguru import logger

RAG1_PATH  = Path("results/rag1/metrics_post_llm.json")
RAG2_PATH  = Path("results/rag2/round_n/metrics_post_llm.json")
QA_PATH    = Path("checkpoints/rag2_qa_pairs.json")
OUT_JSON   = Path("results/faithfulness_comparison.json")
OUT_PNG    = Path("results/faithfulness_comparison.png")


def load_json(p: Path) -> dict | list:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    rag1 = load_json(RAG1_PATH)
    rag2 = load_json(RAG2_PATH)
    qa_pairs = load_json(QA_PATH)
    answer_map = {qa["query_id"]: qa["answer"] for qa in qa_pairs}

    # ── RAG1: collect 50-query judge subset ──────────────────────────────────
    rag1_judge = {
        r["query_id"]: r["point_c_llm_judge"].get("faithfulness")
        for r in rag1["per_query_results"]
        if r.get("in_judge_subset")
    }
    judge_ids = set(rag1_judge.keys())
    logger.info("RAG1 judge subset: {} queries", len(judge_ids))

    # ── RAG2: find matching queries ───────────────────────────────────────────
    rag2_subset = [
        r for r in rag2["per_query_results"]
        if r["query_id"] in judge_ids
    ]
    logger.info("RAG2 matching queries found: {}", len(rag2_subset))

    if not rag2_subset:
        logger.error("No matching queries found — check that both JSONs share query_ids.")
        sys.exit(1)

    # ── Run RAGAS on RAG2 subset in small batches to avoid TPM rate limit ────
    # Groq free tier: 12K TPM. RAGAS makes ~5 LLM calls per query (~1K tokens
    # each) → ~5 queries per minute safely. Batch of 5, sleep 75s between.
    import time

    BATCH = 5
    SLEEP = 75

    queries_list   = [r["query"] for r in rag2_subset]
    responses_list = [r["generated_response"] for r in rag2_subset]
    contexts_list  = [[c["text"] for c in r["approved_chunks"]] for r in rag2_subset]
    refs_list      = [answer_map.get(r["query_id"], "") for r in rag2_subset]

    ragas_scores: list[dict] = []
    n_batches = (len(rag2_subset) + BATCH - 1) // BATCH
    logger.info(
        "Running RAGAS in {} batches of {} ({}s sleep between) — ~{}min total",
        n_batches, BATCH, SLEEP, round(n_batches * SLEEP / 60, 1),
    )
    for i in range(0, len(rag2_subset), BATCH):
        batch_n = i // BATCH + 1
        logger.info("Batch {}/{}", batch_n, n_batches)
        scores = compute_ragas_batch(
            queries=queries_list[i:i + BATCH],
            responses=responses_list[i:i + BATCH],
            contexts_per_query=contexts_list[i:i + BATCH],
            references=refs_list[i:i + BATCH],
        )
        ragas_scores.extend(scores)
        if i + BATCH < len(rag2_subset):
            logger.info("Sleeping {}s to respect TPM limit...", SLEEP)
            time.sleep(SLEEP)

    # ── Build per-query comparison ────────────────────────────────────────────
    per_query = []
    for r, score in zip(rag2_subset, ragas_scores):
        qid = r["query_id"]
        per_query.append({
            "query_id": qid,
            "query": r["query"],
            "rag1_faithfulness_ragas": rag1_judge.get(qid),
            "rag2_faithfulness_ragas": score.get("faithfulness"),
            "rag2_faithfulness_cosine": r["point_c_post_llm"].get("faithfulness"),
        })

    valid = [
        q for q in per_query
        if q["rag1_faithfulness_ragas"] is not None
        and q["rag2_faithfulness_ragas"] is not None
    ]

    rag1_avg = np.mean([q["rag1_faithfulness_ragas"] for q in valid]) if valid else None
    rag2_avg = np.mean([q["rag2_faithfulness_ragas"] for q in valid]) if valid else None

    summary = {
        "n_queries": len(valid),
        "metric": "RAGAS Faithfulness (LLM judge)",
        "rag1_avg": round(float(rag1_avg), 4) if rag1_avg is not None else None,
        "rag2_avg": round(float(rag2_avg), 4) if rag2_avg is not None else None,
        "note": "Both RAGs evaluated on the same 50-query judge subset using RAGAS.",
    }

    OUT_JSON.write_text(
        json.dumps({"summary": summary, "per_query": per_query}, indent=2),
        encoding="utf-8",
    )
    logger.success("JSON saved → {}", OUT_JSON)
    logger.success("RAG1 avg RAGAS faithfulness: {:.4f}", rag1_avg or 0)
    logger.success("RAG2 avg RAGAS faithfulness: {:.4f}", rag2_avg or 0)

    # ── Plot ──────────────────────────────────────────────────────────────────
    if valid:
        _plot(valid, summary)
    else:
        logger.error("No valid pairs to plot — check RAGAS output above.")


def _plot(valid: list[dict], summary: dict) -> None:
    rag1_vals = [q["rag1_faithfulness_ragas"] for q in valid]
    rag2_vals = [q["rag2_faithfulness_ragas"] for q in valid]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: grouped bar of averages ────────────────────────────────────────
    ax = axes[0]
    labels = ["RAG 1\n(Baseline)", "RAG 2\n(SAIL + Active Learning)"]
    avgs   = [summary["rag1_avg"], summary["rag2_avg"]]
    stds   = [float(np.std(rag1_vals)), float(np.std(rag2_vals))]
    colors = ["#90CAF9", "#66BB6A"]

    bars = ax.bar(labels, avgs, color=colors, edgecolor="white", width=0.45,
                  yerr=stds, capsize=6, error_kw={"elinewidth": 1.5, "ecolor": "#555"})
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 1.2)
    ax.set_ylabel("RAGAS Faithfulness", fontsize=11)
    ax.set_title(f"Average Faithfulness (n={summary['n_queries']})\nSame metric, same queries",
                 fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # ── Right: per-query scatter ──────────────────────────────────────────────
    ax = axes[1]
    x = np.arange(len(valid))
    ax.scatter(x, rag1_vals, color="#90CAF9", s=22, label="RAG 1", alpha=0.85, zorder=3)
    ax.scatter(x, rag2_vals, color="#66BB6A", s=22, label="RAG 2", alpha=0.85, zorder=3)
    ax.axhline(summary["rag1_avg"], color="#1565C0", linestyle="--", linewidth=1.2,
               label=f"RAG1 avg {summary['rag1_avg']:.3f}")
    ax.axhline(summary["rag2_avg"], color="#2E7D32", linestyle="--", linewidth=1.2,
               label=f"RAG2 avg {summary['rag2_avg']:.3f}")
    ax.set_xlim(-1, len(valid))
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Query index", fontsize=10)
    ax.set_ylabel("RAGAS Faithfulness", fontsize=10)
    ax.set_title("Per-query Faithfulness (RAGAS)\nRAG 1 vs RAG 2", fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle("Faithfulness Comparison — RAG 1 vs RAG 2\n(RAGAS LLM Judge, same 50-query subset)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success("Chart saved → {}", OUT_PNG)


if __name__ == "__main__":
    main()
