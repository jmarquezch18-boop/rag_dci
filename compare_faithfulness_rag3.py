"""
compare_faithfulness_rag3.py
============================
Runs RAGAS faithfulness on RAG3's saved responses using KEY7 (fresh).
Uses the same 50-query judge subset as RAG1/RAG2 for apples-to-apples comparison.

Output:
    results/faithfulness_comparison.json  — updated with rag3_faithfulness_ragas
    results/faithfulness_comparison_full.png — 3-way bar chart
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Force KEY7 before any Groq client initialises ────────────────────────────
from dotenv import load_dotenv
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)

_key7 = os.getenv("GROQ_API_KEY7")
if not _key7:
    print("ERROR: GROQ_API_KEY7 not found in .env")
    sys.exit(1)

for _k in ("GROQ_API_KEY5", "GROQ_API_KEY4", "GROQ_API_KEY3", "GROQ_API_KEY2", "GROQ_API_KEY6"):
    os.environ.pop(_k, None)
os.environ["GROQ_API_KEY"] = _key7

sys.path.insert(0, str(Path(__file__).parent))
from src.evaluation.judge import compute_ragas_batch
from loguru import logger

RAG1_PATH    = Path("results/rag1/metrics_post_llm.json")
RAG3_PATH    = Path("results/rag3/metrics_post_llm.json")
COMPARE_PATH = Path("results/faithfulness_comparison.json")
QA_PATH      = Path("checkpoints/rag2_qa_pairs.json")
OUT_PNG      = Path("results/faithfulness_comparison_full.png")

BATCH = 5
SLEEP = 75


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _run_ragas(records: list[dict], answer_map: dict, label: str) -> list[float | None]:
    """Run RAGAS only on records where existing_score is None."""
    scores: list[float | None] = [r["existing_score"] for r in records]
    missing_idx = [i for i, s in enumerate(scores) if s is None]

    logger.info("{}: {}/{} already scored, running RAGAS for {} missing",
                label, len(scores) - len(missing_idx), len(scores), len(missing_idx))

    if not missing_idx:
        logger.info("{}: nothing to compute", label)
        return scores

    missing = [records[i] for i in missing_idx]
    n_batches = (len(missing_idx) + BATCH - 1) // BATCH
    logger.info("{}: {} batches (~{}min)", label, n_batches, round(n_batches * SLEEP / 60, 1))

    new_scores: list[float | None] = []
    for i in range(0, len(missing_idx), BATCH):
        batch_n = i // BATCH + 1
        logger.info("{} — batch {}/{}", label, batch_n, n_batches)
        batch = compute_ragas_batch(
            queries=[r["query"] for r in missing[i:i + BATCH]],
            responses=[r["response"] for r in missing[i:i + BATCH]],
            contexts_per_query=[r["contexts"] for r in missing[i:i + BATCH]],
            references=[answer_map.get(r["query_id"], "") for r in missing[i:i + BATCH]],
        )
        new_scores.extend(b.get("faithfulness") for b in batch)
        if i + BATCH < len(missing_idx):
            logger.info("Sleeping {}s (TPM guard)...", SLEEP)
            time.sleep(SLEEP)

    for idx, val in zip(missing_idx, new_scores):
        scores[idx] = val
    return scores


def main() -> None:
    rag1_data    = load_json(RAG1_PATH)
    rag3_data    = load_json(RAG3_PATH)
    qa_pairs     = load_json(QA_PATH)
    answer_map   = {qa["query_id"]: qa["answer"] for qa in qa_pairs}
    prev_compare = load_json(COMPARE_PATH) if COMPARE_PATH.exists() else {"per_query": []}

    # Existing per-query scores from previous compare runs
    prev_by_id: dict[str, dict] = {pq["query_id"]: pq for pq in prev_compare["per_query"]}

    # ── Judge subset IDs from RAG1 ────────────────────────────────────────────
    judge_ids = {
        r["query_id"]
        for r in rag1_data["per_query_results"]
        if r.get("in_judge_subset")
    }
    logger.info("Judge subset: {} queries", len(judge_ids))

    # ── RAG3 records ──────────────────────────────────────────────────────────
    rag3_records: list[dict] = []
    for r in rag3_data["per_query_results"]:
        if r["query_id"] not in judge_ids:
            continue
        existing = prev_by_id.get(r["query_id"], {}).get("rag3_faithfulness_ragas")
        rag3_records.append({
            "query_id":       r["query_id"],
            "query":          r["query"],
            "response":       r.get("generated_response", ""),
            "contexts":       [c["text"] for c in r.get("approved_chunks", [])],
            "existing_score": existing,
        })
    logger.info("RAG3 matching queries: {}", len(rag3_records))

    rag3_scores = _run_ragas(rag3_records, answer_map, "RAG3")

    # ── Merge into existing per_query ─────────────────────────────────────────
    rag3_by_id = {rec["query_id"]: s for rec, s in zip(rag3_records, rag3_scores)}

    updated_per_query = []
    for pq in prev_compare["per_query"]:
        pq["rag3_faithfulness_ragas"] = rag3_by_id.get(pq["query_id"])
        updated_per_query.append(pq)

    # ── Averages (nulls → 0) ──────────────────────────────────────────────────
    r1_vals = [pq.get("rag1_faithfulness_ragas") or 0.0 for pq in updated_per_query]
    r2_vals = [pq.get("rag2_faithfulness_ragas") or 0.0 for pq in updated_per_query]
    r3_vals = [pq.get("rag3_faithfulness_ragas") or 0.0 for pq in updated_per_query]

    r3_null = sum(1 for s in rag3_scores if s is None)
    prev_summary = prev_compare.get("summary", {})

    summary = {
        "n_queries":      50,
        "n_rag1_scored":  prev_summary.get("n_rag1_scored", 50),
        "n_rag2_scored":  prev_summary.get("n_rag2_scored", 49),
        "n_rag3_scored":  len(rag3_scores) - r3_null,
        "metric":         "RAGAS Faithfulness (LLM judge)",
        "rag1_avg":       prev_summary.get("rag1_avg"),
        "rag2_avg":       prev_summary.get("rag2_avg"),
        "rag3_avg":       round(float(np.mean(r3_vals)), 4),
        "note":           f"All 50 queries averaged (nulls→0). RAG3 nulls: {r3_null}.",
    }

    COMPARE_PATH.write_text(
        json.dumps({"summary": summary, "per_query": updated_per_query}, indent=2),
        encoding="utf-8",
    )
    logger.success("JSON updated → {}", COMPARE_PATH)
    logger.success("RAG1 RAGAS avg: {}", summary["rag1_avg"])
    logger.success("RAG2 RAGAS avg: {}", summary["rag2_avg"])
    logger.success("RAG3 RAGAS avg (50q, nulls→0): {:.4f}", summary["rag3_avg"])

    _plot(summary, r1_vals, r2_vals, r3_vals)


def _plot(summary: dict, r1: list, r2: list, r3: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = ["RAG 1\n(Baseline)", "RAG 3\n(Curation only)", "RAG 2\n(Reranker+Curation)"]
    avgs   = [summary["rag1_avg"] or 0, summary["rag3_avg"], summary["rag2_avg"] or 0]
    stds   = [float(np.std(r1)), float(np.std(r3)), float(np.std(r2))]
    colors = ["#90CAF9", "#FF8A65", "#66BB6A"]

    ax = axes[0]
    bars = ax.bar(labels, avgs, color=colors, edgecolor="white", width=0.5,
                  yerr=stds, capsize=6, error_kw={"elinewidth": 1.5, "ecolor": "#555"})
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("RAGAS Faithfulness", fontsize=11)
    ax.set_title(f"Average RAGAS Faithfulness (n=50)\nRAG 1 vs RAG 3 vs RAG 2", fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax = axes[1]
    x = np.arange(50)
    ax.scatter(x, r1, color="#90CAF9", s=18, label="RAG 1", alpha=0.8, zorder=3)
    ax.scatter(x, r3, color="#FF8A65", s=18, label="RAG 3", alpha=0.8, zorder=3)
    ax.scatter(x, r2, color="#66BB6A", s=18, label="RAG 2", alpha=0.8, zorder=3)
    for val, col, label in zip(avgs, colors, ["RAG1", "RAG3", "RAG2"]):
        ax.axhline(val, color=col, linestyle="--", linewidth=1.1,
                   label=f"{label} avg {val:.3f}")
    ax.set_xlim(-1, 50)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Query index")
    ax.set_ylabel("RAGAS Faithfulness")
    ax.set_title("Per-query Faithfulness — RAG 1 vs RAG 3 vs RAG 2", fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle("RAGAS Faithfulness Ablation — RAG 1 vs RAG 3 vs RAG 2\n(LLM Judge, 50-query subset, nulls→0)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success("Chart saved → {}", OUT_PNG)


if __name__ == "__main__":
    main()
