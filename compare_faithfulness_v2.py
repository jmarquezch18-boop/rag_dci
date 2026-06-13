"""
compare_faithfulness_v2.py
==========================
Gap-fill version: reads existing RAGAS scores from previous runs, uses
KEY4 (fresh daily budget) to compute only the missing ones, then averages
properly over all 50 queries for both RAG1 and RAG2.

RAG1 existing: 10/50 scored (yesterday's rate-limit run)
RAG2 existing: 39/50 scored (today's rate-limit run)
Missing to compute: ~40 RAG1 + ~11 RAG2 = ~51 queries via KEY4
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

# ── Force KEY2 (fresh) before any Groq client initialises ────────────────────
# KEY3 org now exhausted (~99640/100K). KEY2 is the remaining fresh account.
# KEY1/KEY4/KEY5 share the other exhausted org.
from dotenv import load_dotenv
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)

_key2 = os.getenv("GROQ_API_KEY2")
if not _key2:
    print("ERROR: GROQ_API_KEY2 not found in .env")
    sys.exit(1)

# Remove all other keys so judge._build_ragas_llm() uses KEY2 only
for _k in ("GROQ_API_KEY5", "GROQ_API_KEY4", "GROQ_API_KEY3"):
    os.environ.pop(_k, None)
os.environ["GROQ_API_KEY"] = _key2   # primary fresh key

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.judge import compute_ragas_batch
from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────────────────
RAG1_PATH    = Path("results/rag1/metrics_post_llm.json")
RAG2_PATH    = Path("results/rag2/round_n/metrics_post_llm.json")
PREV_COMPARE = Path("results/faithfulness_comparison.json")
QA_PATH      = Path("checkpoints/rag2_qa_pairs.json")
OUT_JSON     = Path("results/faithfulness_comparison.json")
OUT_PNG      = Path("results/faithfulness_comparison.png")

BATCH = 5
SLEEP = 75   # seconds between batches to stay within 12K TPM


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _run_missing(records: list[dict], answer_map: dict, label: str) -> list[float | None]:
    """Run RAGAS only on records where existing_score is None. Returns full 50-length list."""
    scores: list[float | None] = [r["existing_score"] for r in records]
    missing_idx = [i for i, s in enumerate(scores) if s is None]

    logger.info("{}: {}/{} scores already exist, running RAGAS for {} missing",
                label, len(scores) - len(missing_idx), len(scores), len(missing_idx))

    if not missing_idx:
        return scores

    missing = [records[i] for i in missing_idx]
    queries   = [r["query"]    for r in missing]
    responses = [r["response"] for r in missing]
    contexts  = [r["contexts"] for r in missing]
    refs      = [answer_map.get(r["query_id"], "") for r in missing]

    n_batches = (len(missing_idx) + BATCH - 1) // BATCH
    logger.info("{}: {} batches of {} (~{}min)", label, n_batches, BATCH,
                round(n_batches * SLEEP / 60, 1))

    new_scores: list[float | None] = []
    for i in range(0, len(missing_idx), BATCH):
        batch_n = i // BATCH + 1
        logger.info("{} — batch {}/{}", label, batch_n, n_batches)
        batch = compute_ragas_batch(
            queries=queries[i:i + BATCH],
            responses=responses[i:i + BATCH],
            contexts_per_query=contexts[i:i + BATCH],
            references=refs[i:i + BATCH],
        )
        new_scores.extend(b.get("faithfulness") for b in batch)
        if i + BATCH < len(missing_idx):
            logger.info("Sleeping {}s (TPM guard)...", SLEEP)
            time.sleep(SLEEP)

    for idx, val in zip(missing_idx, new_scores):
        scores[idx] = val

    return scores


def main() -> None:
    rag1_data = load_json(RAG1_PATH)
    rag2_data = load_json(RAG2_PATH)
    qa_pairs  = load_json(QA_PATH)
    answer_map = {qa["query_id"]: qa["answer"] for qa in qa_pairs}

    # Existing RAG2 RAGAS scores from last compare run
    prev_rag2: dict[str, float | None] = {}
    if PREV_COMPARE.exists():
        for pq in load_json(PREV_COMPARE)["per_query"]:
            prev_rag2[pq["query_id"]] = pq.get("rag2_faithfulness_ragas")

    # ── RAG1 records (50-query judge subset) ─────────────────────────────────
    rag1_records: list[dict] = []
    for r in rag1_data["per_query_results"]:
        if not r.get("in_judge_subset"):
            continue
        existing = r.get("point_c_llm_judge", {}).get("faithfulness")
        rag1_records.append({
            "query_id":      r["query_id"],
            "query":         r["query"],
            "response":      r.get("generated_response", ""),
            "contexts":      [c["text"] for c in r.get("retrieved_chunks", [])],
            "existing_score": existing,
        })
    logger.info("RAG1 judge subset: {} queries", len(rag1_records))

    # ── RAG2 records (same 50 query_ids) ─────────────────────────────────────
    judge_ids = {r["query_id"] for r in rag1_records}
    rag2_cosine: dict[str, float | None] = {}
    rag2_records: list[dict] = []
    for r in rag2_data["per_query_results"]:
        if r["query_id"] not in judge_ids:
            continue
        rag2_cosine[r["query_id"]] = r.get("point_c_post_llm", {}).get("faithfulness")
        rag2_records.append({
            "query_id":      r["query_id"],
            "query":         r["query"],
            "response":      r.get("generated_response", ""),
            "contexts":      [c["text"] for c in r.get("approved_chunks", [])],
            "existing_score": prev_rag2.get(r["query_id"]),
        })
    logger.info("RAG2 matching queries: {}", len(rag2_records))

    # ── Fill gaps: RAG1 first (higher priority), then RAG2 ───────────────────
    rag1_scores = _run_missing(rag1_records, answer_map, "RAG1")
    rag2_scores = _run_missing(rag2_records, answer_map, "RAG2")

    # ── Build per-query output ────────────────────────────────────────────────
    per_query = []
    for rec1, s1, rec2, s2 in zip(rag1_records, rag1_scores, rag2_records, rag2_scores):
        per_query.append({
            "query_id":              rec1["query_id"],
            "query":                 rec1["query"],
            "rag1_faithfulness_ragas": s1,
            "rag2_faithfulness_ragas": s2,
            "rag2_faithfulness_cosine": rag2_cosine.get(rec1["query_id"]),
        })

    # ── Average over all 50 (None → 0.0, flagged in summary) ────────────────
    r1_null = sum(1 for s in rag1_scores if s is None)
    r2_null = sum(1 for s in rag2_scores if s is None)
    r1_vals = [s if s is not None else 0.0 for s in rag1_scores]
    r2_vals = [s if s is not None else 0.0 for s in rag2_scores]

    rag1_avg = float(np.mean(r1_vals))
    rag2_avg = float(np.mean(r2_vals))

    summary = {
        "n_queries":      50,
        "n_rag1_scored":  50 - r1_null,
        "n_rag2_scored":  50 - r2_null,
        "metric":         "RAGAS Faithfulness (LLM judge)",
        "rag1_avg":       round(rag1_avg, 4),
        "rag2_avg":       round(rag2_avg, 4),
        "note": (
            f"All 50 queries averaged (nulls treated as 0). "
            f"RAG1 nulls: {r1_null}, RAG2 nulls: {r2_null}."
        ),
    }

    OUT_JSON.write_text(
        json.dumps({"summary": summary, "per_query": per_query}, indent=2),
        encoding="utf-8",
    )
    logger.success("JSON saved → {}", OUT_JSON)
    logger.success("RAG1 RAGAS faithfulness (50q, nulls→0): {:.4f}", rag1_avg)
    logger.success("RAG2 RAGAS faithfulness (50q, nulls→0): {:.4f}", rag2_avg)

    _plot(per_query, summary)


def _plot(per_query: list[dict], summary: dict) -> None:
    r1 = [q["rag1_faithfulness_ragas"] or 0.0 for q in per_query]
    r2 = [q["rag2_faithfulness_ragas"] or 0.0 for q in per_query]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: bar of averages with std
    ax = axes[0]
    labels = ["RAG 1\n(Baseline)", "RAG 2\n(SAIL + Active Learning)"]
    avgs   = [summary["rag1_avg"], summary["rag2_avg"]]
    stds   = [float(np.std(r1)), float(np.std(r2))]
    colors = ["#90CAF9", "#66BB6A"]

    bars = ax.bar(labels, avgs, color=colors, edgecolor="white", width=0.45,
                  yerr=stds, capsize=6, error_kw={"elinewidth": 1.5, "ecolor": "#555"})
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 1.2)
    ax.set_ylabel("RAGAS Faithfulness", fontsize=11)
    ax.set_title(
        f"Average Faithfulness (n=50, nulls→0)\n"
        f"RAG1 scored: {summary['n_rag1_scored']}/50 | RAG2 scored: {summary['n_rag2_scored']}/50",
        fontweight="bold",
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # Right: per-query scatter
    ax = axes[1]
    x = np.arange(50)
    ax.scatter(x, r1, color="#90CAF9", s=22, label="RAG 1", alpha=0.85, zorder=3)
    ax.scatter(x, r2, color="#66BB6A", s=22, label="RAG 2", alpha=0.85, zorder=3)
    ax.axhline(summary["rag1_avg"], color="#1565C0", linestyle="--", linewidth=1.2,
               label=f"RAG1 avg {summary['rag1_avg']:.3f}")
    ax.axhline(summary["rag2_avg"], color="#2E7D32", linestyle="--", linewidth=1.2,
               label=f"RAG2 avg {summary['rag2_avg']:.3f}")
    ax.set_xlim(-1, 50)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Query index", fontsize=10)
    ax.set_ylabel("RAGAS Faithfulness", fontsize=10)
    ax.set_title("Per-query Faithfulness (RAGAS)\nRAG 1 vs RAG 2 — all 50 queries", fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle(
        "Faithfulness Comparison — RAG 1 vs RAG 2\n"
        "(RAGAS LLM Judge, all 50-query subset, KEY4, nulls treated as 0)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success("Chart saved → {}", OUT_PNG)


if __name__ == "__main__":
    main()
