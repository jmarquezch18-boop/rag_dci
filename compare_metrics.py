"""
compare_metrics.py
==================
Imprime en terminal una tabla comparativa de metricas entre RAG 1, RAG 2 Round 0 y RAG 2 Round N.

Uso:
    python compare_metrics.py

No requiere modelos ni GPU -- solo lee los JSONs de resultados.
"""

import json
from pathlib import Path


def load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def g(d: dict, *keys, default: float = 0.0) -> float:
    v = d
    for k in keys:
        v = v.get(k, {}) if isinstance(v, dict) else {}
    return v if isinstance(v, (int, float)) and v is not None else default


def delta(a: float, b: float) -> str:
    if a == 0:
        return "  N/A  "
    pct = (b - a) / a * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def main() -> None:
    r1   = load("results/rag1/metrics_post_llm.json")
    r2r0 = load("results/rag2/round_0/metrics_post_llm.json")
    r2rn = load("results/rag2/round_n/metrics_post_llm.json")

    a1 = r1.get("aggregate", {})
    a0 = r2r0.get("aggregate", {})
    an = r2rn.get("aggregate", {})

    has_r1   = bool(a1)
    has_r2r0 = bool(a0)
    has_r2rn = bool(an)

    print()
    print("=" * 70)
    print("  SAIL-RAG -- Comparacion de Metricas")
    print("=" * 70)

    # ── RETRIEVAL (Punto A) ───────────────────────────────────────────────
    print()
    print("RETRIEVAL -- Punto A (pre-curacion / pre-LLM)")
    print(f"  {'Metrica':<14} {'RAG 1':>8} {'R2-R0':>8} {'R2-RN':>8}  {'D RAG1->RN':>10}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8}  {'-'*9}")

    rows_a = [
        ("MRR",         "point_a", "mrr"),
        ("Recall@1",    "point_a", "recall_at_1"),
        ("Recall@3",    "point_a", "recall_at_3"),
        ("Recall@5",    "point_a", "recall_at_5"),
        ("Recall@10",   "point_a", "recall_at_10"),
        ("Recall@20",   "point_a", "recall_at_20"),
        ("Precision@5", "point_a", "precision_at_5"),
        ("NDCG@5",      "point_a", "ndcg_at_5"),
    ]
    for label, *keys in rows_a:
        v1  = g(a1,  *keys) if has_r1   else None
        v0  = g(a0,  *keys) if has_r2r0 else None
        vn  = g(an,  *keys) if has_r2rn else None
        s1  = f"{v1:.4f}" if v1 is not None else "  N/A  "
        s0  = f"{v0:.4f}" if v0 is not None else "  N/A  "
        sn  = f"{vn:.4f}" if vn is not None else "  N/A  "
        d   = delta(v1 or 0, vn or 0) if (v1 is not None and vn is not None) else "  N/A  "
        print(f"  {label:<14} {s1:>8} {s0:>8} {sn:>8}  {d:>9}")

    # ── POST-CURATION (Punto B) ───────────────────────────────────────────
    print()
    print("POST-CURACION -- Punto B (RAG 2 only)")
    print(f"  {'Metrica':<14} {'R2-R0':>8} {'R2-RN':>8}  {'D R0->RN':>9}")
    print(f"  {'-'*14} {'-'*8} {'-'*8}  {'-'*9}")

    rows_b = [
        ("MRR",      "point_b", "mrr"),
        ("Recall@5", "point_b", "recall_at_5"),
        ("Coverage", "point_b", "coverage"),
        ("NDCG@5",   "point_b", "ndcg_at_5"),
    ]
    for label, *keys in rows_b:
        v0 = g(a0, *keys) if has_r2r0 else None
        vn = g(an, *keys) if has_r2rn else None
        s0 = f"{v0:.4f}" if v0 is not None else "  N/A  "
        sn = f"{vn:.4f}" if vn is not None else "  N/A  "
        d  = delta(v0 or 0, vn or 0) if (v0 is not None and vn is not None) else "  N/A  "
        print(f"  {label:<14} {s0:>8} {sn:>8}  {d:>9}")

    # ── GENERATION (Punto C / post-LLM) ──────────────────────────────────
    print()
    print("GENERACION -- Punto C (post-LLM)")
    print(f"  {'Metrica':<14} {'RAG 1':>8} {'R2-R0':>8} {'R2-RN':>8}  {'D RAG1->RN':>10}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8}  {'-'*9}")

    r1g = a1.get("point_b", {})   # RAG 1 stores generation in point_b
    r0g = a0.get("point_c", {})
    rng = an.get("point_c", {})

    rows_c = [
        ("ROUGE-L",      "rouge_l"),
        ("BERTScore-F1", "bert_score_f1"),
        ("Faithfulness", "faithfulness"),
    ]
    for label, key in rows_c:
        v1 = r1g.get(key) or 0.0 if has_r1   else None
        v0 = r0g.get(key) or 0.0 if has_r2r0 else None
        vn = rng.get(key) or 0.0 if has_r2rn else None
        s1 = f"{v1:.4f}" if (v1 is not None and v1 != 0.0) else "  N/A  "
        s0 = f"{v0:.4f}" if v0 is not None else "  N/A  "
        sn = f"{vn:.4f}" if vn is not None else "  N/A  "
        d  = delta(v1 or 0, vn or 0) if (v1 and vn is not None) else "  N/A  "
        print(f"  {label:<14} {s1:>8} {s0:>8} {sn:>8}  {d:>9}")

    print()
    print("Escala: 0.0 -> 1.0  (1.0 = perfecto, 100%)")
    print("D = cambio porcentual relativo respecto al valor base")
    print()


if __name__ == "__main__":
    main()
