"""
metrics.py
==========
Métricas de evaluación para los dos puntos de medición del pipeline RAG 1.

Punto A (pre-LLM): MRR, Recall@k, Precision@5 — calidad de recuperación.
Punto B (post-LLM): ROUGE-L, BERTScore-F1 — calidad de generación.

Decisión: BERTScore siempre forzado a CPU para no competir con el embedder
por VRAM. El cuello de botella real es Groq (red), no BERTScore (CPU).
Si BERTScore falla, se retorna None sin detener el pipeline.

Match de evidencia: primero substring, luego Jaccard sobre palabras (>=0.7).
Esto cubre evidencias parcialmente reformuladas o con espacios distintos.
"""

from __future__ import annotations

from loguru import logger
from rouge_score import rouge_scorer


def compute_point_a(
    retrieved_chunks: list[dict],
    ground_truth_evidence: list[str],
    k_values: list[int],
) -> dict:
    """
    Calcula métricas de recuperación (Punto A — pre-LLM).

    Usa matching exacto por chunk_id contra los corpus IDs del ground truth
    (formato MTEB/BEIR). Esto es más preciso que text matching porque los
    IDs son únicos y no dependen de tokenización ni reformulación.

    Args:
        retrieved_chunks: Chunks recuperados, cada uno con campo 'chunk_id'.
        ground_truth_evidence: Lista de corpus IDs relevantes (de PeerQA-qrels).
        k_values: Lista de k para Recall@k (e.g. [1, 3, 5, 10, 20]).
    Returns:
        Dict con mrr, recall_at_{k} para cada k, precision_at_5.
    """
    empty = {"mrr": 0.0, **{f"recall_at_{k}": 0.0 for k in k_values}, "precision_at_5": 0.0}

    if not ground_truth_evidence or not retrieved_chunks:
        return empty

    gt_ids = set(ground_truth_evidence)

    # Relevancia: True si el chunk_id está en el conjunto de IDs relevantes
    relevant_flags = [c["chunk_id"] in gt_ids for c in retrieved_chunks]

    # MRR: inverso del rango del primer chunk relevante
    mrr = 0.0
    for rank, is_rel in enumerate(relevant_flags, start=1):
        if is_rel:
            mrr = 1.0 / rank
            break

    # Recall@k: fracción de IDs relevantes recuperados en top-k
    recall_at: dict[str, float] = {}
    for k in k_values:
        top_k_ids = {c["chunk_id"] for c in retrieved_chunks[:k]}
        covered = len(gt_ids & top_k_ids)
        recall_at[f"recall_at_{k}"] = covered / len(gt_ids)

    # Precision@5: fracción de top-5 que son relevantes
    top5_flags = relevant_flags[:5]
    precision_at_5 = sum(top5_flags) / len(top5_flags) if top5_flags else 0.0

    return {"mrr": mrr, **recall_at, "precision_at_5": precision_at_5}


def compute_rouge_l(generated_response: str, reference_answer: str) -> float:
    """
    Calcula ROUGE-L entre respuesta generada y referencia.

    Ligero y sin estado — se puede llamar por query inline.

    Args:
        generated_response: Respuesta generada por el LLM.
        reference_answer: Respuesta de referencia.
    Returns:
        ROUGE-L F-measure como float.
    """
    if not generated_response or not reference_answer:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(reference_answer, generated_response)["rougeL"].fmeasure)


def compute_bertscore_batch(
    predictions: list[str], references: list[str]
) -> list[float | None]:
    """
    Calcula BERTScore-F1 en batch para todas las queries de una vez.

    Se carga roberta-large UNA SOLA VEZ en CPU al final del pipeline,
    no una vez por query. Si falla, retorna lista de None.

    Args:
        predictions: Lista de respuestas generadas.
        references: Lista de respuestas de referencia.
    Returns:
        Lista de BERTScore-F1 (float o None si falló).
    """
    try:
        import evaluate
        bertscore = evaluate.load("bertscore")
        valid = [(p, r) for p, r in zip(predictions, references) if p and r]
        if not valid:
            return [None] * len(predictions)

        preds, refs = zip(*valid)
        bs = bertscore.compute(predictions=list(preds), references=list(refs),
                               lang="en", device="cuda")
        f1_vals = iter(bs["f1"])

        results = []
        for p, r in zip(predictions, references):
            if p and r:
                results.append(float(next(f1_vals)))
            else:
                results.append(None)
        return results

    except Exception as e:
        logger.error("BERTScore batch falló (no detiene el pipeline): {}", str(e))
        return [None] * len(predictions)


def compute_point_b(generated_response: str, reference_answer: str,
                    bert_score_f1: float | None = None) -> dict:
    """
    Empaqueta métricas de Punto B. ROUGE-L se calcula aquí;
    BERTScore se pasa pre-calculado desde el batch del pipeline.

    Args:
        generated_response: Respuesta generada por el LLM.
        reference_answer: Respuesta de referencia.
        bert_score_f1: BERTScore-F1 pre-calculado (None si no disponible).
    Returns:
        Dict con rouge_l (float) y bert_score_f1 (float | None).
    """
    return {
        "rouge_l": compute_rouge_l(generated_response, reference_answer),
        "bert_score_f1": bert_score_f1,
    }


def aggregate_metrics(per_query_results: list[dict], k_values: list[int]) -> dict:
    """
    Calcula promedios sobre todos los resultados por query.

    Args:
        per_query_results: Lista de dicts con el esquema JSON del pipeline.
        k_values: Lista de k para Recall@k.
    Returns:
        Dict con point_a, point_b y point_c con métricas promediadas.
    """
    n = len(per_query_results)
    if n == 0:
        return {}

    # Point A
    a_keys = ["mrr", "precision_at_5"] + [f"recall_at_{k}" for k in k_values]
    point_a_agg = {
        key: sum(r["point_a_pre_llm"].get(key, 0.0) for r in per_query_results) / n
        for key in a_keys
    }

    # Point B
    rouge_vals = [r["point_b_post_llm"]["rouge_l"] for r in per_query_results]
    bert_vals = [
        r["point_b_post_llm"]["bert_score_f1"]
        for r in per_query_results
        if r["point_b_post_llm"]["bert_score_f1"] is not None
    ]
    point_b_agg = {
        "rouge_l": sum(rouge_vals) / n,
        "bert_score_f1": sum(bert_vals) / len(bert_vals) if bert_vals else None,
    }

    # Point C — LLM judge (RAGAS); None si no se corrió o falló
    faith_vals = [
        r["point_c_llm_judge"]["faithfulness"]
        for r in per_query_results
        if r.get("point_c_llm_judge", {}).get("faithfulness") is not None
    ]
    point_c_agg = {
        "faithfulness": sum(faith_vals) / len(faith_vals) if faith_vals else None,
    }

    return {"point_a": point_a_agg, "point_b": point_b_agg, "point_c": point_c_agg}
