"""
metrics_rag2.py
===============
Métricas adicionales para los 3 puntos de evaluación del pipeline RAG 2.

Extiende metrics.py (RAG 1) sin modificarlo. Las funciones de RAG 1 que
aplican directamente (compute_point_a, compute_rouge_l, compute_bertscore_batch)
se importan desde allí. Este módulo agrega:

  Punto A: NDCG@k (no disponible en RAG 1)
  Punto B: coverage (fracción del ground truth capturada por aprobados)
  Punto C: faithfulness_cosine (trazabilidad de respuesta a chunks aprobados
           usando similitud coseno > umbral — no requiere llamadas LLM)

La faithfulness coseno es la alternativa local al RAGAS judge de RAG 1:
más rápida, determinística, reproducible, sin costo de API. La trade-off
es que mide similitud semántica, no equivalencia lógica.

Aggregate RAG 2 tiene estructura diferente a RAG 1 (3 puntos A/B/C con
campos distintos) por eso se implementa aggregate_rag2 en lugar de
reutilizar aggregate_metrics.
"""

from __future__ import annotations

import math
from typing import Optional

from loguru import logger

# Importar funciones reusables de RAG 1
from src.evaluation.metrics import compute_point_a, compute_rouge_l, compute_bertscore_batch


def compute_ndcg_at_k(
    retrieved_chunks: list[dict],
    ground_truth_ids: list[str],
    k: int,
) -> float:
    """
    Calcula NDCG@k con relevancia binaria (relevante=1, no relevante=0).

    NDCG penaliza que los chunks relevantes aparezcan en posiciones bajas
    dentro del top-k, a diferencia de Recall@k que solo cuenta si están.

    Args:
        retrieved_chunks: Chunks recuperados con campo 'chunk_id'.
        ground_truth_ids: IDs de corpus relevantes.
        k: Cutoff del ranking.
    Returns:
        NDCG@k como float entre 0.0 y 1.0.
    """
    if not ground_truth_ids or not retrieved_chunks:
        return 0.0

    gt_set = set(ground_truth_ids)
    top_k = retrieved_chunks[:k]

    # DCG: suma de (relevancia / log2(rank+1))
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, c in enumerate(top_k)
        if c["chunk_id"] in gt_set
    )

    # IDCG: DCG ideal — todos los relevantes en las primeras posiciones
    n_relevant = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(n_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def compute_point_a_rag2(
    retrieved_chunks: list[dict],
    ground_truth_ids: list[str],
    k_values: list[int],
) -> dict:
    """
    Punto A RAG 2: métricas de recuperación pre-curación (post-reranker).

    Reutiliza compute_point_a de RAG 1 y agrega NDCG@5.

    Args:
        retrieved_chunks: Chunks del reranker con campo 'chunk_id'.
        ground_truth_ids: IDs de corpus relevantes.
        k_values: Valores de k para Recall@k.
    Returns:
        Dict con mrr, recall_at_{k}, precision_at_5, ndcg_at_5.
    """
    base = compute_point_a(retrieved_chunks, ground_truth_ids, k_values)
    base["ndcg_at_5"] = compute_ndcg_at_k(retrieved_chunks, ground_truth_ids, k=5)
    return base


def compute_coverage(
    approved_chunks: list[dict],
    ground_truth_ids: list[str],
) -> float:
    """
    Calcula la cobertura del ground truth por los chunks aprobados.

    Coverage es la fracción de IDs del ground truth que están presentes
    en los chunks aprobados por el oráculo. Mide cuánto de la evidencia
    real sobrevivió la curación.

    Args:
        approved_chunks: Chunks aprobados por el oráculo.
        approved_ids: IDs de corpus aprobados.
        ground_truth_ids: IDs de corpus del ground truth.
    Returns:
        Coverage como float entre 0.0 y 1.0.
    """
    if not ground_truth_ids:
        return 0.0
    if not approved_chunks:
        return 0.0
    approved_id_set = {c.get("chunk_id", "") for c in approved_chunks}
    gt_set = set(ground_truth_ids)
    covered = len(gt_set & approved_id_set)
    return covered / len(gt_set)


def compute_point_b_rag2(
    approved_chunks: list[dict],
    ground_truth_ids: list[str],
    retrieved_chunks_pre_curation: list[dict],
    k_values: list[int],
) -> dict:
    """
    Punto B RAG 2: métricas post-curación / pre-LLM.

    El Punto B mide el impacto de la curación humana (simulada): qué tan
    bueno es el conjunto de chunks DESPUÉS de filtrar con el oráculo pero
    ANTES de enviar al LLM.

    Args:
        approved_chunks: Chunks aprobados por el oráculo.
        ground_truth_ids: IDs de corpus relevantes.
        retrieved_chunks_pre_curation: Chunks pre-curación (para MRR base).
        k_values: Valores de k para Recall@k.
    Returns:
        Dict con mrr, recall_at_5, precision_at_5, coverage, ndcg_at_5.
    """
    base = compute_point_a(approved_chunks, ground_truth_ids, k_values)
    base["coverage"] = compute_coverage(approved_chunks, ground_truth_ids)
    base["ndcg_at_5"] = compute_ndcg_at_k(approved_chunks, ground_truth_ids, k=5)
    return base


def compute_faithfulness_cosine(
    generated_response: str,
    approved_chunks: list[dict],
    embedder,
    threshold: float,
) -> float:
    """
    Calcula faithfulness como fracción de oraciones de la respuesta
    trazables a chunks aprobados vía similitud coseno.

    Una oración es 'fiel' si su embedding tiene coseno >= threshold con
    el embedding de al menos un chunk aprobado. El embedder debe estar
    cargado y configurado para CPU al llamar esta función (para no
    interferir con el reranker en VRAM).

    Args:
        generated_response: Respuesta generada por el LLM.
        approved_chunks: Chunks aprobados por el oráculo.
        embedder: Instancia de BGEEmbedder con modelo cargado.
        threshold: Umbral de similitud coseno (config: faithfulness_cosine_threshold).
    Returns:
        Faithfulness como float entre 0.0 y 1.0. 0.0 si la respuesta es vacía.
    """
    import numpy as np

    if not generated_response.strip() or not approved_chunks:
        return 0.0

    # Dividir respuesta en oraciones simples
    sentences = [
        s.strip()
        for s in generated_response.replace(".", ".\n").splitlines()
        if len(s.strip()) > 10
    ]
    if not sentences:
        return 0.0

    # Embeddings de chunks aprobados
    chunk_texts = [c["text"] for c in approved_chunks]
    try:
        chunk_embeddings = embedder.embed_batch(chunk_texts)
        sentence_embeddings = embedder.embed_batch(sentences)
    except Exception as e:
        logger.warning("faithfulness_cosine: embedder falló | {}", str(e))
        return 0.0

    # Para cada oración, verificar si coseno con algún chunk >= threshold
    faithful_count = 0
    for sent_emb in sentence_embeddings:
        # coseno con todos los chunks (vectores ya normalizados → producto punto)
        similarities = chunk_embeddings @ sent_emb
        if float(np.max(similarities)) >= threshold:
            faithful_count += 1

    return faithful_count / len(sentences)


def compute_point_c_rag2(
    generated_response: str,
    reference_answer: str,
    approved_chunks: list[dict],
    bert_score_f1: float | None,
    faithfulness: float | None,
) -> dict:
    """
    Punto C RAG 2: métricas post-LLM.

    Args:
        generated_response: Respuesta generada.
        reference_answer: Respuesta de referencia del dataset.
        approved_chunks: Chunks aprobados (para faithfulness si no se pre-calculó).
        bert_score_f1: BERTScore-F1 pre-calculado en batch (None si falló).
        faithfulness: Faithfulness coseno pre-calculado (None si no disponible).
    Returns:
        Dict con rouge_l, bert_score_f1, faithfulness.
    """
    return {
        "rouge_l": compute_rouge_l(generated_response, reference_answer),
        "bert_score_f1": bert_score_f1,
        "faithfulness": faithfulness,
    }


def aggregate_rag2(per_query_results: list[dict], k_values: list[int]) -> dict:
    """
    Agrega métricas RAG 2 sobre todos los resultados por query.

    Estructura diferente a aggregate_metrics de RAG 1: 3 puntos A/B/C
    con los campos específicos de RAG 2. Maneja None en bert_score y
    faithfulness con la misma estrategia que RAG 1 (excluir del promedio).

    Args:
        per_query_results: Lista de dicts con el esquema JSON de RAG 2.
        k_values: Lista de k para Recall@k.
    Returns:
        Dict con point_a, point_b, point_c con métricas promediadas.
    """
    n = len(per_query_results)
    if n == 0:
        return {}

    def safe_mean(vals: list) -> float | None:
        valid = [v for v in vals if v is not None]
        return sum(valid) / len(valid) if valid else None

    def mean(key: str, point_key: str) -> float:
        return sum(r.get(point_key, {}).get(key, 0.0) for r in per_query_results) / n

    # ── Punto A ───────────────────────────────────────────────────────────────
    a_keys = ["mrr", "precision_at_5", "ndcg_at_5"] + [f"recall_at_{k}" for k in k_values]
    point_a = {key: mean(key, "point_a_pre_curation") for key in a_keys}

    # ── Punto B ───────────────────────────────────────────────────────────────
    b_keys = ["mrr", "recall_at_5", "precision_at_5", "coverage", "ndcg_at_5"]
    point_b = {key: mean(key, "point_b_post_curation") for key in b_keys}

    # ── Punto C ───────────────────────────────────────────────────────────────
    rouge_vals = [r.get("point_c_post_llm", {}).get("rouge_l", 0.0) for r in per_query_results]
    bs_vals = [r.get("point_c_post_llm", {}).get("bert_score_f1") for r in per_query_results]
    faith_vals = [r.get("point_c_post_llm", {}).get("faithfulness") for r in per_query_results]

    point_c = {
        "rouge_l": sum(rouge_vals) / n,
        "bert_score_f1": safe_mean(bs_vals),
        "faithfulness": safe_mean(faith_vals),
    }

    return {"point_a": point_a, "point_b": point_b, "point_c": point_c}
