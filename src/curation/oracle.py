"""
oracle.py
=========
Simula la intervención de curación humana usando answer_evidence_mapped
de PeerQA como ground truth controlado y reproducible.

Posición en el pipeline RAG 2: Paso 5 — recibe los top-N chunks del
reranker y filtra cuáles contienen evidencia real de la respuesta. Esto
reemplaza la decisión humana de "¿este fragmento es útil?" con un oráculo
determinístico basado en el ground truth del dataset.

Por qué es necesario: SAIL-RAG estudia el impacto de la curación humana
en la cadena RAG. El oráculo permite simular ese impacto de forma
controlada sin necesidad de anotadores humanos, haciendo los experimentos
reproducibles con la misma semilla.

Match de evidencia: mismo algoritmo que metrics.py (RAG 1) — primero
por chunk_id exacto (MTEB/PeerQA), luego Jaccard sobre palabras (>=umbral)
para cubrir reformulaciones parciales del texto de evidencia.
"""

from __future__ import annotations

from loguru import logger


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Similitud de Jaccard sobre conjuntos de palabras en minúsculas.

    Args:
        text_a: Primer texto.
        text_b: Segundo texto.
    Returns:
        Jaccard score entre 0.0 y 1.0.
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _chunk_matches_evidence(
    chunk: dict,
    ground_truth_ids: list[str],
    ground_truth_texts: list[str],
    jaccard_threshold: float,
) -> bool:
    """
    Determina si un chunk contiene evidencia del ground truth.

    Primero intenta match exacto por chunk_id (rápido, O(1) con set).
    Si falla, intenta Jaccard >= jaccard_threshold sobre el texto del chunk
    contra cada texto de evidencia (cubre reformulaciones).

    Args:
        chunk: Dict con chunk_id y text.
        ground_truth_ids: Lista de corpus IDs relevantes de PeerQA-qrels.
        ground_truth_texts: Lista de textos de los IDs relevantes.
        jaccard_threshold: Umbral mínimo para Jaccard match.
    Returns:
        True si el chunk contiene evidencia ground truth.
    """
    chunk_id = chunk.get("chunk_id", "")
    chunk_text = chunk.get("text", "")

    # Match exacto por ID
    if chunk_id in set(ground_truth_ids):
        return True

    # Match fuzzy por texto (para chunks de Docling que no tienen el mismo ID)
    for gt_text in ground_truth_texts:
        if _jaccard_similarity(chunk_text, gt_text) >= jaccard_threshold:
            return True

    return False


class CurationOracle:
    """
    Oráculo de curación que aprueba chunks con evidencia ground truth.

    Simula la decisión humana de "este fragmento es relevante para responder
    la pregunta" de forma determinística usando answer_evidence_mapped.
    Genera tripletas (query, chunk, label) para el bucle de aprendizaje
    activo en feedback_loop.py.
    """

    def __init__(self, config: dict):
        rag2_cfg = config.get("rag2", {})
        self.jaccard_threshold: float = rag2_cfg.get("oracle_similarity_threshold", 0.75)
        logger.info(
            "CurationOracle inicializado | jaccard_threshold={}",
            self.jaccard_threshold,
        )

    def curate(
        self,
        query: str,
        reranked_chunks: list[dict],
        ground_truth_ids: list[str],
        ground_truth_texts: list[str] | None = None,
    ) -> dict:
        """
        Aplica la curación sobre los chunks rerankeados y retorna la traza.

        Punto B del pipeline: este método marca la frontera entre la
        recuperación (automática) y la generación (LLM). Solo los chunks
        aprobados aquí llegan al LLM.

        Args:
            query: Texto de la query.
            reranked_chunks: Lista de chunks del reranker con 'text' y 'chunk_id'.
            ground_truth_ids: IDs de corpus del ground truth (PeerQA-qrels).
            ground_truth_texts: Textos del ground truth (para match fuzzy).
                                Si es None, solo se usa match por ID.
        Returns:
            Dict con:
              - approved_chunks: chunks que superaron la curación
              - rejected_chunks: chunks rechazados
              - triples: lista de tripletas para aprendizaje activo
              - coverage: fracción de ground truth cubierta por aprobados
              - oracle_decisions: lista de {chunk_id, approved, match_type}
        """
        gt_texts = ground_truth_texts or []
        gt_id_set = set(ground_truth_ids)

        approved: list[dict] = []
        rejected: list[dict] = []
        decisions: list[dict] = []

        for chunk in reranked_chunks:
            is_approved = _chunk_matches_evidence(
                chunk, ground_truth_ids, gt_texts, self.jaccard_threshold
            )

            chunk_id = chunk.get("chunk_id", "")
            if is_approved:
                approved.append(chunk)
                match_type = "exact_id" if chunk_id in gt_id_set else "jaccard"
            else:
                rejected.append(chunk)
                match_type = "none"

            decisions.append({
                "chunk_id": chunk_id,
                "approved": is_approved,
                "match_type": match_type,
                "rerank_score": chunk.get("rerank_score", chunk.get("score", 0.0)),
            })

        # Coverage: fracción de IDs ground truth capturados por los aprobados
        approved_ids = {c.get("chunk_id", "") for c in approved}
        covered = len(gt_id_set & approved_ids)
        coverage = covered / len(gt_id_set) if gt_id_set else 0.0

        # Tripletas para aprendizaje activo
        triples = self._build_triples(query, approved, rejected)

        logger.debug(
            "Oracle | aprobados={} | rechazados={} | coverage={:.3f}",
            len(approved), len(rejected), coverage,
        )

        return {
            "approved_chunks": approved,
            "rejected_chunks": rejected,
            "triples": triples,
            "coverage": coverage,
            "oracle_decisions": decisions,
        }

    def _build_triples(
        self,
        query: str,
        approved: list[dict],
        rejected: list[dict],
    ) -> list[dict]:
        """
        Construye tripletas (query, positivo, negativo) para fine-tuning contrastivo.

        Cada aprobado se empareja con el primer rechazado disponible. Si no
        hay rechazados, no se generan tripletas (no hay negativo duro).
        Si no hay aprobados, tampoco (no hay positivo).

        Args:
            query: Texto de la query.
            approved: Chunks aprobados por el oráculo.
            rejected: Chunks rechazados por el oráculo.
        Returns:
            Lista de dicts con query, positive_text, negative_text,
            positive_chunk_id, negative_chunk_id.
        """
        if not approved or not rejected:
            return []

        triples: list[dict] = []
        for pos_chunk in approved:
            # Negativo duro: el mejor rechazado (primero en lista = mayor rerank_score)
            neg_chunk = rejected[0]
            triples.append({
                "query": query,
                "positive_text": pos_chunk["text"],
                "negative_text": neg_chunk["text"],
                "positive_chunk_id": pos_chunk.get("chunk_id", ""),
                "negative_chunk_id": neg_chunk.get("chunk_id", ""),
            })

        return triples
