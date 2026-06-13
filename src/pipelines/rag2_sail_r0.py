"""
rag2_sail_r0.py
===============
Pipeline SAIL-RAG Ronda 0: reranker base sin fine-tuning.

Este pipeline es el punto de partida del estudio SAIL: mide cuánto mejora
la curación humana simulada sobre la recuperación pura (Punto A) y cuánto
impacta en la generación (Punto C), todo sin aprendizaje activo.

Flujo completo:
  [INGEST]
    PeerQASectionLoader (corpus con `title` como sección) → BGEEmbedder
    → QdrantStoreRAG2
    (DoclingParser disponible para fuentes externas sin `title` de sección)

  [EVALUATE R0]
    Phase 1 — Embedding de queries:
      BGEEmbedder.load() → embed_batch(all_queries) → BGEEmbedder.unload()
    Phase 2 — Retrieval + Reranking:
      BGEReranker.load() → por query: ANN_search → rerank → BGEReranker.unload()
    Phase 3 — Curación + Generación:
      CurationOracle → approved_chunks → GroqLLM → acumular resultados
    Phase 4 — Métricas post-LLM:
      BERTScore batch CPU → faithfulness coseno CPU → aggregate → JSON

Gestión de VRAM (RTX 3050 4GB):
  El embedder y el reranker NUNCA coexisten en GPU.
  Phase 1 completa antes de que cargue Phase 2.
  BERTScore y faithfulness coseno corren en CPU al final.

Token limits (mismo patrón que RAG 1):
  - metrics_subset_size: limite de queries evaluadas (default 200)
  - judge_subset_size: limite para métricas costosas (default 50)
  - Muestreo estratificado por longitud de query (_stratified_sample)
  - Fallback en generación: si el oráculo aprueba 0 chunks, usa top-N
    rerankeados (config: rag2.fallback_top_n)
"""

from __future__ import annotations

import gc
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger

from src.active_learning.feedback_loop import FeedbackStore
from src.curation.oracle import CurationOracle
from src.evaluation.metrics import compute_bertscore_batch
from src.evaluation.metrics_rag2 import (
    aggregate_rag2,
    compute_faithfulness_cosine,
    compute_point_a_rag2,
    compute_point_b_rag2,
    compute_point_c_rag2,
)
from src.generation.llm import GroqLLM
from src.ingestion.peerqa_section_loader import PeerQASectionLoader
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.reranker import BGEReranker
from src.retrieval.vector_store_rag2 import QdrantStoreRAG2


@staticmethod
def _stratified_sample(qa_pairs: list[dict], size: Optional[int], seed: int) -> list[dict]:
    """
    Muestra estratificada por longitud de query (tercios: corta/media/larga).

    Mismo algoritmo que RAG 1 para comparabilidad directa de subsets.

    Args:
        qa_pairs: Lista completa de QA pairs.
        size: Tamaño del subset. None = sin muestreo.
        seed: Semilla de aleatoriedad para reproducibilidad.
    Returns:
        Lista muestreada o la lista completa si size >= len(qa_pairs).
    """
    if not qa_pairs or size is None or size >= len(qa_pairs):
        return list(qa_pairs)

    sorted_pairs = sorted(qa_pairs, key=lambda q: len(q["query"]))
    n = len(sorted_pairs)
    t1, t2, t3 = sorted_pairs[:n // 3], sorted_pairs[n // 3: 2 * n // 3], sorted_pairs[2 * n // 3:]

    rng = random.Random(seed)
    n_per = size // 3
    remainder = size - n_per * 3

    s1 = rng.sample(t1, min(n_per + (1 if remainder > 0 else 0), len(t1)))
    s2 = rng.sample(t2, min(n_per + (1 if remainder > 1 else 0), len(t2)))
    s3 = rng.sample(t3, min(n_per, len(t3)))
    return s1 + s2 + s3


class RAG2PipelineR0:
    """Orquesta ingest y evaluate_r0 del pipeline SAIL-RAG Ronda 0."""

    def __init__(self, config: dict):
        self.config = config
        self.rag2_cfg = config.get("rag2", {})
        self.results_dir = Path(config["paths"]["results"]) / "rag2" / "round_0"
        self.checkpoints_dir = Path(config["paths"]["checkpoints"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # ── INGEST ────────────────────────────────────────────────────────────────

    def ingest(self) -> None:
        """
        Descarga PDFs vía Docling, chunkea con secciones e indexa en Qdrant.

        Checkpoint guardado:
          checkpoints/rag2_embeddings.npy    — array (N, 1024)
          checkpoints/rag2_chunks.json       — chunks con section metadata
          checkpoints/rag2_qa_pairs.json     — QA pairs de PeerQA
        """
        logger.info("=== RAG 2 — Modo INGEST ===")

        # 1. Cargar corpus con secciones desde mteb/PeerQA.
        #    PeerQASectionLoader usa el campo `title` del corpus como sección
        #    (e.g. 'Introduction', 'Results', 'Abstract') sin necesidad de
        #    descargar PDFs. DoclingParser queda disponible para fuentes
        #    externas donde no se disponga del título de sección.
        loader = PeerQASectionLoader(self.config)
        data = loader.load()

        qa_path = self.checkpoints_dir / "rag2_qa_pairs.json"
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(data["qa_pairs"], f, ensure_ascii=False, indent=2)
        logger.info("QA pairs guardados | n={}", len(data["qa_pairs"]))

        all_chunks = data["corpus"]

        if not all_chunks:
            logger.error("No se generaron chunks. Verifica la carga de mteb/PeerQA.")
            return

        logger.info("Total chunks con sección: {}", len(all_chunks))

        # 5. Generar embeddings (Phase 1 — embed, unload)
        embedder = BGEEmbedder(self.config)
        embedder.load()
        texts = [c["text"] for c in all_chunks]
        logger.info("Generando embeddings para {} chunks", len(texts))
        embeddings = embedder.embed_batch(texts)
        embedder.unload()
        gc.collect()

        logger.info("Embeddings: shape={} dtype={}", embeddings.shape, embeddings.dtype)

        # 6. Guardar checkpoint
        emb_path = self.checkpoints_dir / "rag2_embeddings.npy"
        np.save(emb_path, embeddings.astype(np.float32))

        chunk_meta = [
            {
                "chunk_id": c["chunk_id"],
                "paper_id": c["paper_id"],
                "section": c.get("section", "Unknown"),
                "text": c["text"],
                "source_url": c.get("source_url", ""),
            }
            for c in all_chunks
        ]
        chunks_path = self.checkpoints_dir / "rag2_chunks.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunk_meta, f, ensure_ascii=False, indent=2)

        logger.success(
            "Ingest RAG 2 completado | {} chunks | checkpoint en {}",
            len(all_chunks), self.checkpoints_dir,
        )

    # ── EVALUATE R0 ───────────────────────────────────────────────────────────

    def evaluate(self) -> None:
        """
        Ronda 0: reranker base sin fine-tuning. Genera 3 JSONs de métricas.

        Puntos de evaluación:
          A — pre-curación (post-reranker)
          B — post-curación / pre-LLM (post-oracle)
          C — post-LLM

        Estrategia de token limits:
          - Muestreo estratificado por longitud (metrics_subset_size)
          - BERTScore en CPU batch al final
          - Faithfulness coseno en CPU al final (BGE-M3 en CPU)
          - Fallback si oracle aprueba 0 chunks
        """
        logger.info("=== RAG 2 — Modo EVALUATE R0 ===")

        emb_path = self.checkpoints_dir / "rag2_embeddings.npy"
        chunks_path = self.checkpoints_dir / "rag2_chunks.json"
        qa_path = self.checkpoints_dir / "rag2_qa_pairs.json"

        for p in [emb_path, chunks_path, qa_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"No se encontró {p}.\n"
                    "Corre primero: python main.py --pipeline rag2 --mode ingest"
                )

        embeddings = np.load(emb_path)
        with open(chunks_path, encoding="utf-8") as f:
            chunks: list[dict] = json.load(f)
        with open(qa_path, encoding="utf-8") as f:
            qa_pairs: list[dict] = json.load(f)

        logger.info("Checkpoint cargado | {} chunks | {} QA pairs", len(chunks), len(qa_pairs))

        # Subset sampling
        metrics_size = self.rag2_cfg.get("metrics_subset_size")
        metrics_seed = self.rag2_cfg.get("metrics_subset_seed", 42)
        metrics_qa = _stratified_sample(qa_pairs, metrics_size, metrics_seed)
        logger.info("Subset | total={} | evaluando={}", len(qa_pairs), len(metrics_qa))

        # Reconstruir Qdrant
        store = QdrantStoreRAG2(self.config)
        store.create_collection()
        store.upsert(chunks, embeddings)

        k_candidates: int = self.rag2_cfg.get("k_candidates", 20)
        reranker_top_n: int = self.rag2_cfg.get("reranker_top_n", 7)
        section_filter: Optional[str] = self.rag2_cfg.get("section_filter")
        k_values: list[int] = self.config["evaluation"]["recall_k_values"]
        fallback_top_n: int = self.rag2_cfg.get("fallback_top_n", 3)
        min_approved: int = self.rag2_cfg.get("min_approved_for_generation", 1)

        oracle = CurationOracle(self.config)
        llm = GroqLLM(self.config)
        feedback_store = FeedbackStore(
            self.rag2_cfg.get("active_learning", {}).get(
                "feedback_store", "./checkpoints/reranker/feedback_store.json"
            )
        )

        # ── Phase 1: Embedding de todas las queries ──────────────────────────
        logger.info("Phase 1: Embedding de {} queries", len(metrics_qa))
        embedder = BGEEmbedder(self.config)
        embedder.load()
        query_vecs = embedder.embed_batch([qa["query"] for qa in metrics_qa])
        embedder.unload()
        gc.collect()

        # ── Phase 2: Reranking de todas las queries ───────────────────────────
        logger.info("Phase 2: Cargando reranker para {} queries", len(metrics_qa))
        reranker = BGEReranker(self.config)
        reranker.load()

        retrieval_results: list[tuple[list[dict], list[dict]]] = []
        for i, (qa, q_vec) in enumerate(zip(metrics_qa, query_vecs)):
            logger.info("Reranking query {}/{}", i + 1, len(metrics_qa))
            retrieved = store.search_with_filter(q_vec, k=k_candidates, section=section_filter)
            reranked = reranker.rerank(qa["query"], retrieved, top_n=reranker_top_n)
            retrieval_results.append((retrieved, reranked))

        reranker.unload()
        gc.collect()

        # ── Phase 3: Curación + Generación ────────────────────────────────────
        logger.info("Phase 3: Curación oracle + Generación LLM")
        per_query_results: list[dict] = []

        for i, (qa, (retrieved, reranked)) in enumerate(zip(metrics_qa, retrieval_results)):
            query_id = qa["query_id"]
            query = qa["query"]
            ground_truth = qa["ground_truth_evidence"]

            # Punto A
            point_a = compute_point_a_rag2(reranked, ground_truth, k_values)

            # Oráculo de curación (Punto B)
            curation = oracle.curate(
                query=query,
                reranked_chunks=reranked,
                ground_truth_ids=ground_truth,
            )
            approved = curation["approved_chunks"]
            feedback_store.add(curation["triples"])

            # Fallback: si oracle aprueba 0 chunks, usar top-N rerankeados
            context_chunks = approved if len(approved) >= min_approved else reranked[:fallback_top_n]
            used_fallback = len(approved) < min_approved
            if used_fallback:
                logger.debug(
                    "Fallback en query_id={}: oracle aprobó 0, usando top-{} rerankeados",
                    query_id, fallback_top_n,
                )

            # Punto B métricas
            point_b = compute_point_b_rag2(approved, ground_truth, reranked, k_values)

            # Generación LLM
            generated = llm.generate(query, context_chunks)

            per_query_results.append({
                "query_id": query_id,
                "query": query,
                "section_filter": section_filter,
                "retrieved_chunks_k20": retrieved,
                "reranked_chunks_top7": reranked,
                "approved_chunks": approved,
                "ground_truth_evidence": ground_truth,
                "point_a_pre_curation": point_a,
                "point_b_post_curation": point_b,
                "generated_response": generated,
                "curation_trace": {
                    "all_retrieved": [c["chunk_id"] for c in retrieved],
                    "presented_to_oracle": [c["chunk_id"] for c in reranked],
                    "approved_by_oracle": [c["chunk_id"] for c in approved],
                    "inline_references": [c.get("source_url", "") for c in approved],
                    "used_fallback": used_fallback,
                    "oracle_decisions": curation["oracle_decisions"],
                },
                "point_c_post_llm": {"rouge_l": 0.0, "bert_score_f1": None, "faithfulness": None},
            })

            logger.info(
                "Query {}/{} | A.MRR={:.3f} | B.coverage={:.3f} | aprobados={}",
                i + 1, len(metrics_qa),
                point_a.get("mrr", 0.0),
                point_b.get("coverage", 0.0),
                len(approved),
            )

        # ── Phase 4: Métricas post-LLM en CPU ────────────────────────────────
        logger.info("Phase 4: BERTScore batch CPU")
        preds = [r["generated_response"] for r in per_query_results]
        refs = [metrics_qa[i]["answer"] for i in range(len(per_query_results))]
        bs_scores = compute_bertscore_batch(preds, refs)

        # Faithfulness coseno en CPU (BGE-M3 en modo CPU)
        logger.info("Phase 4: Faithfulness coseno CPU")
        faith_threshold = self.rag2_cfg.get("faithfulness_cosine_threshold", 0.75)
        faith_embedder = BGEEmbedder(self.config)

        # Forzar CPU para no competir con GPU aunque esté disponible
        import copy
        cpu_config = copy.deepcopy(self.config)
        cpu_config["hardware"]["device"] = "cpu"
        faith_embedder_cpu = BGEEmbedder(cpu_config)
        faith_embedder_cpu.load()

        for r, bs in zip(per_query_results, bs_scores):
            faith = compute_faithfulness_cosine(
                r["generated_response"],
                r["approved_chunks"],
                faith_embedder_cpu,
                faith_threshold,
            )
            qa_match = next(
                (qa for qa in metrics_qa if qa["query_id"] == r["query_id"]), {}
            )
            r["point_c_post_llm"] = compute_point_c_rag2(
                r["generated_response"],
                qa_match.get("answer", ""),
                r["approved_chunks"],
                bs,
                faith,
            )

        faith_embedder_cpu.unload()
        gc.collect()

        # Agregar y guardar
        aggregate = aggregate_rag2(per_query_results, k_values)

        logger.success(
            "R0 completado | A.MRR={:.4f} | B.coverage={:.4f} | C.ROUGE-L={:.4f}",
            aggregate["point_a"]["mrr"],
            aggregate["point_b"]["coverage"],
            aggregate["point_c"]["rouge_l"],
        )

        self._save_results(per_query_results, aggregate, metrics_qa, qa_pairs)

    def _save_results(
        self,
        per_query_results: list[dict],
        aggregate: dict,
        metrics_qa: list[dict],
        all_qa_pairs: list[dict],
    ) -> None:
        """
        Guarda los 3 JSONs de métricas y el JSON completo de resultados.

        Estructura de archivos:
          results/rag2/round_0/metrics_pre_curation.json   — Punto A
          results/rag2/round_0/metrics_post_curation.json  — Punto B
          results/rag2/round_0/metrics_post_llm.json       — completo (todos los puntos)
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        output = {
            "condition": "rag2_sail_r0",
            "timestamp": timestamp,
            "config_snapshot": self.config,
            "subset_info": {
                "total_qa_pairs": len(all_qa_pairs),
                "metrics_subset_size": len(metrics_qa),
            },
            "per_query_results": per_query_results,
            "aggregate": aggregate,
        }

        # JSON completo
        out_path = self.results_dir / "metrics_post_llm.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        # Punto A
        pre_path = self.results_dir / "metrics_pre_curation.json"
        with open(pre_path, "w", encoding="utf-8") as f:
            json.dump(
                {"condition": "rag2_sail_r0", "timestamp": timestamp,
                 "aggregate": {"point_a": aggregate["point_a"]},
                 "per_query": [{"query_id": r["query_id"], **r["point_a_pre_curation"]}
                                for r in per_query_results]},
                f, ensure_ascii=False, indent=2,
            )

        # Punto B
        post_path = self.results_dir / "metrics_post_curation.json"
        with open(post_path, "w", encoding="utf-8") as f:
            json.dump(
                {"condition": "rag2_sail_r0", "timestamp": timestamp,
                 "aggregate": {"point_b": aggregate["point_b"]},
                 "per_query": [{"query_id": r["query_id"], **r["point_b_post_curation"]}
                                for r in per_query_results]},
                f, ensure_ascii=False, indent=2,
            )

        logger.success("Resultados R0 guardados en {}", self.results_dir)
