"""
rag2_sail_rn.py
===============
Pipeline SAIL-RAG Ronda N: carga checkpoint fine-tuneado y evalúa.

Este pipeline es el punto de llegada del bucle de aprendizaje activo: carga
el reranker más reciente de checkpoints/reranker/ y mide si el fine-tuning
contrastivo mejoró la cadena RAG sobre la Ronda 0.

IMPORTANTE: este pipeline NUNCA reentrena. Solo carga checkpoint y evalúa.
Si no hay checkpoint, lanza error claro. Para entrenar: --mode train.

Flujo train (modo train):
  Igual que R0 Phase 1-3, pero al final de cada query:
    CurationOracle → triples → FeedbackStore → ActiveLearningLoop.add()
    Si threshold superado → ActiveLearningLoop.run_fine_tuning(reranker)

Flujo evaluate_rn (modo evaluate_rn):
  Carga el checkpoint más reciente → igual que R0 evaluate pero con
  reranker fine-tuneado.

Los resultados se guardan en results/rag2/round_n/ para distinguirlos
de los de R0 y permitir la comparación visual.
"""

from __future__ import annotations

import copy
import gc
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger

from src.active_learning.feedback_loop import (
    ActiveLearningLoop,
    FeedbackStore,
    get_latest_checkpoint,
)
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
from src.ingestion.peerqa_loader import PeerQALoader
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.reranker import BGEReranker
from src.retrieval.vector_store_rag2 import QdrantStoreRAG2


def _stratified_sample(qa_pairs: list[dict], size: Optional[int], seed: int) -> list[dict]:
    """
    Muestra estratificada por longitud de query (tercios: corta/media/larga).

    Idéntica a la de rag2_sail_r0.py para garantizar comparabilidad entre
    rondas cuando se usa la misma semilla.

    Args:
        qa_pairs: Lista completa de QA pairs.
        size: Tamaño del subset. None = sin muestreo.
        seed: Semilla de aleatoriedad.
    Returns:
        Lista muestreada.
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


class RAG2PipelineRN:
    """Orquesta train y evaluate_rn del pipeline SAIL-RAG con aprendizaje activo."""

    def __init__(self, config: dict):
        self.config = config
        self.rag2_cfg = config.get("rag2", {})
        self.results_dir = Path(config["paths"]["results"]) / "rag2" / "round_n"
        self.checkpoints_dir = Path(config["paths"]["checkpoints"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # ── TRAIN ─────────────────────────────────────────────────────────────────

    def train(self) -> None:
        """
        Corre el bucle de aprendizaje activo: evaluación + acumulación de
        tripletas + fine-tuning del reranker cuando se supera el umbral.

        Este modo SÍ reentrena. Cada fine-tuning completado genera un
        checkpoint y una entrada en active_learning_curve.json.

        Al finalizar el dataset completo, si quedan tripletas sin usar para
        fine-tuning (total < umbral), se fuerza un último ciclo de fine-tuning
        con las tripletas acumuladas para no desperdiciarlas.
        """
        logger.info("=== RAG 2 — Modo TRAIN (aprendizaje activo) ===")
        self._verify_checkpoint_exists(require=False)

        chunks, embeddings, qa_pairs = self._load_checkpoint()
        metrics_size = self.rag2_cfg.get("metrics_subset_size")
        metrics_seed = self.rag2_cfg.get("metrics_subset_seed", 42)
        metrics_qa = _stratified_sample(qa_pairs, metrics_size, metrics_seed)

        store = QdrantStoreRAG2(self.config)
        store.create_collection()
        store.upsert(chunks, embeddings)

        al_loop = ActiveLearningLoop(self.config)
        oracle = CurationOracle(self.config)
        llm = GroqLLM(self.config)

        k_candidates: int = self.rag2_cfg.get("k_candidates", 20)
        reranker_top_n: int = self.rag2_cfg.get("reranker_top_n", 7)
        section_filter: Optional[str] = self.rag2_cfg.get("section_filter")
        fallback_top_n: int = self.rag2_cfg.get("fallback_top_n", 3)
        min_approved: int = self.rag2_cfg.get("min_approved_for_generation", 1)
        k_values: list[int] = self.config["evaluation"]["recall_k_values"]

        # Phase 1: Embedding de todas las queries
        logger.info("Phase 1: Embedding de {} queries", len(metrics_qa))
        embedder = BGEEmbedder(self.config)
        embedder.load()
        query_vecs = embedder.embed_batch([qa["query"] for qa in metrics_qa])
        embedder.unload()
        gc.collect()

        # Phase 2 + 3: Reranker con checkpoints iterativos
        reranker = BGEReranker(self.config)
        reranker.load()

        # Cargar checkpoint previo si existe
        ckpt_path, current_round = get_latest_checkpoint(
            self.rag2_cfg.get("active_learning", {}).get("checkpoint_dir", "./checkpoints/reranker")
        )
        if ckpt_path:
            reranker.load_checkpoint(ckpt_path)
            logger.info("Checkpoint cargado para continuar training | ronda={}", current_round)

        for i, (qa, q_vec) in enumerate(zip(metrics_qa, query_vecs)):
            retrieved = store.search_with_filter(q_vec, k=k_candidates, section=section_filter)
            reranked = reranker.rerank(qa["query"], retrieved, top_n=reranker_top_n)
            curation = oracle.curate(
                query=qa["query"],
                reranked_chunks=reranked,
                ground_truth_ids=qa["ground_truth_evidence"],
            )

            status = al_loop.add_oracle_decisions(curation["triples"])

            if status["trained"]:
                logger.info(
                    "Umbral superado tras query {}/{} — fine-tuning",
                    i + 1, len(metrics_qa),
                )
                round_n = al_loop.run_fine_tuning(reranker)
                logger.info("Fine-tuning ronda {} completado", round_n)

            if (i + 1) % 50 == 0:
                logger.info(
                    "Progreso: {}/{} | tripletas acumuladas: {}",
                    i + 1, len(metrics_qa), len(al_loop.store),
                )

        # Fine-tuning final si quedan tripletas sin usar
        if len(al_loop.store) > 0:
            logger.info(
                "Fine-tuning final con {} tripletas restantes", len(al_loop.store)
            )
            al_loop.run_fine_tuning(reranker)

        reranker.unload()
        gc.collect()
        logger.success("Modo TRAIN completado | ronda final={}", al_loop._current_round)

    # ── EVALUATE RN ───────────────────────────────────────────────────────────

    def evaluate(self) -> None:
        """
        Carga el checkpoint más reciente y evalúa. NUNCA reentrena.

        Si no hay checkpoint, lanza RuntimeError con instrucción clara.
        Guarda resultados en results/rag2/round_n/ incluyendo
        active_learning_curve.json para la curva de aprendizaje.
        """
        logger.info("=== RAG 2 — Modo EVALUATE RN ===")
        ckpt_path, round_n = self._verify_checkpoint_exists(require=True)

        chunks, embeddings, qa_pairs = self._load_checkpoint()
        metrics_size = self.rag2_cfg.get("metrics_subset_size")
        metrics_seed = self.rag2_cfg.get("metrics_subset_seed", 42)
        metrics_qa = _stratified_sample(qa_pairs, metrics_size, metrics_seed)

        store = QdrantStoreRAG2(self.config)
        store.create_collection()
        store.upsert(chunks, embeddings)

        k_candidates: int = self.rag2_cfg.get("k_candidates", 20)
        reranker_top_n: int = self.rag2_cfg.get("reranker_top_n", 7)
        section_filter: Optional[str] = self.rag2_cfg.get("section_filter")
        k_values: list[int] = self.config["evaluation"]["recall_k_values"]
        fallback_top_n: int = self.rag2_cfg.get("fallback_top_n", 3)
        min_approved: int = self.rag2_cfg.get("min_approved_for_generation", 1)
        faith_threshold = self.rag2_cfg.get("faithfulness_cosine_threshold", 0.75)

        oracle = CurationOracle(self.config)
        llm = GroqLLM(self.config)

        # Phase 1: Embedding
        logger.info("Phase 1: Embedding de {} queries", len(metrics_qa))
        embedder = BGEEmbedder(self.config)
        embedder.load()
        query_vecs = embedder.embed_batch([qa["query"] for qa in metrics_qa])
        embedder.unload()
        gc.collect()

        # Phase 2: Reranker con checkpoint fine-tuneado
        logger.info("Phase 2: Cargando reranker con checkpoint ronda={}", round_n)
        reranker = BGEReranker(self.config)
        reranker.load()
        reranker.load_checkpoint(ckpt_path)

        retrieval_results: list[tuple] = []
        for i, (qa, q_vec) in enumerate(zip(metrics_qa, query_vecs)):
            logger.info("Reranking query {}/{}", i + 1, len(metrics_qa))
            retrieved = store.search_with_filter(q_vec, k=k_candidates, section=section_filter)
            reranked = reranker.rerank(qa["query"], retrieved, top_n=reranker_top_n)
            retrieval_results.append((retrieved, reranked))

        reranker.unload()
        gc.collect()

        # Phase 3: Curación + Generación
        logger.info("Phase 3: Curación + Generación")
        per_query_results: list[dict] = []

        for i, (qa, (retrieved, reranked)) in enumerate(zip(metrics_qa, retrieval_results)):
            query_id = qa["query_id"]
            query = qa["query"]
            ground_truth = qa["ground_truth_evidence"]

            point_a = compute_point_a_rag2(reranked, ground_truth, k_values)
            curation = oracle.curate(
                query=query,
                reranked_chunks=reranked,
                ground_truth_ids=ground_truth,
            )
            approved = curation["approved_chunks"]
            point_b = compute_point_b_rag2(approved, ground_truth, reranked, k_values)

            context_chunks = approved if len(approved) >= min_approved else reranked[:fallback_top_n]
            used_fallback = len(approved) < min_approved

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
                "reranker_round": round_n,
            })

            logger.info(
                "Query {}/{} | A.MRR={:.3f} | B.coverage={:.3f}",
                i + 1, len(metrics_qa),
                point_a.get("mrr", 0.0),
                point_b.get("coverage", 0.0),
            )

        # Phase 4: Métricas post-LLM en CPU
        logger.info("Phase 4: BERTScore + Faithfulness CPU")
        preds = [r["generated_response"] for r in per_query_results]
        refs = [metrics_qa[i]["answer"] for i in range(len(per_query_results))]
        bs_scores = compute_bertscore_batch(preds, refs)

        cpu_config = copy.deepcopy(self.config)
        cpu_config["hardware"]["device"] = "cpu"
        faith_embedder = BGEEmbedder(cpu_config)
        faith_embedder.load()

        for r, bs in zip(per_query_results, bs_scores):
            faith = compute_faithfulness_cosine(
                r["generated_response"],
                r["approved_chunks"],
                faith_embedder,
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

        faith_embedder.unload()
        gc.collect()

        aggregate = aggregate_rag2(per_query_results, k_values)

        logger.success(
            "RN completado | ronda={} | A.MRR={:.4f} | B.coverage={:.4f} | C.ROUGE-L={:.4f}",
            round_n,
            aggregate["point_a"]["mrr"],
            aggregate["point_b"]["coverage"],
            aggregate["point_c"]["rouge_l"],
        )

        self._save_results(per_query_results, aggregate, metrics_qa, qa_pairs, round_n)

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _verify_checkpoint_exists(self, require: bool) -> tuple[str | None, int]:
        """
        Verifica la existencia del checkpoint más reciente.

        Args:
            require: Si True, lanza RuntimeError si no hay checkpoint.
        Returns:
            Tupla (checkpoint_path, round_n).
        Raises:
            RuntimeError: Si require=True y no hay checkpoint.
        """
        al_cfg = self.rag2_cfg.get("active_learning", {})
        ckpt_dir = al_cfg.get("checkpoint_dir", "./checkpoints/reranker")
        ckpt_path, round_n = get_latest_checkpoint(ckpt_dir)

        if require and ckpt_path is None:
            raise RuntimeError(
                "No hay checkpoint de fine-tuning.\n"
                "Corre primero: python main.py --pipeline rag2 --mode train"
            )
        return ckpt_path, round_n

    def _load_checkpoint(self) -> tuple[list[dict], np.ndarray, list[dict]]:
        """
        Carga embeddings, chunks y QA pairs del checkpoint de ingest.

        Returns:
            Tupla (chunks, embeddings, qa_pairs).
        Raises:
            FileNotFoundError: Si algún archivo del checkpoint no existe.
        """
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

        logger.info("Checkpoint RAG 2 cargado | {} chunks | {} QA pairs", len(chunks), len(qa_pairs))
        return chunks, embeddings, qa_pairs

    def _save_results(
        self,
        per_query_results: list[dict],
        aggregate: dict,
        metrics_qa: list[dict],
        all_qa_pairs: list[dict],
        round_n: int,
    ) -> None:
        """
        Guarda los 3 JSONs de métricas en results/rag2/round_n/.

        Incluye active_learning_curve.json si existe (para el visualizer).
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        output = {
            "condition": f"rag2_sail_r{round_n}",
            "reranker_round": round_n,
            "timestamp": timestamp,
            "config_snapshot": self.config,
            "subset_info": {
                "total_qa_pairs": len(all_qa_pairs),
                "metrics_subset_size": len(metrics_qa),
            },
            "per_query_results": per_query_results,
            "aggregate": aggregate,
        }

        out_path = self.results_dir / "metrics_post_llm.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        pre_path = self.results_dir / "metrics_pre_curation.json"
        with open(pre_path, "w", encoding="utf-8") as f:
            json.dump(
                {"condition": f"rag2_sail_r{round_n}", "timestamp": timestamp,
                 "aggregate": {"point_a": aggregate["point_a"]},
                 "per_query": [{"query_id": r["query_id"], **r["point_a_pre_curation"]}
                                for r in per_query_results]},
                f, ensure_ascii=False, indent=2,
            )

        post_path = self.results_dir / "metrics_post_curation.json"
        with open(post_path, "w", encoding="utf-8") as f:
            json.dump(
                {"condition": f"rag2_sail_r{round_n}", "timestamp": timestamp,
                 "aggregate": {"point_b": aggregate["point_b"]},
                 "per_query": [{"query_id": r["query_id"], **r["point_b_post_curation"]}
                                for r in per_query_results]},
                f, ensure_ascii=False, indent=2,
            )

        # Copiar curva de aprendizaje al directorio de resultados si existe
        al_cfg = self.rag2_cfg.get("active_learning", {})
        curve_src = Path(al_cfg.get("curve_file", "./checkpoints/reranker/active_learning_curve.json"))
        if curve_src.exists():
            import shutil
            shutil.copy(curve_src, self.results_dir / "active_learning_curve.json")

        logger.success("Resultados RN guardados en {}", self.results_dir)
