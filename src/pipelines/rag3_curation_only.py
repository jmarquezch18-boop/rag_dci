"""
rag3_curation_only.py
=====================
Pipeline RAG 3: Curación sin reranker (ablación).

Aísla la contribución de la curación del oráculo eliminando el paso de
reranking. Usa los mismos chunks de sección de RAG 2 (mismo checkpoint)
pero pasa los top-N candidatos por score vectorial directamente al oráculo.

Cadena:
  BGE-M3 retrieval → top-N por score → Oracle curation → Groq LLM

Ablación completa:
  RAG 1: chunks básicos  → LLM                         (sin curación, sin reranker)
  RAG 3: chunks sección  → curación → LLM              (curación, sin reranker)  ← este
  RAG 2: chunks sección  → reranker → curación → LLM   (curación + reranker)
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
from loguru import logger

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
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.vector_store_rag2 import QdrantStoreRAG2


def _stratified_sample(qa_pairs: list[dict], size: Optional[int], seed: int) -> list[dict]:
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


class RAG3Pipeline:
    """RAG 3: curación directa sin reranker. Ablación entre RAG 1 y RAG 2."""

    def __init__(self, config: dict):
        self.config = config
        self.rag3_cfg = config.get("rag3", {})
        self.results_dir = Path(config["paths"]["results"]) / "rag3"
        self.checkpoints_dir = Path(config["paths"]["checkpoints"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self) -> None:
        """
        Evalúa RAG 3: retrieval vectorial → curación → LLM.
        Sin reranker. Usa el mismo checkpoint de sección que RAG 2.
        """
        logger.info("=== RAG 3 — Modo EVALUATE (curación sin reranker) ===")

        chunks, embeddings, qa_pairs = self._load_checkpoint()

        metrics_size = self.rag3_cfg.get("metrics_subset_size")
        metrics_seed = self.rag3_cfg.get("metrics_subset_seed", 42)
        metrics_qa = _stratified_sample(qa_pairs, metrics_size, metrics_seed)

        store = QdrantStoreRAG2(self.config)
        store.create_collection()
        store.upsert(chunks, embeddings)

        k_candidates: int = self.rag3_cfg.get("k_candidates", 20)
        oracle_top_n: int = self.rag3_cfg.get("oracle_top_n", 7)
        section_filter: Optional[str] = self.rag3_cfg.get("section_filter")
        fallback_top_n: int = self.rag3_cfg.get("fallback_top_n", 3)
        min_approved: int = self.rag3_cfg.get("min_approved_for_generation", 1)
        faith_threshold: float = self.rag3_cfg.get("faithfulness_cosine_threshold", 0.75)
        k_values: list[int] = self.config["evaluation"]["recall_k_values"]

        oracle = CurationOracle(self.config)
        llm = GroqLLM(self.config)

        # Phase 1: Embedding de todas las queries
        logger.info("Phase 1: Embedding de {} queries", len(metrics_qa))
        embedder = BGEEmbedder(self.config)
        embedder.load()
        query_vecs = embedder.embed_batch([qa["query"] for qa in metrics_qa])
        embedder.unload()
        gc.collect()

        # Phase 2: Retrieval vectorial — top-oracle_top_n por score (sin reranker)
        logger.info(
            "Phase 2: Retrieval vectorial top-{} (sin reranker) para {} queries",
            oracle_top_n, len(metrics_qa),
        )
        retrieval_results: list[tuple] = []
        for i, (qa, q_vec) in enumerate(zip(metrics_qa, query_vecs)):
            candidates = store.search_with_filter(q_vec, k=k_candidates, section=section_filter)
            top_n = candidates[:oracle_top_n]
            retrieval_results.append((candidates, top_n))

        # Phase 3: Curación + Generación
        logger.info("Phase 3: Curación + Generación")
        per_query_results: list[dict] = []

        for i, (qa, (candidates, top_n)) in enumerate(zip(metrics_qa, retrieval_results)):
            query_id = qa["query_id"]
            query = qa["query"]
            ground_truth = qa["ground_truth_evidence"]

            # Point A: métricas sobre el top-N vectorial (equivalente al top-7 rerankeado en RAG 2)
            point_a = compute_point_a_rag2(top_n, ground_truth, k_values)

            curation = oracle.curate(
                query=query,
                reranked_chunks=top_n,
                ground_truth_ids=ground_truth,
            )
            approved = curation["approved_chunks"]
            point_b = compute_point_b_rag2(approved, ground_truth, top_n, k_values)

            context_chunks = approved if len(approved) >= min_approved else top_n[:fallback_top_n]
            used_fallback = len(approved) < min_approved

            generated = llm.generate(query, context_chunks)

            per_query_results.append({
                "query_id": query_id,
                "query": query,
                "section_filter": section_filter,
                "retrieved_chunks_k20": candidates,
                "top_n_chunks": top_n,
                "approved_chunks": approved,
                "ground_truth_evidence": ground_truth,
                "point_a_pre_curation": point_a,
                "point_b_post_curation": point_b,
                "generated_response": generated,
                "curation_trace": {
                    "all_retrieved": [c["chunk_id"] for c in candidates],
                    "presented_to_oracle": [c["chunk_id"] for c in top_n],
                    "approved_by_oracle": [c["chunk_id"] for c in approved],
                    "used_fallback": used_fallback,
                    "oracle_decisions": curation["oracle_decisions"],
                },
                "point_c_post_llm": {"rouge_l": 0.0, "bert_score_f1": None, "faithfulness": None},
            })

            logger.info(
                "Query {}/{} | A.MRR={:.3f} | B.coverage={:.3f}",
                i + 1, len(metrics_qa),
                point_a.get("mrr", 0.0),
                point_b.get("coverage", 0.0),
            )

        # Phase 4: BERTScore + Faithfulness en CPU
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
            qa_match = next((qa for qa in metrics_qa if qa["query_id"] == r["query_id"]), {})
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
            "RAG 3 completado | A.MRR={:.4f} | B.coverage={:.4f} | C.ROUGE-L={:.4f}",
            aggregate["point_a"]["mrr"],
            aggregate["point_b"]["coverage"],
            aggregate["point_c"]["rouge_l"],
        )

        self._save_results(per_query_results, aggregate, metrics_qa, qa_pairs)

    def _load_checkpoint(self) -> tuple[list[dict], np.ndarray, list[dict]]:
        """Carga el checkpoint de RAG 2 (mismos chunks de sección)."""
        emb_path = self.checkpoints_dir / "rag2_embeddings.npy"
        chunks_path = self.checkpoints_dir / "rag2_chunks.json"
        qa_path = self.checkpoints_dir / "rag2_qa_pairs.json"

        for p in [emb_path, chunks_path, qa_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"No se encontró {p}.\n"
                    "RAG 3 usa el checkpoint de RAG 2. "
                    "Corre primero: python main.py --pipeline rag2 --mode ingest"
                )

        embeddings = np.load(emb_path)
        with open(chunks_path, encoding="utf-8") as f:
            chunks: list[dict] = json.load(f)
        with open(qa_path, encoding="utf-8") as f:
            qa_pairs: list[dict] = json.load(f)

        logger.info(
            "Checkpoint RAG 2 cargado para RAG 3 | {} chunks | {} QA pairs",
            len(chunks), len(qa_pairs),
        )
        return chunks, embeddings, qa_pairs

    def _save_results(
        self,
        per_query_results: list[dict],
        aggregate: dict,
        metrics_qa: list[dict],
        all_qa_pairs: list[dict],
    ) -> None:
        timestamp = datetime.utcnow().isoformat() + "Z"
        output = {
            "condition": "rag3_curation_only",
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

        logger.success("Resultados RAG 3 guardados en {}", self.results_dir)
