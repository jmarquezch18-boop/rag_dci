"""
rag2_1_oracle_only.py
=====================
Pipeline RAG 2.1: BGE-M3 retrieval + SAIL-RAG oracle, SIN reranker.

Posición en el experimento comparativo:
  RAG 1  → BGE-M3 retrieval solo (sin oracle, sin reranker)
  RAG 2.1 → BGE-M3 retrieval + oracle SAIL-RAG  ← este pipeline
  RAG 2.2 → BGE-M3 retrieval + reranker + oracle SAIL-RAG  (RAG 2 original)
  RAG 3  → DCI agéntico + oracle SAIL-RAG

RAG 2.1 aísla la contribución del oracle: cuánto mejora la curación sola
sin el reranker cruzado. Compara directamente con RAG 1 (sin oracle) y
RAG 2.2 (con reranker + oracle).

Reutiliza el checkpoint de ingest de RAG 2 (mismos embeddings y chunks).
No requiere ingest propio — correr después de: python main.py --pipeline rag2 --mode ingest
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
from src.evaluation.metrics import compute_rouge_l, compute_bertscore_batch
from src.evaluation.metrics_rag2 import (
    compute_point_a_rag2,
    compute_point_b_rag2,
    compute_faithfulness_cosine,
    aggregate_rag2,
)
from src.generation.llm import GroqLLM
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.vector_store import QdrantStore


def _stratified_sample(
    qa_pairs: list[dict], size: Optional[int], seed: int
) -> list[dict]:
    """Muestra estratificada por longitud de query (tercios: corta/media/larga)."""
    if not qa_pairs or size is None or size >= len(qa_pairs):
        return list(qa_pairs)
    sorted_pairs = sorted(qa_pairs, key=lambda q: len(q["query"]))
    n = len(sorted_pairs)
    t1 = sorted_pairs[: n // 3]
    t2 = sorted_pairs[n // 3 : 2 * n // 3]
    t3 = sorted_pairs[2 * n // 3 :]
    rng = random.Random(seed)
    n_per = size // 3
    rem = size - n_per * 3
    s1 = rng.sample(t1, min(n_per + (1 if rem > 0 else 0), len(t1)))
    s2 = rng.sample(t2, min(n_per + (1 if rem > 1 else 0), len(t2)))
    s3 = rng.sample(t3, min(n_per, len(t3)))
    return s1 + s2 + s3


class RAG21Pipeline:
    """
    RAG 2.1: retrieval vectorial directo + oracle SAIL-RAG, sin reranker.

    Flujo por query:
      1. BGE-M3 embed query
      2. Qdrant ANN search → top_k_direct chunks  ← SIN reranker
      3. CurationOracle.curate() → approved_chunks  [Punto A = pre-curación]
      4. GroqLLM.generate()                          [Punto B = post-curación]
      5. ROUGE-L, BERTScore, Faithfulness coseno     [Punto C = post-LLM]

    Reutiliza el checkpoint de RAG 2 (embeddings + chunks + qa_pairs).
    """

    def __init__(self, config: dict):
        self.config = config
        rag2_cfg = config.get("rag2", {})

        self.results_dir = Path(config["paths"]["results"]) / "rag2_1"
        self.checkpoints_dir = Path(config["paths"]["checkpoints"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.k_candidates: int = rag2_cfg.get("k_candidates", 20)
        # Sin reranker: pasamos los top-k directamente al oracle
        # Usamos el mismo número que el reranker_top_n de RAG 2 para comparación justa
        self.top_k_direct: int = rag2_cfg.get("reranker_top_n", 7)
        self.min_approved: int = rag2_cfg.get("min_approved_for_generation", 1)
        self.fallback_top_n: int = rag2_cfg.get("fallback_top_n", 3)
        self.faith_threshold: float = rag2_cfg.get("faithfulness_cosine_threshold", 0.75)

        self.metrics_subset_size: Optional[int] = rag2_cfg.get("metrics_subset_size", 200)
        self.metrics_subset_seed: int = rag2_cfg.get("metrics_subset_seed", 42)
        self.k_values: list[int] = config["evaluation"]["recall_k_values"]

    def evaluate(self) -> None:
        """
        Corre el pipeline RAG 2.1 completo sobre el subset de evaluación.

        Carga el checkpoint de RAG 2 (mismos embeddings y chunks) para
        garantizar que la comparación con RAG 2.2 sea sobre los mismos datos.
        """
        logger.info("=== RAG 2.1 — Oracle Only (sin reranker) ===")

        # ── Cargar checkpoint de RAG 2 ────────────────────────────────────────
        emb_path    = self.checkpoints_dir / "rag2_embeddings.npy"
        chunks_path = self.checkpoints_dir / "rag2_chunks.json"
        qa_path     = self.checkpoints_dir / "rag2_qa_pairs.json"

        for p in [emb_path, chunks_path, qa_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"No se encontró {p}.\n"
                    "Corre primero: python main.py --pipeline rag2 --mode ingest"
                )

        embeddings: np.ndarray = np.load(emb_path)
        with open(chunks_path, encoding="utf-8") as f:
            chunks: list[dict] = json.load(f)
        with open(qa_path, encoding="utf-8") as f:
            qa_pairs: list[dict] = json.load(f)

        logger.info(
            "Checkpoint RAG 2 cargado | {} chunks | {} QA pairs",
            len(chunks), len(qa_pairs),
        )

        eval_qa = _stratified_sample(qa_pairs, self.metrics_subset_size, self.metrics_subset_seed)
        logger.info("Subset | total={} | evaluando={}", len(qa_pairs), len(eval_qa))

        # Reconstruir Qdrant en memoria
        store = QdrantStore(self.config)
        store.create_collection()
        store.upsert(chunks, embeddings)

        oracle = CurationOracle(self.config)
        llm    = GroqLLM(self.config)

        # ── Phase 1: Embed todas las queries de una vez ───────────────────────
        logger.info("Phase 1: Embedding {} queries con BGE-M3", len(eval_qa))
        embedder = BGEEmbedder(self.config)
        embedder.load()
        query_vecs = embedder.embed_batch([qa["query"] for qa in eval_qa])
        embedder.unload()
        gc.collect()

        # ── Phase 2: Retrieve + Oracle + Generación ───────────────────────────
        logger.info(
            "Phase 2: Retrieval ANN (k={}) → Oracle → Generación (sin reranker)",
            self.k_candidates,
        )
        per_query_results: list[dict] = []

        for i, (qa, q_vec) in enumerate(zip(eval_qa, query_vecs)):
            query_id     = qa["query_id"]
            query        = qa["query"]
            ground_truth = qa["ground_truth_evidence"]

            # ANN retrieval — sin reranker, truncamos a top_k_direct
            retrieved_k = store.search(q_vec, k=self.k_candidates)
            top_chunks  = retrieved_k[: self.top_k_direct]

            # Punto A: métricas sobre los chunks recuperados (pre-oracle)
            point_a = compute_point_a_rag2(top_chunks, ground_truth, self.k_values)

            # Oracle SAIL-RAG
            curation = oracle.curate(
                query=query,
                reranked_chunks=top_chunks,
                ground_truth_ids=ground_truth,
            )
            approved     = curation["approved_chunks"]
            used_fallback = len(approved) < self.min_approved
            context_chunks = approved if not used_fallback else top_chunks[: self.fallback_top_n]

            # Punto B: métricas post-oracle
            point_b = compute_point_b_rag2(approved, ground_truth, top_chunks, self.k_values)

            # Generación
            generated = llm.generate(query, context_chunks)
            rouge_l   = compute_rouge_l(generated, qa.get("answer", ""))

            per_query_results.append({
                "query_id":            query_id,
                "query":               query,
                "retrieved_top_k":     top_chunks,
                "approved_chunks":     approved,
                "ground_truth_evidence": ground_truth,
                "point_a_pre_curation":  point_a,
                "point_b_post_curation": point_b,
                "generated_response":  generated,
                "used_fallback":       used_fallback,
                "curation_trace": {
                    "presented_to_oracle": [c["chunk_id"] for c in top_chunks],
                    "approved_by_oracle":  [c["chunk_id"] for c in approved],
                    "oracle_decisions":    curation["oracle_decisions"],
                },
                "point_c_post_llm": {
                    "rouge_l":      rouge_l,
                    "bert_score_f1": None,
                    "faithfulness":  None,
                },
            })

            logger.info(
                "Query {}/{} | A.MRR={:.3f} | B.cov={:.3f} | aprobados={} | fallback={}",
                i + 1, len(eval_qa),
                point_a.get("mrr", 0.0),
                point_b.get("coverage", 0.0),
                len(approved),
                used_fallback,
            )

        # ── Phase 3: BERTScore + Faithfulness en GPU ─────────────────────────
        logger.info("Phase 3: BERTScore GPU para {} queries", len(per_query_results))
        preds = [r["generated_response"]  for r in per_query_results]
        refs  = [eval_qa[j]["answer"]     for j in range(len(per_query_results))]
        bs_scores = compute_bertscore_batch(preds, refs)

        logger.info("Phase 3: Faithfulness coseno GPU")
        faith_embedder = BGEEmbedder(self.config)
        faith_embedder.load()

        for r, bs in zip(per_query_results, bs_scores):
            r["point_c_post_llm"]["bert_score_f1"] = bs
            faith = compute_faithfulness_cosine(
                r["generated_response"],
                r["approved_chunks"],
                faith_embedder,
                self.faith_threshold,
            )
            r["point_c_post_llm"]["faithfulness"] = faith

        faith_embedder.unload()
        gc.collect()

        # ── Agregar y guardar ─────────────────────────────────────────────────
        aggregate = aggregate_rag2(per_query_results, self.k_values)

        logger.success(
            "RAG 2.1 completado | A.MRR={:.4f} | B.cov={:.4f} | C.ROUGE-L={:.4f} | C.Faith={:.4f}",
            aggregate["point_a"]["mrr"],
            aggregate["point_b"]["coverage"],
            aggregate["point_c"]["rouge_l"],
            aggregate["point_c"].get("faithfulness") or 0.0,
        )

        output = {
            "condition":    "rag2_1_oracle_only",
            "timestamp":    datetime.utcnow().isoformat() + "Z",
            "config_snapshot": self.config,
            "subset_info": {
                "total_qa_pairs":      len(qa_pairs),
                "metrics_subset_size": len(eval_qa),
                "metrics_subset_seed": self.metrics_subset_seed,
                "k_candidates":        self.k_candidates,
                "top_k_direct":        self.top_k_direct,
                "reranker":            False,
            },
            "per_query_results": per_query_results,
            "aggregate":         aggregate,
        }

        out_path = self.results_dir / "metrics_post_llm.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.success("Resultados guardados en {}", out_path)
