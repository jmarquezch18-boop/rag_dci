"""
rag2_3_reranker_only.py
=======================
Pipeline RAG 2.3: BGE-M3 retrieval + BGE-reranker, SIN oracle de curación.

Ablation completo del stack de RAG 2:
  RAG 1   — BGE-M3 solo
  RAG 2.1 — BGE-M3 + oracle
  RAG 2.3 — BGE-M3 + reranker           ← este pipeline
  RAG 2.2 — BGE-M3 + reranker + oracle  (RAG 2 original)

RAG 2.3 aísla la contribución del reranker: cuánto mejora solo el reranking
sin curación posterior. Compara con RAG 2.1 (oracle sin reranker) y
RAG 2.2 (reranker + oracle) para descomponer el aporte de cada componente.

Reutiliza el checkpoint de ingest de RAG 2 (mismos embeddings y chunks).
"""

from __future__ import annotations

import gc
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from src.evaluation.metrics import compute_rouge_l, compute_bertscore_batch
from src.evaluation.metrics_rag2 import (
    compute_point_a_rag2,
    compute_faithfulness_cosine,
    aggregate_rag2,
)
from src.generation.llm import GroqLLM
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.reranker import BGEReranker
from src.retrieval.vector_store_rag2 import QdrantStoreRAG2


def _stratified_sample(
    qa_pairs: list[dict], size: Optional[int], seed: int
) -> list[dict]:
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


class RAG23Pipeline:
    """
    RAG 2.3: retrieval vectorial + reranker cruzado, sin oracle SAIL-RAG.

    Flujo por query:
      1. BGE-M3 embed query
      2. Qdrant ANN search → k_candidates chunks
      3. BGE-reranker cruzado → reranker_top_n chunks  [Punto A]
      4. GroqLLM.generate() con los top rerankeados directamente (sin oracle)
      5. ROUGE-L, BERTScore, Faithfulness coseno       [Punto C]

    No hay Punto B (no hay curación), así que point_b se deja vacío.
    """

    def __init__(self, config: dict):
        self.config = config
        rag2_cfg = config.get("rag2", {})

        self.results_dir = Path(config["paths"]["results"]) / "rag2_3"
        self.checkpoints_dir = Path(config["paths"]["checkpoints"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.k_candidates: int    = rag2_cfg.get("k_candidates", 20)
        self.reranker_top_n: int  = rag2_cfg.get("reranker_top_n", 7)
        self.faith_threshold: float = rag2_cfg.get("faithfulness_cosine_threshold", 0.75)
        self.section_filter: Optional[str] = rag2_cfg.get("section_filter")

        self.metrics_subset_size: Optional[int] = rag2_cfg.get("metrics_subset_size", 200)
        self.metrics_subset_seed: int = rag2_cfg.get("metrics_subset_seed", 42)
        self.k_values: list[int] = config["evaluation"]["recall_k_values"]

    def evaluate(self) -> None:
        logger.info("=== RAG 2.3 — Reranker Only (sin oracle) ===")

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

        store = QdrantStoreRAG2(self.config)
        store.create_collection()
        store.upsert(chunks, embeddings)

        llm = GroqLLM(self.config)

        # ── Phase 1: Embed queries ────────────────────────────────────────────
        logger.info("Phase 1: Embedding {} queries con BGE-M3", len(eval_qa))
        embedder = BGEEmbedder(self.config)
        embedder.load()
        query_vecs = embedder.embed_batch([qa["query"] for qa in eval_qa])
        embedder.unload()
        gc.collect()

        # ── Phase 2: Reranking ────────────────────────────────────────────────
        logger.info(
            "Phase 2: Reranking (k={} → top_n={})",
            self.k_candidates, self.reranker_top_n,
        )
        reranker = BGEReranker(self.config)
        reranker.load()

        retrieval_results: list[tuple[list[dict], list[dict]]] = []
        for i, (qa, q_vec) in enumerate(zip(eval_qa, query_vecs)):
            retrieved = store.search_with_filter(
                q_vec, k=self.k_candidates, section=self.section_filter
            )
            reranked = reranker.rerank(qa["query"], retrieved, top_n=self.reranker_top_n)
            retrieval_results.append((retrieved, reranked))
            if (i + 1) % 20 == 0:
                logger.info("Reranking {}/{}", i + 1, len(eval_qa))

        reranker.unload()
        gc.collect()

        # ── Phase 3: Generación directa (sin oracle) ──────────────────────────
        logger.info("Phase 3: Generación LLM con chunks rerankeados (sin oracle)")
        per_query_results: list[dict] = []

        for i, (qa, (retrieved, reranked)) in enumerate(zip(eval_qa, retrieval_results)):
            query_id     = qa["query_id"]
            query        = qa["query"]
            ground_truth = qa["ground_truth_evidence"]

            # Punto A: métricas sobre chunks rerankeados
            point_a = compute_point_a_rag2(reranked, ground_truth, self.k_values)

            # Generación directa con los top rerankeados
            generated = llm.generate(query, reranked)
            rouge_l   = compute_rouge_l(generated, qa.get("answer", ""))

            per_query_results.append({
                "query_id":               query_id,
                "query":                  query,
                "retrieved_top_k":        retrieved,
                "reranked_chunks":        reranked,
                "ground_truth_evidence":  ground_truth,
                "point_a_pre_curation":   point_a,
                "point_b_post_curation":  {},   # no hay oracle
                "generated_response":     generated,
                "point_c_post_llm": {
                    "rouge_l":       rouge_l,
                    "bert_score_f1": None,
                    "faithfulness":  None,
                },
            })

            logger.info(
                "Query {}/{} | A.MRR={:.3f} | ROUGE-L={:.3f}",
                i + 1, len(eval_qa),
                point_a.get("mrr", 0.0),
                rouge_l,
            )

        # ── Phase 4: BERTScore + Faithfulness en GPU ─────────────────────────
        logger.info("Phase 4: BERTScore GPU")
        preds = [r["generated_response"] for r in per_query_results]
        refs  = [eval_qa[j]["answer"]    for j in range(len(per_query_results))]
        bs_scores = compute_bertscore_batch(preds, refs)

        logger.info("Phase 4: Faithfulness coseno GPU")
        faith_embedder = BGEEmbedder(self.config)
        faith_embedder.load()

        for r, bs in zip(per_query_results, bs_scores):
            r["point_c_post_llm"]["bert_score_f1"] = bs
            # Faithfulness sobre los chunks rerankeados directamente (sin oracle)
            faith = compute_faithfulness_cosine(
                r["generated_response"],
                r["reranked_chunks"],
                faith_embedder,
                self.faith_threshold,
            )
            r["point_c_post_llm"]["faithfulness"] = faith

        faith_embedder.unload()
        gc.collect()

        # ── Agregar y guardar ─────────────────────────────────────────────────
        aggregate = aggregate_rag2(per_query_results, self.k_values)

        logger.success(
            "RAG 2.3 completado | A.MRR={:.4f} | C.ROUGE-L={:.4f} | C.Faith={:.4f}",
            aggregate["point_a"]["mrr"],
            aggregate["point_c"]["rouge_l"],
            aggregate["point_c"].get("faithfulness") or 0.0,
        )

        output = {
            "condition":    "rag2_3_reranker_only",
            "timestamp":    datetime.utcnow().isoformat() + "Z",
            "config_snapshot": self.config,
            "subset_info": {
                "total_qa_pairs":      len(qa_pairs),
                "metrics_subset_size": len(eval_qa),
                "metrics_subset_seed": self.metrics_subset_seed,
                "k_candidates":        self.k_candidates,
                "reranker_top_n":      self.reranker_top_n,
                "oracle":              False,
            },
            "per_query_results": per_query_results,
            "aggregate":         aggregate,
        }

        out_path = self.results_dir / "metrics_post_llm.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.success("Resultados guardados en {}", out_path)
