"""
rag1_baseline.py
================
Pipeline completo RAG 1 Baseline: coordina todos los módulos del proyecto.

Flujo ingest:
  PeerQALoader → Chunker → BGEEmbedder.load() → embed_batch()
  → BGEEmbedder.unload() → QdrantStore.upsert() → checkpoint numpy+JSON

Flujo evaluate:
  checkpoint → QdrantStore (reconstruida) → BGEEmbedder.load()
  → por query: embed_query → search(k=20) → top-5 → [Punto A]
  → GroqLLM.generate() → [Punto B] → acumula → BGEEmbedder.unload()
  → JSON completo guardado de una vez al final

Decisiones clave:
  - Embedder nunca coexiste con otro modelo en VRAM (patrón para RAG 2/3)
  - Qdrant in-memory se reconstruye desde checkpoint en cada evaluate
  - JSON guardado solo al terminar (nunca mid-pipeline)
  - BERTScore en CPU dentro de compute_point_b (no afecta este módulo)
"""

from __future__ import annotations

import gc
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from src.evaluation.judge import compute_ragas_batch
from src.evaluation.metrics import (
    aggregate_metrics, compute_point_a, compute_point_b,
    compute_rouge_l, compute_bertscore_batch,
)
from src.generation.llm import GroqLLM
from src.ingestion.chunker import Chunker
from src.ingestion.peerqa_loader import PeerQALoader
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.vector_store import QdrantStore


class RAG1Pipeline:
    """Orquesta ingest y evaluate del pipeline RAG 1 Baseline."""

    def __init__(self, config: dict):
        self.config = config
        self.results_dir = Path(config["paths"]["results"]) / "rag1"
        self.checkpoints_dir = Path(config["paths"]["checkpoints"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _log_gpu_state(self) -> None:
        """Loggea estado de GPU al inicio del pipeline para detectar fugas de VRAM."""
        if torch.cuda.is_available():
            logger.info("GPU: {}", torch.cuda.get_device_name(0))
            logger.info(
                "VRAM total: {:.1f} GB",
                torch.cuda.get_device_properties(0).total_memory / 1e9,
            )
            logger.info(
                "VRAM disponible: {:.1f} GB",
                torch.cuda.memory_reserved(0) / 1e9,
            )
        else:
            logger.warning("CUDA no disponible, usando CPU")

    # ── SUBSET SAMPLING ───────────────────────────────────────────────────────

    @staticmethod
    def _stratified_sample(
        qa_pairs: list[dict],
        size: int | None,
        seed: int,
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
        remainder = size - n_per * 3

        s1 = rng.sample(t1, min(n_per + (1 if remainder > 0 else 0), len(t1)))
        s2 = rng.sample(t2, min(n_per + (1 if remainder > 1 else 0), len(t2)))
        s3 = rng.sample(t3, min(n_per, len(t3)))
        return s1 + s2 + s3

    # ── INGEST ────────────────────────────────────────────────────────────────

    def ingest(self) -> None:
        """
        Descarga PeerQA, chunkea, genera embeddings y guarda checkpoint.

        Checkpoint guardado:
          checkpoints/rag1_embeddings.npy  — array (N, 1024) float32
          checkpoints/rag1_chunks.json     — lista de {chunk_id, paper_id, text}
          checkpoints/rag1_qa_pairs.json   — lista de QA pairs para evaluate
        """
        logger.info("=== RAG 1 — Modo INGEST ===")
        self._log_gpu_state()

        # 1. Cargar dataset
        loader = PeerQALoader(self.config)
        data = loader.load()

        # Guardar QA pairs (necesarios para evaluate)
        qa_path = self.checkpoints_dir / "rag1_qa_pairs.json"
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(data["qa_pairs"], f, ensure_ascii=False, indent=2)
        logger.info("QA pairs guardados | n={} | path={}", len(data["qa_pairs"]), qa_path)

        # 2. Chunking — omitido si el corpus ya viene pre-chunkeado (mteb/PeerQA)
        if data.get("pre_chunked"):
            all_chunks = data["corpus"]
            logger.info("Corpus pre-chunkeado | {} pasajes directos", len(all_chunks))
        else:
            chunker = Chunker(self.config)
            all_chunks = chunker.chunk_documents(data["papers"])

        if not all_chunks:
            logger.error("No se generaron chunks. Verifica el dataset y la carga.")
            return

        # 3. Embedding — carga modelo → embede → descarga
        embedder = BGEEmbedder(self.config)
        embedder.load()

        texts = [c["text"] for c in all_chunks]
        logger.info("Generando embeddings para {} chunks", len(texts))
        embeddings = embedder.embed_batch(texts)

        embedder.unload()
        gc.collect()

        logger.info(
            "Embeddings generados | shape={} | dtype={}",
            embeddings.shape, embeddings.dtype,
        )

        # 4. Checkpoint
        emb_path = self.checkpoints_dir / "rag1_embeddings.npy"
        np.save(emb_path, embeddings.astype(np.float32))

        chunk_meta = [
            {"chunk_id": c["chunk_id"], "paper_id": c["paper_id"], "text": c["text"]}
            for c in all_chunks
        ]
        chunks_path = self.checkpoints_dir / "rag1_chunks.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunk_meta, f, ensure_ascii=False, indent=2)

        logger.success(
            "Ingest completado | {} chunks | checkpoint en {}",
            len(all_chunks), self.checkpoints_dir,
        )

    # ── EVALUATE ──────────────────────────────────────────────────────────────

    def evaluate(self) -> None:
        """
        Carga checkpoint, corre el pipeline completo y guarda JSON de resultados.

        Dos niveles de subset (configurables en config.yaml):
          metrics_subset: queries para MRR, Recall@k, ROUGE-L, BERTScore (gratis).
          judge_subset:   subconjunto de metrics_subset para RAGAS (consume tokens).
        Ambos se muestrean de forma estratificada por longitud de query.

        El JSON se guarda de una sola vez al final.
        """
        logger.info("=== RAG 1 — Modo EVALUATE ===")
        self._log_gpu_state()

        # Verificar checkpoint
        emb_path = self.checkpoints_dir / "rag1_embeddings.npy"
        chunks_path = self.checkpoints_dir / "rag1_chunks.json"
        qa_path = self.checkpoints_dir / "rag1_qa_pairs.json"

        for p in [emb_path, chunks_path, qa_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"No se encontró {p}.\n"
                    "Corre primero: python main.py --pipeline rag1 --mode ingest"
                )

        embeddings = np.load(emb_path)
        with open(chunks_path, encoding="utf-8") as f:
            chunks: list[dict] = json.load(f)
        with open(qa_path, encoding="utf-8") as f:
            qa_pairs: list[dict] = json.load(f)

        logger.info(
            "Checkpoint cargado | {} chunks | {} QA pairs",
            len(chunks), len(qa_pairs),
        )

        # ── SUBSET SAMPLING ────────────────────────────────────────────────
        eval_cfg = self.config["evaluation"]
        metrics_size = eval_cfg.get("metrics_subset_size")
        metrics_seed = eval_cfg.get("metrics_subset_seed", 42)
        judge_size   = eval_cfg.get("judge_subset_size")
        judge_seed   = eval_cfg.get("judge_subset_seed", 42)

        metrics_qa = self._stratified_sample(qa_pairs, metrics_size, metrics_seed)
        judge_qa   = self._stratified_sample(metrics_qa, judge_size, judge_seed)
        judge_ids  = {qa["query_id"] for qa in judge_qa}

        est_llm_calls = len(judge_qa) * 5  # ~5 calls por query (RAGAS Faithfulness)
        logger.info(
            "Subsets | total={} | metrics={} | judge={} (~{} LLM calls estimados)",
            len(qa_pairs), len(metrics_qa), len(judge_qa), est_llm_calls,
        )

        # Reconstruir Qdrant desde checkpoint
        store = QdrantStore(self.config)
        store.create_collection()
        store.upsert(chunks, embeddings)

        # Cargar embedder y LLM
        embedder = BGEEmbedder(self.config)
        embedder.load()
        llm = GroqLLM(self.config)

        k_candidates: int = self.config["retrieval"]["k_candidates"]
        top_n: int = self.config["retrieval"]["top_n_final"]
        k_values: list[int] = self.config["evaluation"]["recall_k_values"]

        per_query_results: list[dict] = []

        for i, qa in enumerate(metrics_qa):
            query_id = qa["query_id"]
            query = qa["query"]
            ground_truth = qa["ground_truth_evidence"]

            logger.info(
                "Query {}/{} | query_id={}",
                i + 1, len(metrics_qa), query_id,
            )

            query_vec = embedder.embed_query(query)
            retrieved = store.search(query_vec, k=k_candidates)

            logger.debug(
                "Recuperados: {} | scores: {}",
                len(retrieved),
                [round(c["score"], 3) for c in retrieved],
            )

            point_a = compute_point_a(retrieved, ground_truth, k_values)
            top_n_chunks = retrieved[:top_n]
            generated = llm.generate(query, top_n_chunks)
            rouge_l = compute_rouge_l(generated, qa["answer"])

            per_query_results.append({
                "query_id": query_id,
                "query": query,
                "retrieved_chunks": retrieved,
                "ground_truth_evidence": ground_truth,
                "point_a_pre_llm": point_a,
                "generated_response": generated,
                "point_b_post_llm": {"rouge_l": rouge_l, "bert_score_f1": None},
                "point_c_llm_judge": {"faithfulness": None},
                "in_judge_subset": query_id in judge_ids,
            })

        embedder.unload()
        gc.collect()

        # ── BERTScore batch (métricas clásicas — sin API tokens) ─────────
        logger.info("Calculando BERTScore en batch para {} queries", len(per_query_results))
        preds = [r["generated_response"] for r in per_query_results]
        refs  = [metrics_qa[i]["answer"] for i in range(len(per_query_results))]
        bs_scores = compute_bertscore_batch(preds, refs)
        for r, bs in zip(per_query_results, bs_scores):
            r["point_b_post_llm"]["bert_score_f1"] = bs

        # ── RAGAS Faithfulness — solo judge subset ────────────────────────
        judge_results = [r for r in per_query_results if r["in_judge_subset"]]
        logger.info(
            "Calculando RAGAS Faithfulness para judge subset ({} queries, ~{} LLM calls)",
            len(judge_results), len(judge_results) * 5,
        )
        qa_answer_map = {qa["query_id"]: qa["answer"] for qa in judge_qa}
        ragas_scores = compute_ragas_batch(
            queries=[r["query"] for r in judge_results],
            responses=[r["generated_response"] for r in judge_results],
            contexts_per_query=[[c["text"] for c in r["retrieved_chunks"]] for r in judge_results],
            references=[qa_answer_map[r["query_id"]] for r in judge_results],
        )
        for r, rs in zip(judge_results, ragas_scores):
            r["point_c_llm_judge"] = rs

        # Agregar métricas
        aggregate = aggregate_metrics(per_query_results, k_values)

        faith = aggregate["point_c"]["faithfulness"]
        logger.success(
            "Evaluación completada | MRR={:.4f} | ROUGE-L={:.4f} | Faithfulness={}",
            aggregate["point_a"]["mrr"],
            aggregate["point_b"]["rouge_l"],
            f"{faith:.4f}" if faith is not None else "N/A",
        )

        # Guardar JSON completo de una sola vez
        output = {
            "condition": "rag1_baseline",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "config_snapshot": self.config,
            "subset_info": {
                "total_qa_pairs": len(qa_pairs),
                "metrics_subset_size": len(metrics_qa),
                "judge_subset_size": len(judge_qa),
                "metrics_subset_seed": metrics_seed,
                "judge_subset_seed": judge_seed,
            },
            "per_query_results": per_query_results,
            "aggregate": aggregate,
        }

        out_path = self.results_dir / "metrics_post_llm.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.success("Resultados guardados en {}", out_path)

    # ── JUDGE ONLY ────────────────────────────────────────────────────────────

    def run_judge_only(self) -> None:
        """
        Lee el JSON de resultados ya existente y aplica RAGAS Faithfulness
        sin repetir ingest ni generación. Guarda el JSON actualizado en su
        mismo path. Útil cuando el evaluate completó pero la cuota de Groq
        se agotó antes de llegar al judge.
        """
        logger.info("=== RAG 1 — Modo JUDGE (solo RAGAS, sin regenerar) ===")

        out_path = self.results_dir / "metrics_post_llm.json"
        if not out_path.exists():
            raise FileNotFoundError(
                f"No se encontró {out_path}.\n"
                "Corre primero: python main.py --pipeline rag1 --mode evaluate"
            )

        with open(out_path, encoding="utf-8") as f:
            output: dict = json.load(f)

        per_query_results: list[dict] = output["per_query_results"]
        k_values: list[int] = self.config["evaluation"]["recall_k_values"]

        already_scored = sum(
            1 for r in per_query_results
            if r.get("point_c_llm_judge", {}).get("faithfulness") is not None
        )
        pending = [
            r for r in per_query_results
            if r.get("in_judge_subset", True)
            and r.get("point_c_llm_judge", {}).get("faithfulness") is None
        ]

        logger.info(
            "JSON cargado | {} queries | ya con score: {} | pendientes: {} | timestamp: {}",
            len(per_query_results), already_scored, len(pending), output.get("timestamp"),
        )

        if not pending:
            logger.success("Todas las queries ya tienen faithfulness score. Nada que hacer.")
            return

        queries   = [r["query"]              for r in pending]
        responses = [r["generated_response"] for r in pending]
        contexts  = [[c["text"] for c in r["retrieved_chunks"]] for r in pending]
        refs      = [r.get("ground_truth_answer", "") or "" for r in pending]

        ragas_scores = compute_ragas_batch(queries, responses, contexts, refs)
        for r, rs in zip(pending, ragas_scores):
            r["point_c_llm_judge"] = rs

        aggregate = aggregate_metrics(per_query_results, k_values)
        output["aggregate"] = aggregate
        output["judge_timestamp"] = datetime.utcnow().isoformat() + "Z"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        faith = aggregate["point_c"]["faithfulness"]
        logger.success(
            "JUDGE completado | Faithfulness={} | JSON actualizado en {}",
            f"{faith:.4f}" if faith is not None else "N/A",
            out_path,
        )
