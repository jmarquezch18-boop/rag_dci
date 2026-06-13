"""
peerqa_loader.py
================
Carga PeerQA desde mteb/PeerQA (formato MTEB/BEIR — Parquet moderno).

Posición en el pipeline: Paso 1 — entrada de datos.

El dataset tiene 3 configs:
  PeerQA-corpus  (18 593 pasajes pre-chunkeados del paper)
  PeerQA-queries (136 preguntas de revisores)
  PeerQA-qrels   (389 pares query→corpus de relevancia ground truth)

Decisión: usar mteb/PeerQA en lugar de UKPLab/PeerQA porque la versión
original usa un script de carga antiguo que la librería datasets ≥3.x
ya no soporta. El formato MTEB es Parquet puro y es el estándar BEIR.

El corpus ya viene pre-chunkeado → el pipeline lo ingesta directo en
Qdrant sin pasar por el Chunker (flag pre_chunked=True en el retorno).
La evaluación usa IDs exactos del corpus, no texto fuzzy-matched.
"""

from __future__ import annotations

import os
from loguru import logger
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


class PeerQALoader:
    """Carga los 3 configs de mteb/PeerQA y los normaliza al esquema interno."""

    DATASET_ID = "mteb/PeerQA"

    def __init__(self, config: dict):
        self.subset_size = config["data"].get("subset_size")
        raw_token = os.getenv("HF_TOKEN", "").strip()
        self.hf_token = raw_token if raw_token else None

    def load(self) -> dict:
        """
        Carga corpus, queries y qrels de mteb/PeerQA.

        Returns:
            Dict con:
              "corpus"      — list[dict] con chunk_id, paper_id, text
              "qa_pairs"    — list[dict] con query_id, query, answer, ground_truth_evidence
              "pre_chunked" — True (indica al pipeline que omita el Chunker)
        """
        logger.info("Cargando mteb/PeerQA | corpus + queries + qrels")

        corpus_ds = load_dataset(self.DATASET_ID, "PeerQA-corpus",  split="test", token=self.hf_token)
        queries_ds = load_dataset(self.DATASET_ID, "PeerQA-queries", split="test", token=self.hf_token)
        qrels_ds   = load_dataset(self.DATASET_ID, "PeerQA-qrels",   split="test", token=self.hf_token)

        logger.info(
            "Descargado | corpus={} pasajes | queries={} | qrels={}",
            len(corpus_ds), len(queries_ds), len(qrels_ds),
        )

        # ── Corpus completo (dict para lookup rápido) ─────────────────────────
        all_corpus = [
            {
                "chunk_id": row["id"],
                "paper_id": row["id"],
                "text": row["text"].strip(),
            }
            for row in corpus_ds
            if row["text"].strip()
        ]
        corpus_text_map = {c["chunk_id"]: c["text"] for c in all_corpus}

        # ── Qrels: query_id → [corpus_id, ...] ───────────────────────────────
        qrels_map: dict[str, list[str]] = {}
        for row in qrels_ds:
            qrels_map.setdefault(row["query-id"], []).append(row["corpus-id"])

        # ── QA pairs ─────────────────────────────────────────────────────────
        qa_pairs = []
        for row in queries_ds:
            qid = row["id"]
            relevant_ids = qrels_map.get(qid, [])
            reference = " ".join(
                corpus_text_map[cid] for cid in relevant_ids if cid in corpus_text_map
            )
            qa_pairs.append({
                "query_id": qid,
                "paper_id": None,
                "query": row["text"],
                "answer": reference,
                "ground_truth_evidence": relevant_ids,
            })

        # ── Subset inteligente: siempre incluye los corpus entries relevantes ─
        if self.subset_size:
            qa_pairs = qa_pairs[: self.subset_size]

            # IDs relevantes para las queries seleccionadas
            needed_ids = {cid for qa in qa_pairs for cid in qa["ground_truth_evidence"]}

            # Corpus relevante + resto hasta completar ~10x las queries
            relevant_corpus = [c for c in all_corpus if c["chunk_id"] in needed_ids]
            other_corpus    = [c for c in all_corpus if c["chunk_id"] not in needed_ids]

            import random as _random
            _random.seed(42)
            n_other = min(len(other_corpus), max(0, self.subset_size * 10 - len(relevant_corpus)))
            corpus = relevant_corpus + _random.sample(other_corpus, n_other)

            logger.info(
                "Subset aplicado | {} queries | {} corpus ({} relevantes + {} negativos)",
                len(qa_pairs), len(corpus), len(relevant_corpus), n_other,
            )
        else:
            corpus = all_corpus

        logger.success(
            "PeerQA listo | {} pasajes de corpus | {} queries",
            len(corpus), len(qa_pairs),
        )
        return {"corpus": corpus, "qa_pairs": qa_pairs, "pre_chunked": True}
