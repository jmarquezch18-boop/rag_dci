"""
peerqa_section_loader.py
========================
Extiende la carga de mteb/PeerQA para incluir el campo `title` de cada
pasaje del corpus como etiqueta de sección.

Posición en el pipeline RAG 2: reemplaza a DoclingParser+SectionChunker
para el dataset mteb/PeerQA. El corpus de mteb/PeerQA ya tiene el campo
`title` que contiene el heading de sección del pasaje (e.g. 'Introduction',
'Results', 'Methods'). Esto hace innecesario descargar los PDFs originales.

DoclingParser sigue siendo válido para pipelines sobre PDFs arbitrarios,
pero para mteb/PeerQA esta ruta es más eficiente y exacta: el heading viene
del propio paper, no de una heurística de matching sobre texto plano.

Diferencia con peerqa_loader.py (RAG 1): agrega `section` y `source_url`
al payload de cada chunk. Sin este campo, QdrantStoreRAG2 no puede filtrar
por sección. peerqa_loader.py NO se toca.

El paper_id se deriva del chunk_id eliminando el sufijo de índice:
  'nlpeer/F1000-22/10-170_5' → paper_id = 'nlpeer/F1000-22/10-170'
"""

from __future__ import annotations

import os
from loguru import logger
from datasets import load_dataset
from dotenv import load_dotenv

from src.ingestion.docling_parser import _infer_section

load_dotenv()

_EMPTY_TITLE_LABEL = "Unknown"


def _extract_paper_id(chunk_id: str) -> str:
    """
    Extrae el paper_id del chunk_id de mteb/PeerQA.

    El formato de mteb/PeerQA es '{source}/{paper_id}_{chunk_idx}'.
    El paper_id es todo lo anterior al último underscore+número.

    Args:
        chunk_id: ID del corpus, e.g. 'nlpeer/F1000-22/10-170_5'.
    Returns:
        paper_id, e.g. 'nlpeer/F1000-22/10-170'.
    """
    parts = chunk_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return chunk_id


class PeerQASectionLoader:
    """
    Carga mteb/PeerQA con secciones derivadas del campo `title` del corpus.

    Preserva exactamente el mismo esquema que peerqa_loader.py para QA pairs
    y qrels, pero agrega `section` y `source_url` al corpus para que
    QdrantStoreRAG2 pueda filtrar por sección.
    """

    DATASET_ID = "mteb/PeerQA"

    def __init__(self, config: dict):
        self.subset_size = config["data"].get("subset_size")
        raw_token = os.getenv("HF_TOKEN", "").strip()
        self.hf_token = raw_token if raw_token else None

    def load(self) -> dict:
        """
        Carga corpus con secciones, queries y qrels de mteb/PeerQA.

        El campo `section` de cada chunk se deriva del campo `title` del
        corpus vía _infer_section(). Para títulos vacíos (''), se asigna
        'Unknown'. Para títulos que son literalmente un nombre de sección
        (e.g. 'Introduction'), se mapea directamente.

        Returns:
            Dict con:
              "corpus"  — list[dict] con chunk_id, paper_id, section,
                          text, source_url
              "qa_pairs" — list[dict] idéntico a peerqa_loader.py
              "pre_chunked" — True
        """
        logger.info(
            "Cargando mteb/PeerQA con secciones | corpus + queries + qrels"
        )

        corpus_ds = load_dataset(
            self.DATASET_ID, "PeerQA-corpus", split="test", token=self.hf_token
        )
        queries_ds = load_dataset(
            self.DATASET_ID, "PeerQA-queries", split="test", token=self.hf_token
        )
        qrels_ds = load_dataset(
            self.DATASET_ID, "PeerQA-qrels", split="test", token=self.hf_token
        )

        logger.info(
            "Descargado | corpus={} | queries={} | qrels={}",
            len(corpus_ds), len(queries_ds), len(qrels_ds),
        )

        # ── Corpus con secciones ──────────────────────────────────────────────
        all_corpus = []
        section_counts: dict[str, int] = {}
        for row in corpus_ds:
            text = row["text"].strip()
            if not text:
                continue

            raw_title = (row.get("title") or "").strip()
            section = _infer_section(raw_title) if raw_title else _EMPTY_TITLE_LABEL

            section_counts[section] = section_counts.get(section, 0) + 1
            all_corpus.append({
                "chunk_id": row["id"],
                "paper_id": _extract_paper_id(row["id"]),
                "section": section,
                "text": text,
                "source_url": "",
            })

        logger.info(
            "Corpus con secciones cargado | {} chunks | distribución: {}",
            len(all_corpus),
            {k: v for k, v in sorted(section_counts.items(), key=lambda x: -x[1])[:8]},
        )

        corpus_text_map = {c["chunk_id"]: c["text"] for c in all_corpus}

        # ── Qrels ─────────────────────────────────────────────────────────────
        qrels_map: dict[str, list[str]] = {}
        for row in qrels_ds:
            qrels_map.setdefault(row["query-id"], []).append(row["corpus-id"])

        # ── QA pairs (idéntico a peerqa_loader.py) ────────────────────────────
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

        # ── Subset ────────────────────────────────────────────────────────────
        if self.subset_size:
            qa_pairs = qa_pairs[: self.subset_size]
            needed_ids = {cid for qa in qa_pairs for cid in qa["ground_truth_evidence"]}
            relevant_corpus = [c for c in all_corpus if c["chunk_id"] in needed_ids]
            other_corpus = [c for c in all_corpus if c["chunk_id"] not in needed_ids]

            import random as _random
            _random.seed(42)
            n_other = min(len(other_corpus), max(0, self.subset_size * 10 - len(relevant_corpus)))
            corpus = relevant_corpus + _random.sample(other_corpus, n_other)
            logger.info(
                "Subset | {} queries | {} corpus ({} rel + {} neg)",
                len(qa_pairs), len(corpus), len(relevant_corpus), n_other,
            )
        else:
            corpus = all_corpus

        logger.success(
            "PeerQA+secciones listo | {} pasajes | {} queries",
            len(corpus), len(qa_pairs),
        )
        return {"corpus": corpus, "qa_pairs": qa_pairs, "pre_chunked": True}
