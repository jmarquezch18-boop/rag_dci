"""
rag3_1_granite_docling.py
=========================
Pipeline RAG 3.1: DCI agéntico con corpus Docling + IBM Granite 3.1 Dense 8B.

Posición en el ablation:
  RAG 3   — DCI + corpus texto plano (PeerQA) + DeepSeek-R1 14B + oracle Jaccard
  RAG 3.1 — DCI + corpus Docling (PDFs) + IBM Granite 3.1 8B + oracle Containment
  RAG 3.2 — DCI + corpus Docling (PDFs) + DeepSeek-R1 14B + oracle Containment

Variables que cambian respecto a RAG 3:
  1. Corpus: Docling-procesado (mayor calidad, detección de secciones por PDF)
  2. Agent LLM: IBM Granite 3.1 Dense 8B (via Ollama, tag: granite3.1-dense:8b)
  3. Oracle: containment recall >= 0.5 en lugar de Jaccard >= 0.75

Variable de control: mismas QA pairs, mismo formato de herramientas, mismo Groq para generación.

Prerrequisitos:
  1. ollama pull granite3.1-dense:8b
  2. python main.py --pipeline rag3 --mode ingest   (para qa_pairs y corpus_map)
  3. python main.py --pipeline rag3_1 --mode ingest  (construye corpus_docling/)
  4. python main.py --pipeline rag3_1 --mode evaluate
"""

from __future__ import annotations

import copy
import gc
import json
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from src.curation.oracle import CurationOracle
from src.dci.agent import DCIAgent
from src.dci.evidence_collector import EvidenceCollector
from src.evaluation.metrics import compute_rouge_l, compute_bertscore_batch
from src.evaluation.metrics_rag2 import compute_faithfulness_cosine
from src.retrieval.embedder import BGEEmbedder


# ── Helpers reutilizados de rag3_dci_sail.py ─────────────────────────────────

def _dci_chunk_relevant(chunk_text: str, gt_texts: list[str], threshold: float = 0.40) -> bool:
    """Containment recall: True si chunk contiene >= threshold de palabras de algún GT."""
    chunk_words = set(chunk_text.lower().split())
    for gt in gt_texts:
        gt_words = set(gt.lower().split())
        if gt_words and len(chunk_words & gt_words) / len(gt_words) >= threshold:
            return True
    return False


def _compute_point_a(evidence_chunks: list[dict], gt_texts: list[str], k_values: list[int]) -> dict:
    empty = {"mrr": 0.0, **{f"recall_at_{k}": 0.0 for k in k_values},
             "precision_at_5": 0.0, "ndcg_at_5": 0.0}
    if not gt_texts or not evidence_chunks:
        return empty

    flags = [_dci_chunk_relevant(c.get("text", ""), gt_texts) for c in evidence_chunks]
    n_gt = len(gt_texts)
    mrr = next((1.0 / (r + 1) for r, f in enumerate(flags) if f), 0.0)
    recalls = {f"recall_at_{k}": min(sum(flags[:k]), n_gt) / n_gt for k in k_values}
    top5 = flags[:5]
    p5 = sum(top5) / len(top5) if top5 else 0.0
    dcg = sum(1.0 / math.log2(r + 2) for r, f in enumerate(top5) if f)
    n_ideal = min(sum(flags), 5)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(n_ideal))
    return {"mrr": mrr, **recalls, "precision_at_5": p5, "ndcg_at_5": dcg / idcg if idcg else 0.0}


def _compute_point_b(approved: list[dict], gt_texts: list[str], k_values: list[int]) -> dict:
    empty = {"mrr": 0.0, **{f"recall_at_{k}": 0.0 for k in k_values},
             "coverage": 0.0, "precision_at_5": 0.0, "ndcg_at_5": 0.0}
    if not approved:
        return empty
    return _compute_point_a(approved, gt_texts, k_values) | {
        "coverage": sum(_dci_chunk_relevant(c.get("text", ""), gt_texts) for c in approved) / len(gt_texts)
        if gt_texts else 0.0
    }


def _stratified_sample(qa_pairs: list[dict], size: Optional[int], seed: int) -> list[dict]:
    if not qa_pairs or size is None or size >= len(qa_pairs):
        return list(qa_pairs)
    sorted_pairs = sorted(qa_pairs, key=lambda q: len(q["query"]))
    n = len(sorted_pairs)
    t1, t2, t3 = sorted_pairs[:n//3], sorted_pairs[n//3:2*n//3], sorted_pairs[2*n//3:]
    rng = random.Random(seed)
    n_per, rem = size // 3, size - (size // 3) * 3
    return (rng.sample(t1, min(n_per + (1 if rem > 0 else 0), len(t1))) +
            rng.sample(t2, min(n_per + (1 if rem > 1 else 0), len(t2))) +
            rng.sample(t3, min(n_per, len(t3))))


def _sanitize_paper_id(paper_id: str) -> str:
    """nlpeer/F1000-22/10-170 → nlpeer_F1000-22_10-170"""
    return paper_id.replace("/", "_")


def _sanitize_section(section: str) -> str:
    """'Related Work' → 'related_work'"""
    return (section or "unknown").lower().replace(" ", "_")


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAG31Pipeline:
    """
    RAG 3.1: DCI agéntico con corpus Docling + IBM Granite 3.1 Dense 8B.

    Modos:
      ingest   — construye corpus_docling/ desde rag2_chunks.json
      evaluate — evalúa el pipeline completo
    """

    GRANITE_MODEL = "granite3.1-dense:8b"
    CORPUS_DOCLING_DIR = "./corpus_docling"
    CONDITION_NAME = "rag3_1_granite_docling"
    LOG_NAME = "RAG 3.1"

    def __init__(self, config: dict):
        self.config = config
        self.rag3_cfg = config.get("rag3", {})
        eval_cfg = self.rag3_cfg.get("evaluation", {})

        self.results_dir = Path(config["paths"]["results"]) / "rag3_1"
        self.agent_traces_dir = self.results_dir / "agent_traces"
        self.checkpoints_dir = Path(config["paths"]["checkpoints"])
        self.corpus_docling = Path(self.CORPUS_DOCLING_DIR).resolve()

        self.metrics_subset_size: Optional[int] = eval_cfg.get("metrics_subset_size", 200)
        self.metrics_subset_seed: int = eval_cfg.get("metrics_subset_seed", 42)
        self.min_approved: int = self.rag3_cfg.get("min_approved_for_generation", 1)
        self.fallback_top_n: int = self.rag3_cfg.get("fallback_top_n", 3)
        self.faith_threshold: float = self.rag3_cfg.get("faithfulness_cosine_threshold", 0.75)
        self.k_values: list[int] = config["evaluation"]["recall_k_values"]

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.agent_traces_dir.mkdir(parents=True, exist_ok=True)

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest(self) -> None:
        """
        Construye corpus_docling/ desde los chunks procesados por Docling en RAG 2.

        Estructura de salida:
          corpus_docling/
            nlpeer_F1000-22_10-170/
              abstract.txt
              introduction.txt
              methodology.txt
              ...

        Cada archivo contiene el texto de todos los chunks Docling de esa sección,
        separados por una línea en blanco — mismo formato que corpus/ de RAG 3.

        Prerrequisito: python main.py --pipeline rag2 --mode ingest
        """
        logger.info("=== RAG 3.1 — Modo INGEST (corpus Docling) ===")

        chunks_path = self.checkpoints_dir / "rag2_chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"No se encontró {chunks_path}.\n"
                "Corre primero: python main.py --pipeline rag2 --mode ingest"
            )

        with open(chunks_path, encoding="utf-8") as f:
            chunks: list[dict] = json.load(f)

        logger.info("Chunks Docling cargados: {}", len(chunks))

        # Agrupar texto por (paper_id_sanitizado, section_sanitizada)
        sections: dict[tuple[str, str], list[str]] = defaultdict(list)
        for chunk in chunks:
            pid = _sanitize_paper_id(chunk.get("paper_id", "unknown"))
            sec = _sanitize_section(chunk.get("section", "unknown"))
            text = chunk.get("text", "").strip()
            if text:
                sections[(pid, sec)].append(text)

        # Escribir corpus_docling/
        self.corpus_docling.mkdir(parents=True, exist_ok=True)
        n_files = 0
        for (pid, sec), texts in sections.items():
            paper_dir = self.corpus_docling / pid
            paper_dir.mkdir(exist_ok=True)
            section_file = paper_dir / f"{sec}.txt"
            section_file.write_text("\n\n".join(texts), encoding="utf-8")
            n_files += 1

        n_papers = len({pid for pid, _ in sections})
        logger.success(
            "corpus_docling/ construido | {} papers | {} archivos de sección",
            n_papers, n_files,
        )

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self) -> None:
        """
        Corre el pipeline RAG 3.1 completo.

        Diferencias respecto a RAG 3:
          - Corpus: corpus_docling/ (Docling) en lugar de corpus/ (PeerQA plain text)
          - Agente: IBM Granite 3.1 Dense 8B en lugar de DeepSeek-R1 14B
          - Oracle: containment recall >= 0.5 en lugar de Jaccard >= 0.75
          - QA pairs y corpus_map: reutilizados de RAG 3 (mismo ground truth)
        """
        logger.info("=== {} — Modo EVALUATE ({} + Docling) ===", self.LOG_NAME, self.GRANITE_MODEL)

        if not self.corpus_docling.exists():
            raise FileNotFoundError(
                f"corpus_docling/ no encontrado en {self.corpus_docling}.\n"
                "Corre primero: python main.py --pipeline rag3_1 --mode ingest"
            )

        # QA pairs y corpus_map de RAG 3 (mismo ground truth)
        qa_path  = self.checkpoints_dir / "rag3_dci_qa_pairs.json"
        map_path = self.checkpoints_dir / "rag3_dci_corpus_map.json"
        for p in [qa_path, map_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"No se encontró {p}.\n"
                    "Corre primero: python main.py --pipeline rag3 --mode ingest"
                )

        with open(qa_path,  encoding="utf-8") as f:
            qa_pairs: list[dict] = json.load(f)
        with open(map_path, encoding="utf-8") as f:
            corpus_map: dict[str, str] = json.load(f)

        eval_qa = _stratified_sample(qa_pairs, self.metrics_subset_size, self.metrics_subset_seed)
        logger.info("QA pairs | total={} | evaluando={}", len(qa_pairs), len(eval_qa))

        # Config parcheado para apuntar al corpus Docling y al modelo Granite
        cfg31 = copy.deepcopy(self.config)
        cfg31.setdefault("rag3", {}).setdefault("corpus", {})["path"] = str(self.corpus_docling)
        cfg31["rag3"].setdefault("ollama", {})["model"] = self.GRANITE_MODEL

        agent = DCIAgent(cfg31)
        agent.check_ollama()

        collector = EvidenceCollector(self.config)
        oracle = CurationOracle(self.config, metric="containment")

        # Reutilizamos GroqLLM con rotación de claves
        from src.generation.llm import GroqLLM
        llm = GroqLLM(self.config)

        per_query_results: list[dict] = []

        for i, qa in enumerate(eval_qa):
            query_id = qa["query_id"]
            query    = qa["query"]
            gt_ids   = qa["ground_truth_evidence"]
            gt_texts = [corpus_map[gid] for gid in gt_ids if gid in corpus_map]

            logger.info("Query {}/{} | id={}", i + 1, len(eval_qa), query_id)

            # ── Paso 1: DCI retrieval (Granite en corpus_docling/) ────────────
            agent_result = agent.run_agent(query)
            evidence_raw    = agent_result["evidence_raw"]
            evidence_chunks = collector.format_evidence(evidence_raw)

            # ── Punto A ───────────────────────────────────────────────────────
            point_a = _compute_point_a(evidence_chunks, gt_texts, self.k_values)

            # ── Paso 2: Curación SAIL-RAG con containment ─────────────────────
            curation = oracle.curate(
                query=query,
                reranked_chunks=evidence_chunks,
                ground_truth_ids=gt_ids,
                ground_truth_texts=gt_texts,
            )
            approved = curation["approved_chunks"]

            # ── Punto B ───────────────────────────────────────────────────────
            point_b = _compute_point_b(approved, gt_texts, self.k_values)

            # ── Paso 3: Generación Groq ───────────────────────────────────────
            context_chunks = approved if len(approved) >= self.min_approved \
                             else evidence_chunks[:self.fallback_top_n]
            used_fallback = len(approved) < self.min_approved
            generated = llm.generate(query, context_chunks)
            rouge_l   = compute_rouge_l(generated, qa.get("answer", ""))

            per_query_results.append({
                "query_id":    query_id,
                "query":       query,
                "agent_result": {
                    "evidence_count":  len(evidence_chunks),
                    "iterations_used": agent_result.get("iterations_used", 0),
                    "think_len":       agent_result.get("think_len", 0),
                    "searched_terms":  agent_result.get("searched_terms", []),
                },
                "evidence_chunks":        evidence_chunks,
                "approved_chunks":        approved,
                "ground_truth_evidence":  gt_ids,
                "point_a_pre_curation":   point_a,
                "point_b_post_curation":  point_b,
                "generated_response":     generated,
                "used_fallback":          used_fallback,
                "curation_trace":         curation["oracle_decisions"],
                "point_c_post_llm": {
                    "rouge_l":       rouge_l,
                    "bert_score_f1": None,
                    "faithfulness":  None,
                },
            })

            logger.info(
                "  A.MRR={:.3f} | B.cov={:.3f} | aprobados={} | fallback={}",
                point_a["mrr"], point_b["coverage"], len(approved), used_fallback,
            )

        # ── Phase 2: BERTScore + Faithfulness ────────────────────────────────
        logger.info("Phase 2: BERTScore GPU")
        preds = [r["generated_response"] for r in per_query_results]
        refs  = [eval_qa[j]["answer"]    for j in range(len(per_query_results))]
        bs_scores = compute_bertscore_batch(preds, refs)

        logger.info("Phase 2: Faithfulness coseno GPU")
        faith_embedder = BGEEmbedder(self.config)
        faith_embedder.load()

        for r, bs in zip(per_query_results, bs_scores):
            r["point_c_post_llm"]["bert_score_f1"] = bs
            context = r["approved_chunks"] or r["evidence_chunks"][:self.fallback_top_n]
            r["point_c_post_llm"]["faithfulness"] = compute_faithfulness_cosine(
                r["generated_response"], context, faith_embedder, self.faith_threshold,
            )

        faith_embedder.unload()
        gc.collect()

        # ── Agregar ───────────────────────────────────────────────────────────
        def _mean(key: str, sub: str) -> float:
            vals = [r[key].get(sub, 0.0) or 0.0 for r in per_query_results]
            return sum(vals) / len(vals) if vals else 0.0

        aggregate = {
            "point_a": {
                "mrr":            _mean("point_a_pre_curation", "mrr"),
                "recall_at_5":    _mean("point_a_pre_curation", "recall_at_5"),
                "precision_at_5": _mean("point_a_pre_curation", "precision_at_5"),
                "ndcg_at_5":      _mean("point_a_pre_curation", "ndcg_at_5"),
                **{f"recall_at_{k}": _mean("point_a_pre_curation", f"recall_at_{k}")
                   for k in self.k_values},
            },
            "point_b": {
                "coverage":    _mean("point_b_post_curation", "coverage"),
                "mrr":         _mean("point_b_post_curation", "mrr"),
                "recall_at_5": _mean("point_b_post_curation", "recall_at_5"),
            },
            "point_c": {
                "rouge_l":       _mean("point_c_post_llm", "rouge_l"),
                "bert_score_f1": _mean("point_c_post_llm", "bert_score_f1"),
                "faithfulness":  _mean("point_c_post_llm", "faithfulness"),
            },
        }

        logger.success(
            "{} completado | A.MRR={:.4f} | B.cov={:.4f} | C.ROUGE-L={:.4f}",
            self.LOG_NAME,
            aggregate["point_a"]["mrr"],
            aggregate["point_b"]["coverage"],
            aggregate["point_c"]["rouge_l"],
        )

        output = {
            "condition":   self.CONDITION_NAME,
            "timestamp":   datetime.utcnow().isoformat() + "Z",
            "config_snapshot": {
                "agent_model":    self.GRANITE_MODEL,
                "corpus":         "docling",
                "oracle_metric":  "containment",
                "oracle_threshold": 0.5,
            },
            "subset_info": {
                "total_qa_pairs":      len(qa_pairs),
                "metrics_subset_size": len(eval_qa),
                "metrics_subset_seed": self.metrics_subset_seed,
            },
            "per_query_results": per_query_results,
            "aggregate":         aggregate,
        }

        out_path = self.results_dir / "metrics_dci.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.success("Resultados guardados en {}", out_path)
