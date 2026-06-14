"""
rag3_dci_sail.py
================
Pipeline completo RAG 3: DCI (retrieval agéntico) + SAIL-RAG (oráculo de curación).

Arquitectura:
  query
    → DCIAgent.run_agent(query)          # retrieval: DeepSeek-R1 busca en corpus/
    → EvidenceCollector.format_evidence() # normalizar al esquema estándar de chunk
    → [PUNTO A: métricas pre-curación]
    → CurationOracle.curate()             # SAIL-RAG: filtra con ground truth Jaccard
    → [PUNTO B: métricas post-curación]
    → GroqLLM.generate()                  # generación: llama-3.3-70b-versatile
    → [PUNTO C: métricas post-LLM]
    → guardar traza JSON en results/rag3/agent_traces/

Qué hace diferente vs RAG 1 y RAG 2:
  - Retrieval: el agente DCI reemplaza el retrieval vectorial de Qdrant.
    DeepSeek-R1 itera hasta max_iterations buscando con grep/read_lines.
  - Curación: MISMO oráculo SAIL-RAG que RAG 2 (oracle.py sin cambios).
  - Generación: MISMO Groq LLM que RAG 1 y RAG 2 (llm.py sin cambios).

Por qué esta comparación es válida: el LLM de generación es Groq en los 3 RAGs.
La única diferencia es el retrieval (vectorial one-shot vs agéntico iterativo).
El oráculo garantiza que la calidad de la curación sea comparable.

Modos:
  ingest   — carga PeerQA, construye corpus/, guarda checkpoints DCI
  evaluate — corre el pipeline completo y guarda métricas + trazas
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from src.curation.oracle import CurationOracle
from src.dci.agent import DCIAgent
from src.dci.evidence_collector import EvidenceCollector
from src.evaluation.metrics import compute_rouge_l, compute_bertscore_batch
from src.evaluation.metrics_rag2 import compute_faithfulness_cosine
from src.generation.llm import GroqLLM
from src.ingestion.corpus_builder import CorpusBuilder


# ── Rotación de claves Groq ───────────────────────────────────────────────────

def _load_groq_keys() -> list[str]:
    """
    Carga todas las claves GROQ_API_KEY* del entorno.

    Busca GROQ_API_KEY, GROQ_API_KEY2 … GROQ_API_KEY9 y retorna las que
    existan. Permite rotar automáticamente cuando una clave alcanza el
    rate limit de Groq (30 req/min por clave en tier gratuito).
    """
    keys: list[str] = []
    for suffix in ["", "2", "3", "4", "5", "6", "7", "8", "9"]:
        k = os.getenv(f"GROQ_API_KEY{suffix}", "").strip()
        if k and k not in keys:
            keys.append(k)
    return keys


class _RotatingGroqLLM:
    """
    Wrapper sobre GroqLLM con rotación automática de claves API.

    Con 200 queries y 6 claves (30 req/min/clave), la rotación evita
    bloqueos por rate limit sin modificar llm.py ni los pipelines de
    RAG 1 o RAG 2.
    """

    def __init__(self, config: dict, keys: list[str]):
        self.config = config
        self.keys = keys if keys else [os.getenv("GROQ_API_KEY", "")]
        self.idx = 0
        self._client = self._make(self.idx)
        logger.info("RotatingGroqLLM | {} claves disponibles", len(self.keys))

    def _make(self, idx: int) -> GroqLLM:
        os.environ["GROQ_API_KEY"] = self.keys[idx]
        return GroqLLM(self.config)

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        """Genera con rotación de clave si hay rate limit."""
        for attempt in range(len(self.keys)):
            result = self._client.generate(query, context_chunks)
            if result:
                return result
            # resultado vacío → rate limit agotado → rotar
            self.idx = (self.idx + 1) % len(self.keys)
            logger.warning(
                "Groq key agotada, rotando a clave {}/{}", self.idx + 1, len(self.keys)
            )
            self._client = self._make(self.idx)
            time.sleep(2)
        return ""


# ── Helpers de métricas DCI ───────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    """Similitud de Jaccard sobre conjuntos de palabras en minúsculas."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _dci_chunk_relevant(chunk_text: str, gt_texts: list[str]) -> bool:
    """
    Determina si un chunk DCI contiene evidencia del ground truth.

    Usa "containment recall": el chunk es relevante si al menos el 40% de las
    palabras de algún pasaje de ground truth aparecen en él. Esto es más
    apropiado que Jaccard simétrico porque los chunks DCI son fragmentos de
    secciones completas que contienen los pasajes cortos del ground truth.

    Args:
        chunk_text: Texto del chunk DCI (puede ser una sección completa).
        gt_texts: Textos de los pasajes ground truth de PeerQA.
    Returns:
        True si el chunk contiene al menos un pasaje de ground truth.
    """
    chunk_words = set(chunk_text.lower().split())
    for gt in gt_texts:
        gt_words = set(gt.lower().split())
        if not gt_words:
            continue
        recall = len(chunk_words & gt_words) / len(gt_words)
        if recall >= 0.40:
            return True
    return False


def _compute_point_a_dci(
    evidence_chunks: list[dict],
    gt_texts: list[str],
    k_values: list[int],
) -> dict:
    """
    Métricas de recuperación para el Punto A del pipeline DCI.

    Usa containment recall (40%) en lugar de chunk_id exacto porque los IDs
    DCI no corresponden a los IDs del corpus PeerQA. La relevancia binaria se
    determina por solapamiento de palabras con los textos del ground truth.

    Args:
        evidence_chunks: Chunks formateados por EvidenceCollector.
        gt_texts: Textos del ground truth de PeerQA.
        k_values: Valores de k para Recall@k.
    Returns:
        Dict con mrr, recall_at_k, ndcg_at_5, precision_at_5.
    """
    empty = {
        "mrr": 0.0,
        **{f"recall_at_{k}": 0.0 for k in k_values},
        "precision_at_5": 0.0,
        "ndcg_at_5": 0.0,
        "deepeval_contextual_relevancy": 0.0,
    }
    if not gt_texts or not evidence_chunks:
        return empty

    flags = [_dci_chunk_relevant(c.get("text", ""), gt_texts) for c in evidence_chunks]
    n_gt = len(gt_texts)

    # MRR
    mrr = next((1.0 / (r + 1) for r, f in enumerate(flags) if f), 0.0)

    # Recall@k (tratamos cada gt_text como una unidad de relevancia distinta)
    recalls: dict[str, float] = {}
    for k in k_values:
        top_k_rel = sum(flags[:k])
        recalls[f"recall_at_{k}"] = min(top_k_rel, n_gt) / n_gt if n_gt else 0.0

    # Precision@5
    top5 = flags[:5]
    precision_at_5 = sum(top5) / len(top5) if top5 else 0.0

    # NDCG@5
    k5 = 5
    top_k5 = flags[:k5]
    dcg = sum(1.0 / math.log2(r + 2) for r, f in enumerate(top_k5) if f)
    n_ideal = min(sum(1 for f in flags if f), k5)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(n_ideal))
    ndcg_at_5 = dcg / idcg if idcg > 0 else 0.0

    return {
        "mrr": mrr,
        **recalls,
        "precision_at_5": precision_at_5,
        "ndcg_at_5": ndcg_at_5,
        "deepeval_contextual_relevancy": 0.0,
    }


def _compute_point_b_dci(
    approved_chunks: list[dict],
    gt_texts: list[str],
    k_values: list[int],
) -> dict:
    """
    Métricas post-curación para el Punto B del pipeline DCI.

    Reutiliza la misma lógica de containment recall que Point A, aplicada
    sobre los chunks aprobados por el oráculo SAIL-RAG.

    Args:
        approved_chunks: Chunks aprobados por el oráculo.
        gt_texts: Textos del ground truth de PeerQA.
        k_values: Valores de k para Recall@k.
    Returns:
        Dict con mrr, recall_at_5, coverage, ndcg_at_5.
    """
    base = _compute_point_a_dci(approved_chunks, gt_texts, k_values)
    n_gt = len(gt_texts)
    covered = sum(
        1 for gt in gt_texts
        if any(_dci_chunk_relevant(c.get("text", ""), [gt]) for c in approved_chunks)
    )
    coverage = covered / n_gt if n_gt else 0.0
    return {
        "mrr": base["mrr"],
        "recall_at_5": base.get("recall_at_5", 0.0),
        "coverage": coverage,
        "ndcg_at_5": base["ndcg_at_5"],
        "ragas_context_precision": 0.0,
        "ragas_context_recall": 0.0,
        "deepeval_contextual_relevancy": 0.0,
    }


def _stratified_sample(qa_pairs: list[dict], size: Optional[int], seed: int) -> list[dict]:
    """Muestreo estratificado por longitud de query (mismo que RAG 2)."""
    if not qa_pairs or size is None or size >= len(qa_pairs):
        return list(qa_pairs)
    sorted_pairs = sorted(qa_pairs, key=lambda q: len(q["query"]))
    n = len(sorted_pairs)
    t1, t2, t3 = sorted_pairs[:n // 3], sorted_pairs[n // 3: 2 * n // 3], sorted_pairs[2 * n // 3:]
    rng = random.Random(seed)
    n_per = size // 3
    rem = size - n_per * 3
    s1 = rng.sample(t1, min(n_per + (1 if rem > 0 else 0), len(t1)))
    s2 = rng.sample(t2, min(n_per + (1 if rem > 1 else 0), len(t2)))
    s3 = rng.sample(t3, min(n_per, len(t3)))
    return s1 + s2 + s3


def _aggregate_dci(per_query: list[dict], k_values: list[int]) -> dict:
    """Agrega métricas DCI sobre todos los resultados por query."""
    n = len(per_query)
    if n == 0:
        return {}

    def mean(key: str, point_key: str) -> float:
        return sum(r.get(point_key, {}).get(key, 0.0) for r in per_query) / n

    def safe_mean(key: str, point_key: str) -> Optional[float]:
        vals = [r.get(point_key, {}).get(key) for r in per_query]
        valid = [v for v in vals if v is not None]
        return sum(valid) / len(valid) if valid else None

    a_keys = ["mrr", "ndcg_at_5", "precision_at_5"] + [f"recall_at_{k}" for k in k_values]
    point_a = {k: mean(k, "point_a_pre_curation") for k in a_keys}

    b_keys = ["mrr", "recall_at_5", "coverage", "ndcg_at_5"]
    point_b = {k: mean(k, "point_b_post_curation") for k in b_keys}

    point_c = {
        "rouge_l": mean("rouge_l", "point_c_post_llm"),
        "bert_score_f1": safe_mean("bert_score_f1", "point_c_post_llm"),
        "faithfulness": safe_mean("faithfulness", "point_c_post_llm"),
        "ragas_faithfulness": 0.0,
        "ragas_answer_relevancy": 0.0,
        "ragas_noise_sensitivity": 0.0,
        "deepeval_hallucination": 0.0,
        "deepeval_answer_relevancy": 0.0,
    }

    return {"point_a": point_a, "point_b": point_b, "point_c": point_c}


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAG3DCIPipeline:
    """
    Pipeline RAG 3: DCI retrieval agéntico + SAIL-RAG curation + Groq generation.

    El retrieval usa DeepSeek-R1 14B via Ollama para buscar iterativamente en el
    corpus de texto plano. El resto del pipeline (curación y generación) es
    idéntico al de RAG 2 para que la comparación sea válida.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Configuración global (config.yaml).
        """
        self.config = config
        self.rag3_cfg = config.get("rag3", {})
        eval_cfg = self.rag3_cfg.get("evaluation", {})

        self.results_dir = Path(config["paths"]["results"]) / "rag3"
        self.agent_traces_dir = self.results_dir / "agent_traces"
        self.checkpoints_dir = Path(config["paths"]["checkpoints"])

        self.metrics_subset_size: Optional[int] = eval_cfg.get("metrics_subset_size", 200)
        self.metrics_subset_seed: int = eval_cfg.get("metrics_subset_seed", 42)
        self.min_approved: int = self.rag3_cfg.get("min_approved_for_generation", 1)
        self.fallback_top_n: int = self.rag3_cfg.get("fallback_top_n", 3)
        self.faith_threshold: float = self.rag3_cfg.get("faithfulness_cosine_threshold", 0.75)
        self.k_values: list[int] = config["evaluation"]["recall_k_values"]

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.agent_traces_dir.mkdir(parents=True, exist_ok=True)

    # ── Modo ingest ───────────────────────────────────────────────────────────

    def ingest(self) -> None:
        """
        Descarga PeerQA y construye el corpus de texto plano para DCI.

        Genera:
          - corpus/{paper_id}/{section}.txt   — texto por sección para el agente
          - checkpoints/rag3_dci_qa_pairs.json — queries y ground truth
          - checkpoints/rag3_dci_corpus_map.json — mapa chunk_id→texto
        """
        logger.info("=== RAG 3 DCI — Modo INGEST ===")

        builder = CorpusBuilder(self.config)
        qa_pairs, corpus_map = builder.build()
        builder.save_checkpoints(qa_pairs, corpus_map, self.checkpoints_dir)

        logger.success(
            "Ingest completado | {} papers en corpus | {} QA pairs",
            sum(1 for p in (self.checkpoints_dir.parent / self.rag3_cfg.get("corpus", {}).get("path", "./corpus")).iterdir() if p.is_dir()),
            len(qa_pairs),
        )

    # ── Modo evaluate ─────────────────────────────────────────────────────────

    def evaluate(self) -> None:
        """
        Corre el pipeline DCI completo sobre el subset de evaluación.

        Flujo por query:
          1. DCIAgent busca en corpus/ (hasta max_iterations iteraciones)
          2. EvidenceCollector normaliza la evidencia al esquema estándar
          3. Punto A: métricas sobre la evidencia del agente (pre-curación)
          4. CurationOracle filtra con ground truth Jaccard (SAIL-RAG)
          5. Punto B: métricas post-curación
          6. GroqLLM genera respuesta
          7. Punto C: ROUGE-L, BERTScore, faithfulness cosine

        Guarda traza JSON por query y métricas agregadas en metrics_dci.json.
        """
        logger.info("=== RAG 3 DCI — Modo EVALUATE ===")

        qa_pairs, corpus_map = self._load_checkpoints()
        eval_qa = _stratified_sample(qa_pairs, self.metrics_subset_size, self.metrics_subset_seed)

        agent = DCIAgent(self.config)
        agent.check_ollama()

        collector = EvidenceCollector(self.config)
        oracle = CurationOracle(self.config)
        groq_keys = _load_groq_keys()
        llm = _RotatingGroqLLM(self.config, groq_keys)

        per_query_results: list[dict] = []

        for i, qa in enumerate(eval_qa):
            query_id = qa["query_id"]
            query = qa["query"]
            gt_ids = qa["ground_truth_evidence"]
            gt_texts = [corpus_map[gid] for gid in gt_ids if gid in corpus_map]

            logger.info("Query {}/{} | id={}", i + 1, len(eval_qa), query_id)

            # ── Paso 1: DCI retrieval agéntico ───────────────────────────────
            agent_result = agent.run_agent(query)
            evidence_raw = agent_result["evidence_raw"]
            evidence_chunks = collector.format_evidence(evidence_raw)

            # ── Punto A: métricas pre-curación ───────────────────────────────
            point_a = _compute_point_a_dci(evidence_chunks, gt_texts, self.k_values)

            # ── Paso 2: Curación SAIL-RAG ─────────────────────────────────────
            curation = oracle.curate(
                query=query,
                reranked_chunks=evidence_chunks,
                ground_truth_ids=gt_ids,
                ground_truth_texts=gt_texts,
            )
            approved = curation["approved_chunks"]

            # ── Punto B: métricas post-curación ──────────────────────────────
            point_b = _compute_point_b_dci(approved, gt_texts, self.k_values)

            # ── Paso 3: Generación Groq ───────────────────────────────────────
            context_chunks = approved if len(approved) >= self.min_approved else evidence_chunks[:self.fallback_top_n]
            used_fallback = len(approved) < self.min_approved
            generated = llm.generate(query, context_chunks)

            # ── Punto C: métricas post-LLM (sin BERTScore — se calcula en batch) ──
            point_c_partial = {
                "rouge_l": compute_rouge_l(generated, qa.get("answer", "")),
                "bert_score_f1": None,
                "faithfulness": None,
                "ragas_faithfulness": 0.0,
                "ragas_answer_relevancy": 0.0,
                "ragas_noise_sensitivity": 0.0,
                "deepeval_hallucination": 0.0,
                "deepeval_answer_relevancy": 0.0,
            }

            query_result = {
                "query_id": query_id,
                "query": query,
                "agent_iterations": agent_result["iterations"],
                "think_blocks": agent_result["think_blocks"],
                "tool_calls": agent_result["tool_calls"],
                "stopped_reason": agent_result["stopped_reason"],
                "evidence_found": evidence_chunks,
                "approved_chunks": approved,
                "ground_truth_evidence": gt_ids,
                "point_a_pre_curation": point_a,
                "point_b_post_curation": point_b,
                "generated_response": generated,
                "used_fallback": used_fallback,
                "curation_trace": {
                    "presented_to_oracle": [c["chunk_id"] for c in evidence_chunks],
                    "approved_by_oracle": [c["chunk_id"] for c in approved],
                    "oracle_decisions": curation["oracle_decisions"],
                },
                "point_c_post_llm": point_c_partial,
            }

            per_query_results.append(query_result)
            self._save_trace(query_result, i)

            logger.info(
                "  A.MRR={:.3f} | B.cov={:.3f} | C.rouge={:.3f} | agent_iters={} | fallback={}",
                point_a.get("mrr", 0),
                point_b.get("coverage", 0),
                point_c_partial["rouge_l"],
                agent_result["iterations"],
                used_fallback,
            )

        # ── Phase 2: BERTScore + Faithfulness en GPU ─────────────────────────
        # Corre después de que el agente DCI terminó todas las queries,
        # así que Ollama ya no está activo y la GPU está disponible.
        logger.info("Phase 2: BERTScore + Faithfulness GPU para {} queries", len(per_query_results))
        preds = [r["generated_response"] for r in per_query_results]
        refs = [eval_qa[j]["answer"] for j in range(len(per_query_results))]
        bs_scores = compute_bertscore_batch(preds, refs)

        from src.retrieval.embedder import BGEEmbedder
        faith_embedder = BGEEmbedder(self.config)
        faith_embedder.load()

        for r, bs in zip(per_query_results, bs_scores):
            faith = compute_faithfulness_cosine(
                r["generated_response"],
                r["approved_chunks"],
                faith_embedder,
                self.faith_threshold,
            )
            r["point_c_post_llm"]["bert_score_f1"] = bs
            r["point_c_post_llm"]["faithfulness"] = faith

        faith_embedder.unload()
        gc.collect()

        # ── Agregar y guardar resultados ──────────────────────────────────────
        aggregate = _aggregate_dci(per_query_results, self.k_values)

        agent_stats = self._compute_agent_stats(per_query_results)

        logger.success(
            "RAG 3 DCI completado | A.MRR={:.4f} | B.cov={:.4f} | C.ROUGE-L={:.4f}",
            aggregate["point_a"]["mrr"],
            aggregate["point_b"]["coverage"],
            aggregate["point_c"]["rouge_l"],
        )

        self._save_results(per_query_results, aggregate, agent_stats, eval_qa, qa_pairs)

    # ── Helpers privados ──────────────────────────────────────────────────────

    def _load_checkpoints(self) -> tuple[list[dict], dict[str, str]]:
        """
        Carga qa_pairs y corpus_map desde checkpoints DCI.

        Raises:
            FileNotFoundError: Si no existen los checkpoints. Mensaje claro
                              con el comando para generarlos.
        """
        qa_path = self.checkpoints_dir / "rag3_dci_qa_pairs.json"
        map_path = self.checkpoints_dir / "rag3_dci_corpus_map.json"

        for p in [qa_path, map_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"No se encontró {p}.\n"
                    "Corre primero: python main.py --pipeline rag3 --mode ingest"
                )

        with open(qa_path, encoding="utf-8") as f:
            qa_pairs: list[dict] = json.load(f)
        with open(map_path, encoding="utf-8") as f:
            corpus_map: dict[str, str] = json.load(f)

        logger.info(
            "Checkpoints DCI cargados | {} QA pairs | {} corpus entries",
            len(qa_pairs), len(corpus_map),
        )
        return qa_pairs, corpus_map

    def _save_trace(self, query_result: dict, idx: int) -> None:
        """Guarda la traza completa de una query en agent_traces/."""
        trace_path = self.agent_traces_dir / f"query_{idx:03d}_trace.json"
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(query_result, f, ensure_ascii=False, indent=2)

    def _compute_agent_stats(self, per_query: list[dict]) -> dict:
        """Estadísticas agregadas del comportamiento del agente."""
        iterations = [r["agent_iterations"] for r in per_query]
        tool_counts: dict[str, int] = {}
        think_lengths: list[int] = []

        for r in per_query:
            for tc in r.get("tool_calls", []):
                t = tc.get("tool", "unknown")
                tool_counts[t] = tool_counts.get(t, 0) + 1
            for tb in r.get("think_blocks", []):
                think_lengths.append(len(tb.get("think", "").split()))

        return {
            "mean_iterations": sum(iterations) / len(iterations) if iterations else 0.0,
            "min_iterations": min(iterations) if iterations else 0,
            "max_iterations": max(iterations) if iterations else 0,
            "tool_usage": tool_counts,
            "mean_think_length_words": sum(think_lengths) / len(think_lengths) if think_lengths else 0.0,
        }

    def _save_results(
        self,
        per_query_results: list[dict],
        aggregate: dict,
        agent_stats: dict,
        eval_qa: list[dict],
        all_qa_pairs: list[dict],
    ) -> None:
        """Guarda métricas agregadas en results/rag3/metrics_dci.json."""
        output = {
            "condition": "rag3_dci_sail",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "config_snapshot": self.config,
            "subset_info": {
                "total_qa_pairs": len(all_qa_pairs),
                "metrics_subset_size": len(eval_qa),
            },
            "aggregate": aggregate,
            "agent_stats": agent_stats,
            "per_query_results": per_query_results,
        }

        out_path = self.results_dir / "metrics_dci.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.success("Resultados RAG 3 DCI guardados en {}", out_path)
