"""
rag3_2_deepseek_docling.py
==========================
Pipeline RAG 3.2: DCI agéntico con corpus Docling + DeepSeek-R1 14B.

Posición en el ablation DCI:
  RAG 3   — DeepSeek-R1 14B + corpus texto plano (PeerQA) + oracle Jaccard
  RAG 3.1 — Granite 3.1 8B  + corpus Docling              + oracle Containment
  RAG 3.2 — DeepSeek-R1 14B + corpus Docling              + oracle Containment  ← este

RAG 3.2 aísla el efecto del corpus (Docling vs texto plano) manteniendo fijo el modelo.
Comparando RAG 3 vs RAG 3.2: diferencia = corpus quality únicamente.
Comparando RAG 3.1 vs RAG 3.2: diferencia = modelo (Granite 8B vs DeepSeek 14B).

Prerrequisitos:
  1. python main.py --pipeline rag3 --mode ingest       (qa_pairs + corpus_map)
  2. python main.py --pipeline rag3_1 --mode ingest     (corpus_docling/)
  3. python main.py --pipeline rag3_2 --mode evaluate
"""

from src.pipelines.rag3_1_granite_docling import RAG31Pipeline


class RAG32Pipeline(RAG31Pipeline):
    """RAG 3.2: igual que RAG 3.1 pero con DeepSeek-R1 14B en lugar de Granite 3.1 8B."""

    GRANITE_MODEL    = "deepseek-r1:14b"
    CONDITION_NAME   = "rag3_2_deepseek_docling"
    LOG_NAME         = "RAG 3.2"

    def __init__(self, config: dict):
        super().__init__(config)
        # Sobreescribir directorios para no pisar resultados de RAG 3.1
        from pathlib import Path
        self.results_dir = Path(config["paths"]["results"]) / "rag3_2"
        self.agent_traces_dir = self.results_dir / "agent_traces"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.agent_traces_dir.mkdir(parents=True, exist_ok=True)
