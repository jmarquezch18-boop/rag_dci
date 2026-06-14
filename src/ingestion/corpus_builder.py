"""
corpus_builder.py
=================
Descarga y organiza el corpus de papers en texto plano para el agente DCI.

Posición en el pipeline RAG 3: Paso 1 — genera la estructura de archivos .txt
sobre la que el agente DeepSeek-R1 hará búsquedas lexicales (grep, read_file).

Por qué texto plano: DCI requiere un corpus navegable por herramientas de shell
estándar (grep, cat). Los archivos .txt por sección permiten que el agente
filtre por tipo de contenido (e.g., buscar solo en "results.txt") y que los
fragmentos sean comparables con los pasajes del ground truth de PeerQA.

Fuente de datos: se usa el corpus de mteb/PeerQA directamente desde HuggingFace
(la misma fuente que RAG 1 y RAG 2) para garantizar que el texto sea idéntico
y la comparación entre RAGs sea válida. El campo `title` de cada pasaje se
mapea a una sección retórica estándar vía _infer_section() de docling_parser.

Estructura de salida:
  corpus/
    {paper_id}/
      abstract.txt
      introduction.txt
      methodology.txt
      results.txt
      discussion.txt
      limitations.txt

Los paper_id son saneados (/ → _) para uso seguro como nombre de directorio.
Si una sección no tiene pasajes, no se crea el archivo (sin archivos vacíos).
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

from loguru import logger


def _sanitize_paper_id(paper_id: str) -> str:
    """
    Convierte un paper_id con caracteres problemáticos a nombre de directorio seguro.

    Los IDs de PeerQA tienen formato 'nlpeer/F1000-22/10-170', que no puede
    usarse directamente como nombre de directorio en Linux.

    Args:
        paper_id: ID original del paper.
    Returns:
        ID saneado con / reemplazado por _.
    """
    return re.sub(r"[/\\:*?\"<>|]", "_", paper_id)


class CorpusBuilder:
    """
    Construye el corpus de texto plano a partir del dataset mteb/PeerQA.

    Reutiliza la misma fuente de datos que RAG 1 y RAG 2 para garantizar
    comparabilidad. Agrupa pasajes por paper_id y sección, y los escribe
    como archivos .txt separados en corpus/{paper_id}/{seccion}.txt.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Configuración global del proyecto (config.yaml).
        """
        self.config = config
        rag3_cfg = config.get("rag3", {})
        corpus_cfg = rag3_cfg.get("corpus", {})

        corpus_path = corpus_cfg.get("path", "./corpus")
        self.corpus_dir = Path(corpus_path)
        self.valid_sections: set[str] = {
            s.lower() for s in corpus_cfg.get(
                "sections",
                ["abstract", "introduction", "methodology", "results", "discussion", "limitations"],
            )
        }

        self.hf_token: str | None = os.getenv("HF_TOKEN") or None
        logger.info(
            "CorpusBuilder inicializado | corpus_dir={} | sections={}",
            self.corpus_dir, sorted(self.valid_sections),
        )

    def build(self) -> tuple[list[dict], dict[str, str]]:
        """
        Descarga PeerQA y construye el corpus de texto plano en corpus/.

        Carga el dataset desde HuggingFace, agrupa los pasajes por paper_id y
        sección, y escribe cada grupo como un archivo .txt. Retorna los qa_pairs
        y el mapa chunk_id→text para que el pipeline los guarde como checkpoints.

        Returns:
            Tupla (qa_pairs, corpus_map) donde:
            - qa_pairs: lista de dicts con query_id, query, answer, ground_truth_evidence
            - corpus_map: dict {chunk_id: text} para lookup de textos de ground truth
        Raises:
            ImportError: si `datasets` no está instalado.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets no está instalado. Corre: pip install datasets")

        try:
            from src.ingestion.docling_parser import _infer_section
        except ImportError:
            def _infer_section(heading: str) -> str:
                return heading.capitalize() if heading else "Unknown"

        logger.info("Cargando mteb/PeerQA desde HuggingFace para corpus DCI...")

        corpus_ds = load_dataset(
            "mteb/PeerQA", "PeerQA-corpus", split="test", token=self.hf_token
        )
        queries_ds = load_dataset(
            "mteb/PeerQA", "PeerQA-queries", split="test", token=self.hf_token
        )
        qrels_ds = load_dataset(
            "mteb/PeerQA", "PeerQA-qrels", split="test", token=self.hf_token
        )

        logger.info(
            "Dataset cargado | corpus={} | queries={} | qrels={}",
            len(corpus_ds), len(queries_ds), len(qrels_ds),
        )

        # ── Construir corpus_map y agrupar por paper + sección ────────────────
        corpus_map: dict[str, str] = {}
        # {safe_paper_id → {section_lower → [text, ...]}}
        paper_sections: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

        for row in corpus_ds:
            text = (row.get("text") or "").strip()
            if not text:
                continue

            chunk_id = row["id"]
            corpus_map[chunk_id] = text

            # Derivar paper_id y sección
            raw_paper_id = self._extract_paper_id(chunk_id)
            safe_pid = _sanitize_paper_id(raw_paper_id)

            raw_title = (row.get("title") or "").strip()
            section = _infer_section(raw_title).lower() if raw_title else "unknown"

            paper_sections[safe_pid][section].append(text)

        logger.info(
            "Corpus procesado | {} papers únicos | {} chunks totales",
            len(paper_sections), len(corpus_map),
        )

        # ── Escribir archivos .txt ────────────────────────────────────────────
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        files_written = 0
        papers_processed = 0

        for safe_pid, sections in paper_sections.items():
            paper_dir = self.corpus_dir / safe_pid
            paper_dir.mkdir(parents=True, exist_ok=True)
            paper_files = []

            for section, texts in sections.items():
                if section not in self.valid_sections:
                    continue
                content = "\n\n".join(texts).strip()
                if not content:
                    continue
                file_path = paper_dir / f"{section}.txt"
                file_path.write_text(content, encoding="utf-8")
                paper_files.append(section)
                files_written += 1

            if paper_files:
                papers_processed += 1
                logger.debug("Paper {} | secciones: {}", safe_pid, paper_files)

        logger.success(
            "Corpus construido | {} papers | {} archivos .txt en {}",
            papers_processed, files_written, self.corpus_dir,
        )

        # ── Construir qa_pairs ────────────────────────────────────────────────
        qrels_map: dict[str, list[str]] = defaultdict(list)
        for row in qrels_ds:
            qrels_map[row["query-id"]].append(row["corpus-id"])

        qa_pairs: list[dict] = []
        for row in queries_ds:
            qid = row["id"]
            relevant_ids = qrels_map.get(qid, [])
            reference = " ".join(
                corpus_map[cid] for cid in relevant_ids if cid in corpus_map
            )
            qa_pairs.append({
                "query_id": qid,
                "query": row["text"],
                "answer": reference,
                "ground_truth_evidence": relevant_ids,
            })

        logger.info("QA pairs construidos: {}", len(qa_pairs))
        return qa_pairs, corpus_map

    @staticmethod
    def _extract_paper_id(chunk_id: str) -> str:
        """
        Extrae el paper_id del chunk_id de PeerQA.

        Para IDs con formato '{source}/{paper}_{idx}' (e.g. 'nlpeer/F1000-22/10-170_5'),
        elimina el último sufijo numérico. Para otros formatos, retorna el chunk_id.

        Args:
            chunk_id: ID del corpus de PeerQA.
        Returns:
            paper_id sin el sufijo de índice.
        """
        parts = chunk_id.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return chunk_id

    @staticmethod
    def sanitize_paper_id(paper_id: str) -> str:
        """Expuesto como método estático para uso externo."""
        return _sanitize_paper_id(paper_id)

    def save_checkpoints(
        self,
        qa_pairs: list[dict],
        corpus_map: dict[str, str],
        checkpoints_dir: Path,
    ) -> None:
        """
        Guarda qa_pairs y corpus_map en archivos JSON para el modo evaluate.

        Args:
            qa_pairs: Lista de QA pairs.
            corpus_map: Mapa chunk_id → text.
            checkpoints_dir: Directorio de checkpoints del proyecto.
        """
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        qa_path = checkpoints_dir / "rag3_dci_qa_pairs.json"
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        logger.info("QA pairs guardados: {}", qa_path)

        map_path = checkpoints_dir / "rag3_dci_corpus_map.json"
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(corpus_map, f, ensure_ascii=False)
        logger.info(
            "Corpus map guardado: {} | {} entradas | {:.1f} MB",
            map_path, len(corpus_map),
            map_path.stat().st_size / 1e6,
        )
