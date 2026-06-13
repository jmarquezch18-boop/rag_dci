"""
docling_parser.py
=================
Descarga y parsea PDFs científicos usando Docling, preservando la estructura
retórica (Abstract, Introduction, Methodology, Results, Discussion, etc.).

Posición en el pipeline RAG 2: Paso 1b — reemplaza el corpus pre-chunkeado de
MTEB/PeerQA con una representación consciente de secciones. Docling se elige
sobre PyMuPDF/pdfplumber porque preserva estructura semántica (headings,
listas, tablas) sin post-procesamiento ad hoc.

Los paper_ids se derivan de los corpus IDs de mteb/PeerQA cuyo formato es
"arxiv_id_pN_M" (e.g. "2202.09778_p0_0"). Se construye la URL de ArXiv
automáticamente si no se proporciona una URL explícita.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

from loguru import logger


_ARXIV_URL_TEMPLATE = "https://arxiv.org/pdf/{paper_id}.pdf"
_ARXIV_ID_PATTERN = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?")

# Secciones retóricas estándar que Docling suele etiquetar como headings
_SECTION_KEYWORDS: dict[str, list[str]] = {
    "Abstract": ["abstract"],
    "Introduction": ["introduction", "intro"],
    "Related Work": ["related work", "related works", "background", "literature"],
    "Methodology": ["method", "methodology", "approach", "model", "framework", "proposed"],
    "Results": ["result", "results", "experiment", "experiments", "evaluation", "empirical"],
    "Discussion": ["discussion", "analysis", "ablation"],
    "Conclusion": ["conclusion", "conclusions", "summary", "future work"],
    "Limitations": ["limitation", "limitations"],
    "References": ["references", "bibliography"],
}


def _extract_arxiv_id(corpus_id: str) -> Optional[str]:
    """
    Extrae el ID de ArXiv del formato de corpus de mteb/PeerQA.

    mteb/PeerQA usa IDs como "2202.09778_p0_0". El paper_id es la parte
    antes del primer underscore.

    Args:
        corpus_id: ID de corpus de mteb/PeerQA.
    Returns:
        ID de ArXiv como string, o None si no se puede extraer.
    """
    parts = corpus_id.split("_")
    if parts:
        candidate = parts[0]
        if _ARXIV_ID_PATTERN.match(candidate):
            return candidate
    return None


def _infer_section(heading_text: str) -> str:
    """
    Mapea un texto de heading a una categoría retórica estándar.

    Usa matching de keywords case-insensitive. Si ninguna keyword coincide,
    retorna "Unknown" en lugar de inventar una categoría.

    Args:
        heading_text: Texto del heading detectado por Docling.
    Returns:
        Nombre de sección estándar.
    """
    heading_lower = heading_text.lower().strip()
    for section_name, keywords in _SECTION_KEYWORDS.items():
        if any(kw in heading_lower for kw in keywords):
            return section_name
    return "Unknown"


class DoclingParser:
    """
    Descarga PDFs por URL (o construye URL de ArXiv) y los parsea con Docling.

    Diseñado para procesar los papers de PeerQA: recibe un conjunto de
    paper_ids únicos, descarga cada PDF, y retorna una lista de segmentos
    con su sección retórica asignada.

    El rate limiting entre descargas evita bans de ArXiv (1 req/s).
    """

    def __init__(self, config: dict):
        self.config = config
        self.sleep_between_downloads: float = config.get("rag2", {}).get(
            "download_sleep_s", 1.5
        )
        logger.info("DoclingParser inicializado | sleep_between={}s", self.sleep_between_downloads)

    def parse_paper(self, paper_id: str, paper_url: Optional[str] = None) -> list[dict]:
        """
        Descarga y parsea un paper, retornando sus segmentos con sección asignada.

        Docling acepta URLs directamente; si no se provee URL se construye la
        URL de ArXiv desde el paper_id. Los segmentos retornados aún no están
        chunkeados — eso lo hace section_chunker.py.

        Args:
            paper_id: ID del paper (ArXiv ID u otro identificador único).
            paper_url: URL del PDF. Si es None, se construye desde paper_id.
        Returns:
            Lista de dicts con keys: paper_id, section, text, source_url.
            Lista vacía si el parsing falla.
        Raises:
            No lanza excepciones — errores se loggean y retorna lista vacía.
        """
        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            logger.error("Docling no está instalado. Corre: pip install docling")
            return []

        url = paper_url or _ARXIV_URL_TEMPLATE.format(paper_id=paper_id)
        logger.info("Parseando paper | paper_id={} | url={}", paper_id, url)

        try:
            converter = DocumentConverter()
            result = converter.convert(url)
            doc = result.document
        except Exception as e:
            logger.warning("Docling falló para paper_id={} | error={}", paper_id, str(e))
            return []

        segments = self._extract_segments(doc, paper_id, url)
        logger.debug(
            "paper_id={} | {} segmentos extraídos", paper_id, len(segments)
        )
        return segments

    def parse_papers_batch(
        self,
        paper_ids: list[str],
        paper_url_map: Optional[dict[str, str]] = None,
    ) -> dict[str, list[dict]]:
        """
        Parsea un batch de papers con rate limiting entre descargas.

        Args:
            paper_ids: Lista de IDs únicos a parsear.
            paper_url_map: Mapa opcional paper_id → URL. Si no se provee,
                           se construyen URLs de ArXiv automáticamente.
        Returns:
            Dict paper_id → lista de segmentos (vacía si el paper falló).
        """
        url_map = paper_url_map or {}
        results: dict[str, list[dict]] = {}
        total = len(paper_ids)

        for i, pid in enumerate(paper_ids):
            logger.info("Procesando paper {}/{} | paper_id={}", i + 1, total, pid)
            segments = self.parse_paper(pid, url_map.get(pid))
            results[pid] = segments

            if i < total - 1:
                time.sleep(self.sleep_between_downloads)

        success = sum(1 for v in results.values() if v)
        logger.success(
            "Batch completado | {}/{} papers parseados exitosamente",
            success, total,
        )
        return results

    def _extract_segments(self, doc, paper_id: str, source_url: str) -> list[dict]:
        """
        Recorre el DoclingDocument y agrupa texto por sección retórica.

        Estrategia: itera los items del documento en orden. Cada vez que
        aparece un heading, actualiza la sección activa. El texto acumulado
        entre headings se emite como un segmento de esa sección.

        Args:
            doc: Objeto DoclingDocument retornado por Docling.
            paper_id: ID del paper para metadata.
            source_url: URL de origen para metadata.
        Returns:
            Lista de dicts con paper_id, section, text, source_url.
        """
        segments: list[dict] = []
        current_section = "Unknown"
        current_texts: list[str] = []

        def flush(section: str) -> None:
            joined = " ".join(current_texts).strip()
            if joined:
                segments.append({
                    "paper_id": paper_id,
                    "section": section,
                    "text": joined,
                    "source_url": source_url,
                })
            current_texts.clear()

        try:
            # Docling expone el documento como una secuencia de elementos
            # con tipo (heading, paragraph, list_item, table, etc.)
            for item, _ in doc.iterate_items():
                item_type = type(item).__name__.lower()

                if "heading" in item_type or "section" in item_type:
                    heading_text = getattr(item, "text", "") or ""
                    if heading_text.strip():
                        flush(current_section)
                        current_section = _infer_section(heading_text)
                elif "text" in item_type or "paragraph" in item_type:
                    text = getattr(item, "text", "") or ""
                    if text.strip():
                        current_texts.append(text.strip())

            flush(current_section)

        except Exception as e:
            # Fallback: exportar el documento completo como texto plano
            logger.warning(
                "iterate_items falló para paper_id={}, fallback a export_to_markdown | {}",
                paper_id, str(e),
            )
            try:
                full_text = doc.export_to_markdown()
                if full_text.strip():
                    segments.append({
                        "paper_id": paper_id,
                        "section": "Unknown",
                        "text": full_text.strip(),
                        "source_url": source_url,
                    })
            except Exception as e2:
                logger.error("Fallback también falló para paper_id={} | {}", paper_id, str(e2))

        return segments


def extract_unique_paper_ids(corpus: list[dict]) -> list[str]:
    """
    Extrae paper IDs únicos de los corpus entries de mteb/PeerQA.

    Los IDs de corpus tienen formato "arxiv_id_pN_M" — el paper_id es la
    parte antes del primer underscore.

    Args:
        corpus: Lista de corpus entries con campo 'chunk_id'.
    Returns:
        Lista deduplicada de paper IDs.
    """
    seen: set[str] = set()
    ids: list[str] = []
    for entry in corpus:
        pid = _extract_arxiv_id(entry.get("chunk_id", ""))
        if pid and pid not in seen:
            seen.add(pid)
            ids.append(pid)
    logger.info("Paper IDs únicos extraídos del corpus: {}", len(ids))
    return ids
