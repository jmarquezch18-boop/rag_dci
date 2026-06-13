"""
section_chunker.py
==================
Chunking consciente de secciones sobre los segmentos de Docling.

Posición en el pipeline RAG 2: Paso 2b — recibe segmentos con sección
asignada (output de DoclingParser) y los divide en chunks indexables
preservando la etiqueta de sección como metadato en Qdrant.

Diferencia clave con chunker.py (RAG 1): cada chunk hereda la sección
retórica de su segmento padre. Un chunk "Results" y uno "Introduction"
del mismo paper quedan separados en Qdrant y son filtrables por metadato.
El tokenizer usado es el mismo de BGE-M3 para garantizar coherencia con
el modelo de embedding.
"""

from __future__ import annotations

import os
from loguru import logger
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()


class SectionChunker:
    """
    Chunking con ventana deslizante que preserva la etiqueta de sección.

    Usa el tokenizer de BGE-M3 para garantizar que ningún chunk supere
    el límite real del modelo de embedding (512 tokens).
    """

    def __init__(self, config: dict):
        model_id = config["embedding"]["model_id"]
        rag2_cfg = config.get("rag2", {})
        self.chunk_size = rag2_cfg.get(
            "section_chunk_size_tokens",
            config["chunking"]["chunk_size_tokens"],
        )
        self.chunk_overlap = rag2_cfg.get(
            "section_chunk_overlap_tokens",
            config["chunking"]["chunk_overlap_tokens"],
        )
        self.stride = self.chunk_size - self.chunk_overlap

        raw_token = os.getenv("HF_TOKEN", "").strip()
        hf_token = raw_token if raw_token else None

        logger.info("Cargando tokenizer para SectionChunker | model_id={}", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        logger.info(
            "SectionChunker listo | chunk_size={} | overlap={} | stride={}",
            self.chunk_size, self.chunk_overlap, self.stride,
        )

    def chunk_segments(self, segments: list[dict]) -> list[dict]:
        """
        Chunkea una lista de segmentos de Docling con sección asignada.

        Cada chunk resultante tiene la misma sección que su segmento padre.
        El chunk_id es único y trazable al paper_id y al índice.

        Args:
            segments: Lista de dicts con paper_id, section, text, source_url.
        Returns:
            Lista de chunks con chunk_id, paper_id, section, text,
            token_count, source_url.
        """
        all_chunks: list[dict] = []
        for seg in segments:
            chunks = self._chunk_segment(seg)
            all_chunks.extend(chunks)

        logger.info(
            "SectionChunker completado | {} segmentos → {} chunks",
            len(segments), len(all_chunks),
        )
        return all_chunks

    def chunk_papers(self, paper_segments: dict[str, list[dict]]) -> list[dict]:
        """
        Chunkea un dict de paper_id → segmentos (output de DoclingParser.parse_papers_batch).

        Shortcut para procesar el output directo del parser batch.

        Args:
            paper_segments: Dict paper_id → lista de segmentos.
        Returns:
            Lista plana de chunks de todos los papers.
        """
        all_segments = [
            seg for segs in paper_segments.values() for seg in segs
        ]
        logger.info(
            "Chunkeando {} segmentos de {} papers",
            len(all_segments), len(paper_segments),
        )
        return self.chunk_segments(all_segments)

    def _chunk_segment(self, segment: dict) -> list[dict]:
        """
        Divide un segmento individual en chunks con ventana deslizante.

        Los chunks heredan paper_id, section y source_url del segmento.
        El chunk_id tiene formato: {paper_id}_{section_slug}_{idx}.

        Args:
            segment: Dict con paper_id, section, text, source_url.
        Returns:
            Lista de chunks con metadatos completos.
        """
        text = segment.get("text", "").strip()
        if not text:
            return []

        paper_id = segment["paper_id"]
        section = segment.get("section", "Unknown")
        source_url = segment.get("source_url", "")
        section_slug = section.lower().replace(" ", "_")

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunks: list[dict] = []
        chunk_idx = 0
        start = 0

        while start < len(token_ids):
            end = min(start + self.chunk_size, len(token_ids))
            chunk_ids = token_ids[start:end]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": f"{paper_id}_{section_slug}_{chunk_idx}",
                    "paper_id": paper_id,
                    "section": section,
                    "text": chunk_text,
                    "token_count": len(chunk_ids),
                    "source_url": source_url,
                })
                chunk_idx += 1

            if end == len(token_ids):
                break
            start += self.stride

        return chunks
