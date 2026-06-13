"""
chunker.py
==========
Chunking de texto con ventana deslizante usando el tokenizer de BGE-M3.

Posición en el pipeline: Paso 2 — transforma documentos en fragmentos indexables.
Sin section-awareness (eso es RAG 2): el texto se trata como secuencia plana.

Decisión: usar el tokenizer del modelo de embedding garantiza que cada
chunk nunca exceda el límite real del modelo (512 tokens). Un chunk basado
en caracteres podría silenciosamente truncarse en el embedder.
"""

import os
from loguru import logger
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()


class Chunker:
    """Chunking por ventana deslizante usando el tokenizer de BGE-M3."""

    def __init__(self, config: dict):
        model_id = config["embedding"]["model_id"]
        self.chunk_size = config["chunking"]["chunk_size_tokens"]
        self.chunk_overlap = config["chunking"]["chunk_overlap_tokens"]
        self.stride = self.chunk_size - self.chunk_overlap
        raw_token = os.getenv("HF_TOKEN", "").strip()
        hf_token = raw_token if raw_token else None

        logger.info("Cargando tokenizer | model_id={}", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        logger.info(
            "Chunker listo | chunk_size={} | overlap={} | stride={}",
            self.chunk_size, self.chunk_overlap, self.stride,
        )

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        """
        Chunkea una lista de documentos completos.

        Por qué: el pipeline necesita indexar papers enteros, pero BGE-M3
        solo acepta hasta 512 tokens. La ventana deslizante preserva
        contexto en los bordes de cada chunk gracias al overlap.

        Args:
            documents: Lista de dicts con keys paper_id y text.
        Returns:
            Lista plana de chunks con chunk_id, paper_id, text, token_count.
        """
        all_chunks: list[dict] = []
        for doc in documents:
            chunks = self._chunk_text(doc["text"], doc["paper_id"])
            all_chunks.extend(chunks)

        logger.info(
            "Chunking completo | {} docs → {} chunks",
            len(documents), len(all_chunks),
        )
        return all_chunks

    def _chunk_text(self, text: str, paper_id: str) -> list[dict]:
        """
        Divide un texto en chunks con overlap usando token IDs de BGE-M3.

        Args:
            text: Texto completo del documento.
            paper_id: Identificador del documento padre.
        Returns:
            Lista de dicts con chunk_id, paper_id, text, token_count.
        """
        if not text.strip():
            return []

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
                    "chunk_id": f"{paper_id}_chunk_{chunk_idx}",
                    "paper_id": paper_id,
                    "text": chunk_text,
                    "token_count": len(chunk_ids),
                })
                chunk_idx += 1

            if end == len(token_ids):
                break
            start += self.stride

        return chunks
