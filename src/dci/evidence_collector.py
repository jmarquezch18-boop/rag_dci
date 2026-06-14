"""
evidence_collector.py
=====================
Convierte la evidencia acumulada por el agente DCI al formato estándar de chunk
que usa el oráculo SAIL-RAG (oracle.py) sin cambios.

Por qué este módulo: el agente DCI acumula evidencia como resultados de
read_file/read_lines (texto crudo con metadatos de herramienta). El oráculo
espera una lista de dicts con "text", "chunk_id", "section", etc. — el mismo
esquema que usan RAG 1 y RAG 2 desde Qdrant. Este módulo hace la traducción.

El score se fija en 1.0 para todos los fragmentos DCI porque no existe un
score de relevancia numérico en la búsqueda lexical — todos los fragmentos
que el agente seleccionó deliberadamente son igualmente candidatos.

Para textos muy largos (read_file sobre una sección completa), se realiza
chunking en ventanas de 300 palabras para que la similitud Jaccard del oráculo
funcione sobre fragmentos comparables con los pasajes del ground truth.
"""

from __future__ import annotations

from loguru import logger

_MAX_CHUNK_WORDS = 300
_CHUNK_OVERLAP_WORDS = 50


def _split_text(text: str, max_words: int = _MAX_CHUNK_WORDS, overlap: int = _CHUNK_OVERLAP_WORDS) -> list[str]:
    """
    Divide un texto largo en ventanas de palabras con solapamiento.

    Necesario para read_file (secciones completas) donde el texto es demasiado
    largo para que Jaccard funcione contra pasajes cortos de ground truth.

    Args:
        text: Texto a dividir.
        max_words: Máximo de palabras por chunk.
        overlap: Palabras de solapamiento entre chunks consecutivos.
    Returns:
        Lista de sub-textos. Si el texto es corto, retorna [text].
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i: i + max_words]
        chunks.append(" ".join(chunk_words))
        i += max_words - overlap
    return chunks


class EvidenceCollector:
    """
    Formatea evidencia DCI al esquema estándar de chunk para el oráculo.

    Recibe la lista de evidence_raw del DCIAgent y retorna una lista de dicts
    con el mismo esquema que QdrantStoreRAG2 entrega al oráculo en RAG 2.
    Así oracle.py funciona sin ningún cambio en el pipeline DCI.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Configuración global del proyecto.
        """
        agent_cfg = config.get("rag3", {}).get("agent", {})
        self.max_chunks: int = agent_cfg.get("max_evidence_chunks", 10)

    def format_evidence(self, evidence_raw: list[dict]) -> list[dict]:
        """
        Convierte evidence_raw del agente al formato estándar de chunk.

        Para cada entrada de evidence_raw:
        - Si proviene de read_lines (texto corto), se emite como un chunk único.
        - Si proviene de read_file (texto largo), se divide en sub-chunks de
          ~300 palabras para que el oráculo Jaccard pueda hacer matching.

        Args:
            evidence_raw: Lista de {"tool", "args", "text"} del DCIAgent.
        Returns:
            Lista de dicts con: text, section, paper_id, chunk_id, source_url, score.
            Limitada a max_evidence_chunks del config.
        """
        formatted: list[dict] = []
        chunk_counter = 0

        for entry in evidence_raw:
            if len(formatted) >= self.max_chunks:
                break

            args = entry.get("args", {})
            raw_text = entry.get("text", "").strip()
            tool = entry.get("tool", "")

            paper_id = args.get("paper_id", "unknown")
            section = args.get("section", "unknown")

            # Limpiar el header que read_file/read_lines agrega al texto
            if raw_text.startswith("[") and "]\n" in raw_text:
                _, raw_text = raw_text.split("]\n", 1)
                raw_text = raw_text.strip()

            if not raw_text:
                continue

            # read_file → potencialmente largo → dividir
            # read_lines → ya es una ventana pequeña → emitir directo
            sub_texts = _split_text(raw_text) if tool == "read_file" else [raw_text]

            for sub_i, sub_text in enumerate(sub_texts):
                if len(formatted) >= self.max_chunks:
                    break

                chunk_id = f"dci_{paper_id}_{section}_{chunk_counter:04d}"
                chunk_counter += 1

                formatted.append({
                    "text": sub_text,
                    "section": section,
                    "paper_id": paper_id,
                    "chunk_id": chunk_id,
                    "source_url": f"https://arxiv.org/abs/{paper_id}",
                    "score": 1.0,
                })

        logger.debug(
            "EvidenceCollector | raw={} → formatted={} chunks",
            len(evidence_raw), len(formatted),
        )
        return formatted
