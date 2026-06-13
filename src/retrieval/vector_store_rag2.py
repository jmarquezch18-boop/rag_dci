"""
vector_store_rag2.py
====================
Colección Qdrant separada para RAG 2 con soporte de metadatos de sección.

Posición en el pipeline RAG 2: Paso 4 — almacena chunks con sección como
metadato y permite búsqueda ANN filtrada por sección. Se usa una colección
distinta ("peerqa_chunks_rag2") para no interferir con el índice RAG 1.

La colección de RAG 1 ("peerqa_chunks") queda intacta. Esta clase NO hereda
de QdrantStore para evitar cualquier acoplamiento con RAG 1.

Filtrado por sección: Qdrant soporta filtros sobre el payload de los puntos.
Un filtro de sección reduce el espacio de búsqueda ANN y mejora la precisión
cuando el usuario conoce en qué parte del paper está la evidencia.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


class QdrantStoreRAG2:
    """
    Qdrant in-memory para RAG 2 con colección section-aware.

    La colección almacena chunk_id, paper_id, section, text y source_url
    como payload. El método search_with_filter() acepta un filtro de sección
    opcional; si es None, la búsqueda es idéntica a la de RAG 1.
    """

    DIM = 1024  # dimensión de BGE-M3

    def __init__(self, config: dict):
        rag2_cfg = config.get("rag2", {})
        self.collection = rag2_cfg.get("qdrant_collection", "peerqa_chunks_rag2")
        self.client = QdrantClient(":memory:")
        logger.info(
            "QdrantStoreRAG2 iniciado | colección='{}' | in-memory",
            self.collection,
        )

    def create_collection(self) -> None:
        """Crea (o recrea) la colección RAG 2 con distancia coseno."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
            logger.debug("Colección '{}' existente eliminada para recrear", self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.DIM, distance=Distance.COSINE),
        )
        logger.info(
            "Colección '{}' creada | dim={} | distancia=COSINE",
            self.collection, self.DIM,
        )

    def upsert(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        """
        Inserta chunks section-aware con sus embeddings.

        El payload incluye section para que los filtros de búsqueda
        puedan restringir el espacio ANN a una sección específica.

        Args:
            chunks: Lista de dicts con chunk_id, paper_id, section, text, source_url.
            embeddings: Array (N, 1024) de embeddings normalizados.
        Raises:
            ValueError: Si la longitud de chunks y embeddings no coincide.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks pero {len(embeddings)} embeddings"
            )

        logger.info("Indexando {} chunks en QdrantStoreRAG2", len(chunks))

        points = [
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={
                    "chunk_id": chunks[i]["chunk_id"],
                    "paper_id": chunks[i]["paper_id"],
                    "section": chunks[i].get("section", "Unknown"),
                    "text": chunks[i]["text"],
                    "source_url": chunks[i].get("source_url", ""),
                },
            )
            for i in range(len(chunks))
        ]

        batch_size = 500
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=points[start : start + batch_size],
            )

        logger.success("Indexados {} puntos en QdrantStoreRAG2", len(points))

    def search(self, query_vec: np.ndarray, k: int) -> list[dict]:
        """
        Búsqueda ANN sin filtro de sección. Equivalente a RAG 1.

        Args:
            query_vec: Vector de query, shape (1024,).
            k: Número de resultados.
        Returns:
            Lista de dicts con chunk_id, paper_id, section, text, source_url, score.
        """
        return self._query(query_vec, k, section_filter=None)

    def search_with_filter(
        self, query_vec: np.ndarray, k: int, section: str | None = None
    ) -> list[dict]:
        """
        Búsqueda ANN con filtro opcional de sección.

        Si section es None, la búsqueda es idéntica a search(). Si se provee
        una sección (e.g. "Results"), Qdrant solo busca entre chunks de esa
        sección, mejorando precisión para queries dirigidas.

        Args:
            query_vec: Vector de query, shape (1024,).
            k: Número de resultados.
            section: Nombre de sección para filtrar, o None para sin filtro.
        Returns:
            Lista de dicts con chunk_id, paper_id, section, text, source_url, score.
        """
        return self._query(query_vec, k, section_filter=section)

    def _query(
        self, query_vec: np.ndarray, k: int, section_filter: str | None
    ) -> list[dict]:
        """Implementa la consulta con o sin filtro de sección."""
        qdrant_filter = None
        if section_filter:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="section",
                        match=MatchValue(value=section_filter),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vec.tolist(),
            limit=k,
            query_filter=qdrant_filter,
        ).points

        return [
            {
                "chunk_id": r.payload["chunk_id"],
                "paper_id": r.payload["paper_id"],
                "section": r.payload.get("section", "Unknown"),
                "text": r.payload["text"],
                "source_url": r.payload.get("source_url", ""),
                "score": float(r.score),
            }
            for r in results
        ]
