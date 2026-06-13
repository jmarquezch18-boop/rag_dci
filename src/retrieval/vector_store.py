"""
vector_store.py
===============
Wrapper sobre Qdrant in-memory para indexar y recuperar chunks de PeerQA.

Posición en el pipeline: Paso 4 — almacena y consulta vectores de chunks.
Decisión: Qdrant in-memory evita dependencias de servidor externo y es
suficiente para el tamaño de PeerQA. La colección se reconstruye en cada
run de evaluate desde el checkpoint numpy (Qdrant in-memory no persiste).
"""

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, QueryRequest


class QdrantStore:
    """Qdrant in-memory con colección 'peerqa_chunks' y distancia coseno."""

    COLLECTION = "peerqa_chunks"
    DIM = 1024  # dimensión de BGE-M3 dense embeddings

    def __init__(self, config: dict):
        self.client = QdrantClient(":memory:")
        logger.info("Qdrant iniciado en modo in-memory")

    def create_collection(self) -> None:
        """Crea (o recrea limpia) la colección con distancia coseno."""
        if self.client.collection_exists(self.COLLECTION):
            self.client.delete_collection(self.COLLECTION)
            logger.debug("Colección existente eliminada para recrear limpia")

        self.client.create_collection(
            collection_name=self.COLLECTION,
            vectors_config=VectorParams(size=self.DIM, distance=Distance.COSINE),
        )
        logger.info(
            "Colección '{}' creada | dim={} | distancia=COSINE",
            self.COLLECTION, self.DIM,
        )

    def upsert(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        """
        Inserta chunks con sus embeddings en Qdrant.

        Los embeddings se insertan en batches de 500 para no saturar
        la memoria del proceso aunque el dataset sea grande.

        Args:
            chunks: Lista de dicts con keys chunk_id, paper_id, text.
            embeddings: Array (N, 1024) de embeddings normalizados.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks pero {len(embeddings)} embeddings"
            )

        logger.info("Indexando {} chunks en Qdrant", len(chunks))

        points = [
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={
                    "chunk_id": chunks[i]["chunk_id"],
                    "paper_id": chunks[i]["paper_id"],
                    "text": chunks[i]["text"],
                },
            )
            for i in range(len(chunks))
        ]

        batch_size = 500
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.COLLECTION,
                points=points[start : start + batch_size],
            )

        logger.success("Indexados {} puntos en Qdrant", len(points))

    def search(self, query_vec: np.ndarray, k: int) -> list[dict]:
        """
        Búsqueda ANN por similitud coseno.

        Args:
            query_vec: Vector de la query, shape (1024,).
            k: Número de resultados a retornar.
        Returns:
            Lista de dicts con chunk_id, paper_id, text, score (desc por score).
        """
        results = self.client.query_points(
            collection_name=self.COLLECTION,
            query=query_vec.tolist(),
            limit=k,
        ).points

        return [
            {
                "chunk_id": r.payload["chunk_id"],
                "paper_id": r.payload["paper_id"],
                "text": r.payload["text"],
                "score": float(r.score),
            }
            for r in results
        ]
