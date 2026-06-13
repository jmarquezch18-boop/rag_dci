"""
embedder.py
===========
Genera embeddings densos con BGE-M3 (BAAI/bge-m3) via sentence-transformers.

Posición en el pipeline: Paso 3 — transforma texto en vectores para Qdrant.
Decisión: sentence-transformers sobre FlagEmbedding por compatibilidad con
Python 3.13. BGE-M3 produce vectores densos de 1024 dimensiones.

Patrón de VRAM: load/unload son métodos explícitos (no __init__/__del__)
para que el pipeline controle exactamente cuándo el modelo ocupa GPU.
Este patrón se reutiliza en RAG 2/3 donde coexisten embedder + reranker.
"""

from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from typing import Generator, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger
from sentence_transformers import SentenceTransformer

load_dotenv()


@contextmanager
def gpu_model(model: object) -> Generator:
    """
    Context manager reutilizable para liberar VRAM al salir del bloque.

    Uso:
        with gpu_model(my_model) as m:
            embeddings = m.encode(texts)
        # VRAM liberada aquí automáticamente
    """
    try:
        yield model
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.debug("VRAM liberada")


class BGEEmbedder:
    """
    Wrapper sobre SentenceTransformer para BGE-M3 con manejo explícito de VRAM.

    Los métodos load() y unload() están separados de __init__ para que
    el pipeline decida cuándo el modelo ocupa memoria. Esto permite el
    patrón 'un modelo a la vez en GPU' sin acoplamiento en el constructor.
    """

    def __init__(self, config: dict):
        self.model_id: str = config["embedding"]["model_id"]
        self.normalize: bool = config["embedding"]["normalize"]
        self.fp16: bool = config["hardware"]["fp16"]
        self.batch_size: int = config["hardware"]["batch_size_embed"]
        raw_token = os.getenv("HF_TOKEN", "").strip()
        self.hf_token: Optional[str] = raw_token if raw_token else None

        hw = config["hardware"]
        self.device = "cuda" if (hw["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
        self._model: Optional[SentenceTransformer] = None

    def load(self) -> None:
        """
        Carga BGE-M3 en VRAM (o RAM si no hay CUDA). Debe llamarse antes de embed_*.

        Loggea estado de VRAM antes de cargar para detectar fugas de memoria
        de runs anteriores.
        """
        if torch.cuda.is_available():
            logger.info("GPU: {}", torch.cuda.get_device_name(0))
            logger.info(
                "VRAM total: {:.1f} GB",
                torch.cuda.get_device_properties(0).total_memory / 1e9,
            )
            logger.info(
                "VRAM disponible: {:.1f} GB",
                torch.cuda.memory_reserved(0) / 1e9,
            )
        else:
            logger.warning("CUDA no disponible, usando CPU")

        logger.info("Cargando BGE-M3 | device={} | fp16={}", self.device, self.fp16)

        model_kwargs: dict = {}
        if self.fp16 and self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        self._model = SentenceTransformer(
            self.model_id,
            device=self.device,
            token=self.hf_token,
            model_kwargs=model_kwargs if model_kwargs else None,
        )

        dim = self._model.get_sentence_embedding_dimension()
        logger.success("BGE-M3 cargado | embedding_dim={} | device={}", dim, self.device)

    def unload(self) -> None:
        """Descarga el modelo y libera VRAM. Seguro de llamar aunque no esté cargado."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.debug("BGE-M3 descargado, VRAM liberada")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Convierte una query en texto a vector denso con BGE-M3.

        Por qué: RAG requiere comparar queries con fragmentos en el mismo
        espacio vectorial. BGE-M3 fue elegido por su rendimiento en MTEB
        y soporte multilingüe.

        Args:
            query: Texto de la pregunta del usuario.
        Returns:
            Vector numpy de dimensión 1024 normalizado a norma unitaria.
        Raises:
            RuntimeError: Si el modelo no está cargado.
        """
        if self._model is None:
            raise RuntimeError("Llama a load() antes de embed_query()")
        return self.embed_batch([query])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Genera embeddings para una lista de textos con retry automático en OOM.

        Cada OOM divide el batch a la mitad y reintenta. Si batch=1 sigue
        fallando, el error se propaga (el modelo no cabe en la GPU disponible).

        Args:
            texts: Lista de textos a embeder.
        Returns:
            Array numpy de forma (len(texts), 1024) normalizado si normalize=True.
        Raises:
            RuntimeError: Si el modelo no está cargado.
        """
        if self._model is None:
            raise RuntimeError("Llama a load() antes de embed_batch()")
        return self._embed_with_retry(texts, self.batch_size)

    def _embed_with_retry(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Embede con reducción automática de batch en caso de OOM."""
        try:
            return self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=len(texts) > 50,
                convert_to_numpy=True,
            )
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 1:
                raise
            new_batch = batch_size // 2
            logger.warning("Batch OOM, reduciendo de {} a {}", batch_size, new_batch)
            torch.cuda.empty_cache()
            return self._embed_with_retry(texts, new_batch)
