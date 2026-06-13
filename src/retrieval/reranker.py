"""
reranker.py
===========
BGE-reranker-v2-m3 (cross-encoder) para reranking de chunks candidatos.

Posición en el pipeline RAG 2: Paso 3 — recibe los k=20 chunks del ANN
search y los reordena por relevancia con un cross-encoder. El cross-encoder
puntúa el par (query, chunk) de forma conjunta, lo que captura interacciones
semánticas que el bi-encoder de BGE-M3 no puede.

Patrón de VRAM: mismo load/unload explícito que BGEEmbedder. En RTX 3050
4GB no pueden coexistir embedder + reranker en GPU. El pipeline carga uno,
descarga, carga el otro.

Fine-tuning: el método fine_tune_contrastive recibe tripletas positivas/
negativas y actualiza los pesos in-place. El checkpoint se guarda en
checkpoints/reranker/ para el modo evaluate_rn.
"""

from __future__ import annotations

import gc
import os
from typing import Optional

import torch
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class BGEReranker:
    """
    Wrapper sobre BGE-reranker-v2-m3 con load/unload explícito de VRAM.

    Expone rerank() para inferencia y fine_tune_contrastive() para el
    bucle de aprendizaje activo. El estado del modelo (pesos fine-tuneados)
    se persiste via save_checkpoint() / load_checkpoint().
    """

    def __init__(self, config: dict):
        rag2_cfg = config.get("rag2", {})
        self.model_id: str = rag2_cfg.get("reranker_model_id", "BAAI/bge-reranker-v2-m3")
        hw = config["hardware"]
        self.device = "cuda" if (hw["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
        self.fp16: bool = hw.get("fp16", True)
        self.batch_size: int = hw.get("batch_size_rerank", 4)

        raw_token = os.getenv("HF_TOKEN", "").strip()
        self.hf_token: Optional[str] = raw_token if raw_token else None

        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        """
        Carga BGE-reranker-v2-m3 en VRAM (o CPU si no hay CUDA).

        Loggea estado de VRAM para detectar fugas antes de cargar. Usa
        AutoModelForSequenceClassification porque el reranker es un
        clasificador binario relevante/no-relevante.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if torch.cuda.is_available():
            logger.info(
                "VRAM antes de reranker: reservada={:.2f}GB",
                torch.cuda.memory_reserved(0) / 1e9,
            )

        logger.info(
            "Cargando BGE-reranker-v2-m3 | device={} | fp16={}",
            self.device, self.fp16,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=self.hf_token
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, token=self.hf_token
        )

        if self.fp16 and self.device == "cuda":
            self._model = self._model.half()

        self._model = self._model.to(self.device)
        self._model.eval()
        logger.success("BGE-reranker cargado | device={}", self.device)

    def unload(self) -> None:
        """Descarga el modelo y libera VRAM. Seguro de llamar sin modelo cargado."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.debug("BGE-reranker descargado, VRAM liberada")

    def rerank(self, query: str, chunks: list[dict], top_n: int) -> list[dict]:
        """
        Reordena chunks por relevancia al query usando el cross-encoder.

        Puntúa cada par (query, chunk_text) con el reranker. El score
        es el logit de la clase 'relevante' (índice 0 para rerankers BGE).
        Los chunks retornados incluyen el campo 'rerank_score'.

        Args:
            query: Texto de la query.
            chunks: Chunks candidatos del ANN search, cada uno con 'text'.
            top_n: Número de chunks a retornar después del reranking.
        Returns:
            Lista de top_n chunks ordenados por rerank_score descendente,
            cada uno con campo 'rerank_score' agregado.
        Raises:
            RuntimeError: Si el modelo no está cargado.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Llama a load() antes de rerank()")

        if not chunks:
            return []

        scores = self._score_pairs(query, [c["text"] for c in chunks])

        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        result = []
        for chunk, score in ranked[:top_n]:
            c = dict(chunk)
            c["rerank_score"] = float(score)
            result.append(c)

        return result

    def _score_pairs(self, query: str, texts: list[str]) -> list[float]:
        """
        Calcula scores de relevancia para pares (query, text) en batches.

        BGE-reranker-v2-m3 usa el logit del label 1 (relevante).
        Los batches pequeños evitan OOM en RTX 3050 4GB.

        Args:
            query: Texto de la query.
            texts: Lista de textos de chunks candidatos.
        Returns:
            Lista de scores (float) en el mismo orden que texts.
        """
        all_scores: list[float] = []

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            pairs = [[query, t] for t in batch_texts]

            encoded = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self._model(**encoded).logits

            # BGE reranker: score de relevancia es logits[:, 0] o sigmoid(logits)
            if logits.shape[-1] == 1:
                batch_scores = torch.sigmoid(logits.squeeze(-1)).cpu().tolist()
            else:
                batch_scores = logits[:, 0].cpu().tolist()

            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]

            all_scores.extend(batch_scores)

        return all_scores

    def save_checkpoint(self, path: str, round_n: int) -> None:
        """
        Guarda los pesos actuales del reranker como checkpoint de la ronda N.

        El nombre del archivo incluye el número de ronda para que evaluate_rn
        pueda cargar siempre el más reciente sin ambigüedad.

        Args:
            path: Directorio donde guardar (config: active_learning.checkpoint_dir).
            round_n: Número de ronda del fine-tuning.
        Raises:
            RuntimeError: Si el modelo no está cargado.
        """
        if self._model is None:
            raise RuntimeError("No hay modelo cargado para guardar")
        import pathlib
        ckpt_dir = pathlib.Path(path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"reranker_round_{round_n}.pt"
        torch.save(self._model.state_dict(), ckpt_path)
        logger.success("Checkpoint guardado | ronda={} | path={}", round_n, ckpt_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Carga los pesos de un checkpoint de fine-tuning.

        Debe llamarse DESPUÉS de load() (que carga la arquitectura base).
        Esto permite cargar los pesos fine-tuneados sobre la arquitectura
        ya cargada en el device correcto.

        Args:
            checkpoint_path: Ruta al archivo .pt del checkpoint.
        Raises:
            RuntimeError: Si el modelo base no está cargado.
            FileNotFoundError: Si el checkpoint no existe.
        """
        import pathlib
        if self._model is None:
            raise RuntimeError("Llama a load() antes de load_checkpoint()")
        p = pathlib.Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {p}")
        state_dict = torch.load(p, map_location=self.device)
        self._model.load_state_dict(state_dict)
        self._model.eval()
        logger.success("Checkpoint cargado | path={}", p)

    def fine_tune_contrastive(
        self,
        triples: list[dict],
        learning_rate: float,
        num_epochs: int,
        batch_size: int,
        margin: float,
    ) -> dict:
        """
        Fine-tuning contrastivo del reranker con tripletas (query, pos, neg).

        Usa MarginRankingLoss: el score del chunk positivo debe superar al
        negativo por al menos `margin`. Esto replica la dinámica SAIL donde
        los chunks aprobados por el oráculo son positivos y los rechazados
        son negativos duros.

        Args:
            triples: Lista de dicts con keys 'query', 'positive_text', 'negative_text'.
            learning_rate: Tasa de aprendizaje para Adafactor.
            num_epochs: Número de épocas de entrenamiento.
            batch_size: Batch size para el fine-tuning.
            margin: Margen mínimo de separación en MarginRankingLoss.
        Returns:
            Dict con 'final_loss' (float) y 'num_triples' (int).
        Raises:
            RuntimeError: Si el modelo no está cargado.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Llama a load() antes de fine_tune_contrastive()")

        # Fine-tuning en CPU para evitar OOM en RTX 3050 4GB.
        # Adafactor reemplaza AdamW: usa O(√n) de estado en lugar de O(2n),
        # ahorrando ~4GB de RAM para un modelo de ~560M params.
        was_on_gpu = next(self._model.parameters()).device.type == "cuda"
        if was_on_gpu:
            self._model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Modelo movido a CPU para fine-tuning (evita OOM en 4GB VRAM)")

        self._model.float()  # fp32 en CPU: estable, sin NaN
        self._model.train()
        train_device = "cpu"

        # Adafactor: estado del optimizador O(√n) vs O(2n) de AdamW.
        # Para bge-reranker-v2-m3 (~560M params): ~100MB vs ~4.5GB.
        # scale_parameter=False + relative_step=False para usar lr fijo explícito.
        try:
            from transformers.optimization import Adafactor
            optimizer = Adafactor(
                self._model.parameters(),
                lr=learning_rate,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
            )
            optimizer_name = "Adafactor"
        except ImportError:
            optimizer = torch.optim.AdamW(self._model.parameters(), lr=learning_rate)
            optimizer_name = "AdamW"

        loss_fn = torch.nn.MarginRankingLoss(margin=margin)
        target = torch.ones(1).to(train_device)

        total_loss = 0.0
        steps = 0
        nan_skipped = 0

        logger.info(
            "Fine-tuning contrastivo | {} tripletas | epochs={} | lr={} | margin={} | fp32=True | optimizer={}",
            len(triples), num_epochs, learning_rate, margin, optimizer_name,
        )

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for start in range(0, len(triples), batch_size):
                batch = triples[start : start + batch_size]

                for triple in batch:
                    query = triple["query"]
                    pos_text = triple["positive_text"]
                    neg_text = triple["negative_text"]

                    pos_enc = self._tokenizer(
                        [[query, pos_text]], padding=True, truncation=True,
                        max_length=512, return_tensors="pt",
                    ).to(train_device)
                    neg_enc = self._tokenizer(
                        [[query, neg_text]], padding=True, truncation=True,
                        max_length=512, return_tensors="pt",
                    ).to(train_device)

                    pos_logit = self._model(**pos_enc).logits
                    neg_logit = self._model(**neg_enc).logits

                    pos_score = pos_logit[:, 0] if pos_logit.shape[-1] > 1 else torch.sigmoid(pos_logit.squeeze(-1))
                    neg_score = neg_logit[:, 0] if neg_logit.shape[-1] > 1 else torch.sigmoid(neg_logit.squeeze(-1))

                    loss = loss_fn(pos_score, neg_score, target)

                    if torch.isnan(loss) or torch.isinf(loss):
                        nan_skipped += 1
                        del pos_enc, neg_enc, pos_logit, neg_logit
                        continue

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    steps += 1

                    del pos_enc, neg_enc, pos_logit, neg_logit, loss

            gc.collect()
            logger.info(
                "Epoch {}/{} | loss={:.4f} | nan_skipped={}",
                epoch + 1, num_epochs, epoch_loss, nan_skipped,
            )
            total_loss = epoch_loss

        # Liberar optimizer ANTES de convertir de vuelta a fp16/GPU para
        # evitar OOM en la transición (optimizer ocupa ~4GB para modelos ~560M).
        del optimizer
        gc.collect()

        # Volver a GPU fp16 para inferencia eficiente
        if was_on_gpu:
            if self.fp16:
                self._model.half()
            self._model.to(self.device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Modelo devuelto a {} fp16 para inferencia", self.device)

        self._model.eval()
        avg_loss = total_loss / max(steps, 1)
        logger.success(
            "Fine-tuning completado | {} pasos | {} NaN saltados | loss_final={:.4f}",
            steps, nan_skipped, avg_loss,
        )
        return {"final_loss": avg_loss, "num_triples": len(triples)}
