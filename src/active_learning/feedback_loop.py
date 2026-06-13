"""
feedback_loop.py
================
Bucle de aprendizaje activo: acumula decisiones del oráculo y dispara
fine-tuning contrastivo del reranker cuando se supera el umbral.

Posición en el pipeline RAG 2: coordina entre oracle.py (genera tripletas)
y reranker.py (fine-tuning). El FeedbackStore persiste las tripletas en
checkpoints/reranker/feedback_store.json para que el entrenamiento sea
reanudable sin repetir queries.

Flujo:
  Ronda de evaluación → oracle.curate() → tripletas → FeedbackStore.add()
  Si len(store) >= accumulation_threshold → reranker.fine_tune_contrastive()
  → reranker.save_checkpoint(round_N) → ActiveLearningCurve.record()

El número de ronda N se incrementa en cada fine-tuning y se persiste en
el mismo directorio del checkpoint para que evaluate_rn lo detecte.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger


class FeedbackStore:
    """
    Almacén persistente de tripletas (query, positivo, negativo).

    Las tripletas se acumulan en disco para sobrevivir reinicios del proceso.
    Cuando se supera el umbral, el pipeline lee todas las tripletas y lanza
    fine-tuning. Después del fine-tuning el store puede vaciarse (clear_after_train=True)
    o acumularse indefinidamente para rondas futuras (configurable).
    """

    def __init__(self, store_path: str):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._triples: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self.store_path.exists():
            with open(self.store_path, encoding="utf-8-sig") as f:
                data = json.load(f)
            logger.info(
                "FeedbackStore cargado | {} tripletas | path={}",
                len(data), self.store_path,
            )
            return data
        return []

    def add(self, triples: list[dict]) -> int:
        """
        Agrega tripletas al store y persiste en disco.

        Args:
            triples: Lista de dicts con query, positive_text, negative_text.
        Returns:
            Total de tripletas acumuladas después de la adición.
        """
        self._triples.extend(triples)
        self._save()
        return len(self._triples)

    def get_all(self) -> list[dict]:
        """Retorna todas las tripletas acumuladas."""
        return list(self._triples)

    def clear(self) -> None:
        """Vacía el store después de un ciclo de fine-tuning."""
        self._triples = []
        self._save()
        logger.info("FeedbackStore vaciado después de fine-tuning")

    def __len__(self) -> int:
        return len(self._triples)

    def _save(self) -> None:
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self._triples, f, ensure_ascii=False, indent=2)


class ActiveLearningCurve:
    """
    Registra métricas del reranker por ronda de fine-tuning.

    Se persiste en checkpoints/reranker/active_learning_curve.json para
    que el modo visualize lo lea y grafique la curva de aprendizaje sin
    necesidad de recargar el modelo.
    """

    def __init__(self, curve_path: str):
        self.curve_path = Path(curve_path)
        self.curve_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self.curve_path.exists():
            with open(self.curve_path, encoding="utf-8-sig") as f:
                return json.load(f)
        return []

    def record(self, round_n: int, metrics: dict, train_info: dict) -> None:
        """
        Registra métricas de evaluación y de entrenamiento para la ronda N.

        Args:
            round_n: Número de ronda de fine-tuning.
            metrics: Dict de métricas de evaluación post-entrenamiento
                     (mrr, recall_at_5, etc.).
            train_info: Dict con info del entrenamiento (final_loss, num_triples).
        """
        record = {"round": round_n, **metrics, **train_info}
        self._records.append(record)
        self._save()
        logger.info(
            "Curva AL registrada | ronda={} | MRR={:.4f} | loss={:.4f}",
            round_n,
            metrics.get("mrr", 0.0),
            train_info.get("final_loss", 0.0),
        )

    def get_all(self) -> list[dict]:
        """Retorna todos los registros de la curva."""
        return list(self._records)

    def _save(self) -> None:
        with open(self.curve_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2)


def get_latest_checkpoint(checkpoint_dir: str) -> tuple[str | None, int]:
    """
    Encuentra el checkpoint más reciente en el directorio.

    Los checkpoints tienen formato reranker_round_{N}.pt. Se retorna el
    de mayor N. Si no hay ninguno, retorna (None, 0).

    Args:
        checkpoint_dir: Directorio donde buscar checkpoints .pt.
    Returns:
        Tupla (path_str | None, round_number).
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None, 0

    checkpoints = sorted(ckpt_dir.glob("reranker_round_*.pt"))
    if not checkpoints:
        return None, 0

    latest = checkpoints[-1]
    try:
        round_n = int(latest.stem.split("_")[-1])
    except ValueError:
        round_n = 0

    logger.info("Checkpoint más reciente encontrado | ronda={} | path={}", round_n, latest)
    return str(latest), round_n


class ActiveLearningLoop:
    """
    Orquesta el ciclo completo de aprendizaje activo.

    Acumula tripletas del oráculo, decide cuándo disparar fine-tuning,
    coordina con el reranker y registra la curva de aprendizaje.
    """

    def __init__(self, config: dict):
        al_cfg = config.get("rag2", {}).get("active_learning", {})
        self.accumulation_threshold: int = al_cfg.get("accumulation_threshold", 50)
        self.learning_rate: float = al_cfg.get("learning_rate", 2e-5)
        self.num_epochs: int = al_cfg.get("num_epochs", 3)
        self.batch_size: int = al_cfg.get("batch_size", 8)
        self.margin: float = al_cfg.get("margin", 0.5)
        self.checkpoint_dir: str = al_cfg.get("checkpoint_dir", "./checkpoints/reranker")

        self.store = FeedbackStore(al_cfg.get("feedback_store", "./checkpoints/reranker/feedback_store.json"))
        self.curve = ActiveLearningCurve(al_cfg.get("curve_file", "./checkpoints/reranker/active_learning_curve.json"))
        self._current_round = self._detect_current_round()

        logger.info(
            "ActiveLearningLoop | threshold={} | ronda_actual={} | tripletas_acumuladas={}",
            self.accumulation_threshold, self._current_round, len(self.store),
        )

    def _detect_current_round(self) -> int:
        """Detecta la ronda actual buscando el checkpoint más reciente."""
        _, round_n = get_latest_checkpoint(self.checkpoint_dir)
        return round_n

    def add_oracle_decisions(self, triples: list[dict]) -> dict:
        """
        Agrega tripletas del oráculo y dispara fine-tuning si se supera el umbral.

        Si no se supera el umbral, las tripletas quedan en el store para
        la próxima iteración.

        Args:
            triples: Tripletas generadas por CurationOracle.curate().
        Returns:
            Dict con 'trained' (bool), 'total_triples' (int),
            'round_n' (int si trained, else None).
        """
        total = self.store.add(triples)
        logger.debug(
            "Tripletas agregadas | nuevas={} | total_acumulado={} | umbral={}",
            len(triples), total, self.accumulation_threshold,
        )

        if total >= self.accumulation_threshold:
            logger.info(
                "Umbral superado ({} >= {}) — disparando fine-tuning",
                total, self.accumulation_threshold,
            )
            return {"trained": True, "total_triples": total, "round_n": None}

        return {"trained": False, "total_triples": total, "round_n": None}

    def run_fine_tuning(self, reranker, eval_metrics: dict | None = None) -> int:
        """
        Ejecuta el ciclo completo de fine-tuning para la ronda actual.

        Debe llamarse solo cuando add_oracle_decisions() retorna trained=True
        o al final del modo train para forzar un ciclo.

        Args:
            reranker: Instancia de BGEReranker con modelo cargado.
            eval_metrics: Métricas de evaluación post-training (opcionales).
        Returns:
            Número de ronda completada.
        """
        triples = self.store.get_all()
        if not triples:
            logger.warning("No hay tripletas para fine-tuning")
            return self._current_round

        train_info = reranker.fine_tune_contrastive(
            triples=triples,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            margin=self.margin,
        )

        self._current_round += 1
        reranker.save_checkpoint(self.checkpoint_dir, self._current_round)

        metrics = eval_metrics or {}
        self.curve.record(self._current_round, metrics, train_info)
        self.store.clear()

        logger.success(
            "Ronda {} completada | loss={:.4f} | {} tripletas usadas",
            self._current_round, train_info.get("final_loss", 0.0), len(triples),
        )
        return self._current_round
