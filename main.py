"""
main.py
=======
Punto de entrada del proyecto SAIL-RAG.

Modos de ejecución:
  python main.py --pipeline rag1 --mode ingest
      Descarga PeerQA, genera embeddings, guarda checkpoint en checkpoints/.

  python main.py --pipeline rag1 --mode ingest --force-reindex
      Re-indexa desde cero aunque exista un checkpoint previo.

  python main.py --pipeline rag1 --mode evaluate
      Carga checkpoint, corre el pipeline completo, guarda JSON en results/rag1/.

  python main.py --pipeline rag1 --mode visualize
      Lee results/rag1/metrics_post_llm.json y genera 5 PNG en results/rag1/figures/.
      NUNCA carga torch ni ningún modelo en este modo.

  python main.py --pipeline rag2 --mode ingest
      Descarga PDFs vía Docling, re-indexa con secciones en results/rag2/.

  python main.py --pipeline rag2 --mode evaluate_r0
      Ronda 0: reranker base sin fine-tuning, guarda 3 JSONs en results/rag2/round_0/.

  python main.py --pipeline rag2 --mode train
      Corre bucle de aprendizaje activo, guarda checkpoints en checkpoints/reranker/.

  python main.py --pipeline rag2 --mode evaluate_rn
      Ronda N: carga checkpoint más reciente, evalúa, guarda en results/rag2/round_n/.
      NUNCA reentrena — requiere haber corrido --mode train primero.

  python main.py --pipeline rag2 --mode visualize
      Lee JSONs de RAG 2 y genera figuras. NUNCA carga modelos.

El .env se busca primero en el directorio padre de este archivo (juancho/.env),
luego en el CWD como fallback para entornos de desarrollo alternativos.
"""

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Carga config.yaml. Todos los hiperparámetros vienen de aquí."""
    path = Path(config_path)
    if not path.exists():
        logger.error("No se encontró config en {}", path.resolve())
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    """Configura loguru con rotación automática."""
    logs_dir = Path(config["paths"]["logs"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        logs_dir / "rag1_{time}.log",
        rotation="1 MB",
        retention="7 days",
        level="DEBUG",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAIL-RAG Comparative Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pipeline", choices=["rag1", "rag2", "rag3"], required=True,
        help="Pipeline a ejecutar",
    )
    parser.add_argument(
        "--mode",
        choices=["ingest", "evaluate", "judge", "visualize",
                 "evaluate_r0", "train", "evaluate_rn"],
        required=True,
        help="Modo de ejecución",
    )
    parser.add_argument(
        "--force-reindex", action="store_true",
        help="Re-indexa desde cero aunque exista checkpoint (solo para ingest)",
    )
    args = parser.parse_args()

    # Cargar .env desde juancho/.env (directorio padre del proyecto)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.debug(".env cargado desde {}", env_path)
    else:
        load_dotenv()
        logger.debug(".env buscado en CWD (fallback)")

    config = load_config()
    setup_logging(config)

    logger.info(
        "SAIL-RAG iniciado | pipeline={} | mode={}",
        args.pipeline, args.mode,
    )

    # ── VISUALIZE: importación tardía para no cargar torch ──────────────────
    if args.mode == "visualize" and args.pipeline == "rag1":
        from src.evaluation.visualizer import RAG1Visualizer
        viz = RAG1Visualizer(config["paths"]["results"])
        viz.visualize_all()
        return

    if args.mode == "visualize" and args.pipeline == "rag3":
        from src.evaluation.visualizer_rag3 import RAG3Visualizer
        RAG3Visualizer(config["paths"]["results"]).visualize_all()
        return

    if args.mode == "visualize" and args.pipeline == "rag2":
        from src.evaluation.visualizer_rag2 import RAG2Visualizer
        # Visualiza round_0 por defecto; round_n si existe
        for round_dir in ["round_0", "round_n"]:
            json_path = (
                Path(config["paths"]["results"]) / "rag2" / round_dir / "metrics_post_llm.json"
            )
            if json_path.exists():
                viz = RAG2Visualizer(config["paths"]["results"], round_dir=round_dir)
                viz.visualize_all()
        return

    # ── RAG 1 ────────────────────────────────────────────────────────────────
    if args.pipeline == "rag1":
        from src.pipelines.rag1_baseline import RAG1Pipeline
        pipeline = RAG1Pipeline(config)

        if args.mode == "ingest":
            emb_path = Path(config["paths"]["checkpoints"]) / "rag1_embeddings.npy"
            if emb_path.exists() and not args.force_reindex:
                logger.info(
                    "Checkpoint encontrado en {}. "
                    "Usa --force-reindex para re-indexar desde cero.",
                    emb_path,
                )
                return
            pipeline.ingest()

        elif args.mode == "evaluate":
            pipeline.evaluate()

        elif args.mode == "judge":
            pipeline.run_judge_only()

    # ── RAG 2 ────────────────────────────────────────────────────────────────
    elif args.pipeline == "rag2":
        if args.mode == "ingest":
            from src.pipelines.rag2_sail_r0 import RAG2PipelineR0
            pipeline_r0 = RAG2PipelineR0(config)
            emb_path = Path(config["paths"]["checkpoints"]) / "rag2_embeddings.npy"
            if emb_path.exists() and not args.force_reindex:
                logger.info(
                    "Checkpoint RAG 2 encontrado en {}. "
                    "Usa --force-reindex para re-indexar desde cero.",
                    emb_path,
                )
            else:
                pipeline_r0.ingest()

        elif args.mode == "evaluate_r0":
            from src.pipelines.rag2_sail_r0 import RAG2PipelineR0
            RAG2PipelineR0(config).evaluate()

        elif args.mode == "train":
            from src.pipelines.rag2_sail_rn import RAG2PipelineRN
            RAG2PipelineRN(config).train()

        elif args.mode == "evaluate_rn":
            from src.pipelines.rag2_sail_rn import RAG2PipelineRN
            RAG2PipelineRN(config).evaluate()

        else:
            logger.error(
                "Modo '{}' no válido para --pipeline rag2. "
                "Modos disponibles: ingest, evaluate_r0, train, evaluate_rn, visualize",
                args.mode,
            )


    # ── RAG 3 ────────────────────────────────────────────────────────────────
    elif args.pipeline == "rag3":
        if args.mode == "evaluate":
            from src.pipelines.rag3_curation_only import RAG3Pipeline
            RAG3Pipeline(config).evaluate()
        else:
            logger.error(
                "Modo '{}' no válido para --pipeline rag3. "
                "Modos disponibles: evaluate, visualize",
                args.mode,
            )


if __name__ == "__main__":
    main()
