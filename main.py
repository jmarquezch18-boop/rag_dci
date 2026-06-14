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

  python main.py --pipeline rag3 --mode ingest
      Descarga PeerQA desde HuggingFace, construye corpus/ de texto plano por sección,
      guarda checkpoints/rag3_dci_qa_pairs.json y checkpoints/rag3_dci_corpus_map.json.
      Requiere que Ollama esté corriendo: ollama serve &

  python main.py --pipeline rag3 --mode evaluate
      Corre el pipeline DCI completo: DeepSeek-R1 busca en corpus/ → SAIL-RAG oracle
      → Groq generación. Guarda metrics_dci.json y agent_traces/ en results/rag3/.
      REQUIERE haber corrido --mode ingest primero.

  python main.py --pipeline rag3 --mode visualize
      Lee results/rag3/metrics_dci.json y genera figuras DCI. NUNCA carga modelos.
      Si existen JSONs de RAG 1 y RAG 2, genera también final_comparison_rag1_rag2_rag3.png.

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
        "--pipeline", choices=["rag1", "rag2", "rag2_1", "rag2_3", "rag3", "rag3_1", "rag3_2"], required=True,
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
        from src.evaluation.visualizer_rag3_dci import RAG3DCIVisualizer
        RAG3DCIVisualizer(config["paths"]["results"]).visualize_all()
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


    # ── RAG 2.1 — Oracle only (sin reranker) ────────────────────────────────
    elif args.pipeline == "rag2_1":
        from src.pipelines.rag2_1_oracle_only import RAG21Pipeline
        if args.mode == "evaluate":
            RAG21Pipeline(config).evaluate()
        else:
            logger.error(
                "Modo '{}' no válido para --pipeline rag2_1. "
                "Modos disponibles: evaluate",
                args.mode,
            )

    # ── RAG 2.3 — Reranker only (sin oracle) ────────────────────────────────
    elif args.pipeline == "rag2_3":
        from src.pipelines.rag2_3_reranker_only import RAG23Pipeline
        if args.mode == "evaluate":
            RAG23Pipeline(config).evaluate()
        else:
            logger.error(
                "Modo '{}' no válido para --pipeline rag2_3. "
                "Modos disponibles: evaluate",
                args.mode,
            )

    # ── RAG 3.1 — DCI + Docling corpus + IBM Granite 3.2 ────────────────────
    elif args.pipeline == "rag3_1":
        from src.pipelines.rag3_1_granite_docling import RAG31Pipeline
        pipeline31 = RAG31Pipeline(config)
        if args.mode == "ingest":
            pipeline31.ingest()
        elif args.mode == "evaluate":
            pipeline31.evaluate()
        else:
            logger.error(
                "Modo '{}' no válido para --pipeline rag3_1. "
                "Modos disponibles: ingest, evaluate",
                args.mode,
            )

    # ── RAG 3.2 — DCI + Docling corpus + DeepSeek-R1 14B ────────────────────
    elif args.pipeline == "rag3_2":
        from src.pipelines.rag3_2_deepseek_docling import RAG32Pipeline
        pipeline32 = RAG32Pipeline(config)
        if args.mode == "ingest":
            pipeline32.ingest()
        elif args.mode == "evaluate":
            pipeline32.evaluate()
        else:
            logger.error(
                "Modo '{}' no válido para --pipeline rag3_2. "
                "Modos disponibles: ingest, evaluate",
                args.mode,
            )

    # ── RAG 3 — DCI + SAIL-RAG ───────────────────────────────────────────────
    elif args.pipeline == "rag3":
        from src.pipelines.rag3_dci_sail import RAG3DCIPipeline
        pipeline_dci = RAG3DCIPipeline(config)

        if args.mode == "ingest":
            pipeline_dci.ingest()

        elif args.mode == "evaluate":
            pipeline_dci.evaluate()

        else:
            logger.error(
                "Modo '{}' no válido para --pipeline rag3. "
                "Modos disponibles: ingest, evaluate, visualize",
                args.mode,
            )


if __name__ == "__main__":
    main()
