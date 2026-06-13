"""
judge.py
========
LLM-judge metrics via RAGAS (Punto C del pipeline).

Métrica:
  - faithfulness: ¿Está la respuesta fundamentada en los chunks recuperados?
                  1.0 = totalmente soportada, 0.0 = hallucination pura.

Configuración via .env:
  RAGAS_LLM=gemini/gemini-2.0-flash   (o groq/llama-3.3-70b-versatile)
  GEMINI_API_KEY=<tu_key>              (para provider gemini)
  GROQ_API_KEY=<tu_key>               (para provider groq)

Si falla (rate limit, error de red, etc.), retorna None sin detener el pipeline.
"""

from __future__ import annotations

import math
import os
import sys
import types

from loguru import logger


def _patch_langchain_community() -> None:
    """
    Inyecta stubs para módulos que RAGAS 0.2.x espera de langchain-community
    pero que fueron eliminados en langchain-community >=0.4.x (movidos a
    paquetes separados como langchain-google-vertexai).
    Debe llamarse antes de cualquier import de ragas.
    """
    stubs = [
        "langchain_community.chat_models.vertexai",
    ]
    for name in stubs:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.ChatVertexAI = None
            sys.modules[name] = mod

            # Asegurar que el paquete padre conozca el sub-módulo
            parent_name, _, attr = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, attr, mod)


def _build_ragas_llm():
    """
    Construye el wrapper RAGAS→LangChain leyendo RAGAS_LLM del env.

    Soporta dos proveedores via endpoint OpenAI-compatible:
      - groq/<model>   → api.groq.com          (GROQ_API_KEY)
      - gemini/<model> → generativelanguage.googleapis.com  (GEMINI_API_KEY)
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    ragas_llm_env = os.getenv("RAGAS_LLM", "groq/llama-3.3-70b-versatile")
    provider, _, model_id = ragas_llm_env.partition("/")

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    else:  # groq (default)
        api_key = (
            os.getenv("GROQ_API_KEY5")
            or os.getenv("GROQ_API_KEY4")
            or os.getenv("GROQ_API_KEY3")
            or os.getenv("GROQ_API_KEY2")
            or os.getenv("GROQ_API_KEY")
        )
        base_url = "https://api.groq.com/openai/v1"

    chat = ChatOpenAI(
        model=model_id,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        max_tokens=2048,
    )
    return LangchainLLMWrapper(chat)


def compute_ragas_batch(
    queries: list[str],
    responses: list[str],
    contexts_per_query: list[list[str]],
    references: list[str] | None = None,
) -> list[dict]:
    """
    Calcula RAGAS Faithfulness sobre el batch completo en una sola llamada.

    Args:
        queries:             Preguntas del dataset.
        responses:           Respuestas generadas por el LLM del pipeline.
        contexts_per_query:  Lista de listas de textos de chunks recuperados.
        references:          Respuestas de referencia (ground truth). Opcional.
    Returns:
        Lista de dicts {"faithfulness": float | None}, uno por query.
    """
    n = len(queries)
    fallback = [{"faithfulness": None}] * n

    try:
        _patch_langchain_community()
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import Faithfulness

        ragas_llm = _build_ragas_llm()
        faithfulness_metric = Faithfulness(llm=ragas_llm)

        refs = references if references else [""] * n
        samples = [
            SingleTurnSample(
                user_input=q,
                response=r,
                retrieved_contexts=ctxs,
                reference=ref,
            )
            for q, r, ctxs, ref in zip(queries, responses, contexts_per_query, refs)
        ]

        from ragas import RunConfig
        run_config = RunConfig(timeout=180, max_retries=3, max_wait=60)

        dataset = EvaluationDataset(samples=samples)
        logger.info("Lanzando RAGAS Faithfulness sobre {} queries (Groq judge)...", n)
        result = evaluate(dataset, metrics=[faithfulness_metric], run_config=run_config)
        df = result.to_pandas()

        out = []
        for _, row in df.iterrows():
            val = row.get("faithfulness")
            try:
                f = float(val)
                out.append({"faithfulness": None if math.isnan(f) else f})
            except (TypeError, ValueError):
                out.append({"faithfulness": None})

        logger.success(
            "RAGAS completado | faithfulness media={:.4f}",
            sum(r["faithfulness"] for r in out if r["faithfulness"] is not None)
            / max(1, sum(1 for r in out if r["faithfulness"] is not None)),
        )
        return out

    except Exception as exc:
        logger.error("RAGAS batch falló (no detiene el pipeline): {}", str(exc))
        return fallback
