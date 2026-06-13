"""
llm.py
======
Cliente Groq para generación de respuestas con llama-3.3-70b-versatile.

Posición en el pipeline: Paso 6 — genera respuesta a partir de top-5 chunks.
Decisión: Groq corre en la nube, no consume VRAM local. Es el cuello
de botella real del pipeline (rate limits, latencia de red), no la GPU.

Retry con backoff exponencial: 5s → 10s → 20s en rate limit.
APIError genérico retorna string vacío para no detener el pipeline.
"""

import os
import time

from dotenv import load_dotenv
from groq import APIError, Groq, RateLimitError
from loguru import logger

load_dotenv()

_SYSTEM_PROMPT = (
    "You are a scientific paper reviewer assistant. "
    "Your answers must be grounded EXCLUSIVELY in the context excerpts provided by the user. "
    "Do NOT use any prior knowledge, training data, or information outside those excerpts. "
    "If the excerpts do not contain enough information to answer, respond only with: "
    "'The provided context does not contain sufficient information to answer this question.'"
)

_PROMPT_TEMPLATE = """Context excerpts from the paper:

{context}

Question: {question}

Answer using ONLY the excerpts above. Do not add information that is not explicitly stated in the excerpts."""


class GroqLLM:
    """Wrapper sobre groq.Groq con retry en rate limit y error silencioso en APIError."""

    def __init__(self, config: dict):
        self.model_id: str = config["generation"]["model_id"]
        self.max_tokens: int = config["generation"]["max_tokens"]
        self.temperature: float = config["generation"]["temperature"]
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        logger.info("GroqLLM inicializado | modelo={}", self.model_id)

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        """
        Genera una respuesta dada la query y los chunks de contexto recuperados.

        Por qué no streaming: necesitamos la respuesta completa antes de evaluar
        ROUGE-L y BERTScore en Punto B.

        Args:
            query: Pregunta del usuario.
            context_chunks: Lista de dicts con key 'text' (top-5 de Qdrant).
        Returns:
            Respuesta generada como string. String vacío si la API falla.
        """
        parts = []
        for i, c in enumerate(context_chunks):
            section = c.get("section") or c.get("section_label") or ""
            label = f"[Excerpt {i + 1}" + (f" — {section}]" if section else "]")
            parts.append(f"{label}\n{c['text']}")
        context = "\n\n---\n\n".join(parts)
        prompt = _PROMPT_TEMPLATE.format(context=context, question=query)
        return self._call_with_retry(prompt, max_retries=3)

    def _call_with_retry(self, prompt: str, max_retries: int) -> str:
        """Llamada a Groq con backoff exponencial en RateLimitError."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()

            except RateLimitError as e:
                wait = (2 ** attempt) * 5  # 5s, 10s, 20s
                logger.warning(
                    "Rate limit Groq (intento {}/{}) | esperando {}s | {}",
                    attempt + 1, max_retries, wait, str(e),
                )
                time.sleep(wait)

            except APIError as e:
                logger.error("Fallo Groq API: {}", str(e))
                return ""

        logger.error("Groq falló después de {} intentos — retornando string vacío", max_retries)
        return ""
