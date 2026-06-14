"""
llm.py
======
Cliente Groq para generación de respuestas con llama-3.3-70b-versatile.

Posición en el pipeline: Paso 6 — genera respuesta a partir de top-5 chunks.
Decisión: Groq corre en la nube, no consume VRAM local. Es el cuello
de botella real del pipeline (rate limits, latencia de red), no la GPU.

Rotación de claves: carga GROQ_API_KEY … GROQ_API_KEY9. En rate limit
pasa a la siguiente clave antes de reintentar, permitiendo esquivar tanto
límites por minuto (TPM) como límites diarios (TPD) de distintas cuentas.
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


def _load_groq_keys() -> list[str]:
    """Carga GROQ_API_KEY y GROQ_API_KEY2…GROQ_API_KEY9 del entorno."""
    names = ["GROQ_API_KEY"] + [f"GROQ_API_KEY{i}" for i in range(2, 10)]
    keys = [k for name in names if (k := os.getenv(name))]
    return list(dict.fromkeys(keys))  # deduplica preservando orden


class GroqLLM:
    """Wrapper sobre groq.Groq con rotación de claves y retry en rate limit."""

    def __init__(self, config: dict):
        self.model_id: str = config["generation"]["model_id"]
        self.max_tokens: int = config["generation"]["max_tokens"]
        self.temperature: float = config["generation"]["temperature"]

        self._keys = _load_groq_keys()
        if not self._keys:
            raise RuntimeError("No se encontró ninguna GROQ_API_KEY en el entorno")
        self._key_idx = 0
        self._client = Groq(api_key=self._keys[0])
        logger.info(
            "GroqLLM inicializado | modelo={} | claves={}",
            self.model_id, len(self._keys),
        )

    def _rotate_key(self) -> bool:
        """Pasa a la siguiente clave. Retorna True si hay otra disponible."""
        next_idx = self._key_idx + 1
        if next_idx >= len(self._keys):
            return False
        self._key_idx = next_idx
        self._client = Groq(api_key=self._keys[self._key_idx])
        logger.info("Rotando a clave Groq #{}", self._key_idx + 1)
        return True

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        """
        Genera una respuesta dada la query y los chunks de contexto recuperados.

        Por qué no streaming: necesitamos la respuesta completa antes de evaluar
        ROUGE-L y BERTScore en Punto B.

        Args:
            query: Pregunta del usuario.
            context_chunks: Lista de dicts con key 'text' (top-5 de Qdrant).
        Returns:
            Respuesta generada como string. String vacío si todas las claves fallan.
        """
        parts = []
        for i, c in enumerate(context_chunks):
            section = c.get("section") or c.get("section_label") or ""
            label = f"[Excerpt {i + 1}" + (f" — {section}]" if section else "]")
            parts.append(f"{label}\n{c['text']}")
        context = "\n\n---\n\n".join(parts)
        prompt = _PROMPT_TEMPLATE.format(context=context, question=query)
        return self._call_with_retry(prompt)

    def _call_with_retry(self, prompt: str) -> str:
        """Llamada con rotación de claves en RateLimitError."""
        keys_tried = 0
        total_keys = len(self._keys)
        # Claves marcadas como muertas (401/403) en esta sesión — skip inmediato
        dead_keys: set[int] = getattr(self, "_dead_keys", set())
        self._dead_keys = dead_keys

        while keys_tried < total_keys:
            # Saltar claves ya marcadas como muertas
            while self._key_idx in dead_keys and keys_tried < total_keys:
                if not self._rotate_key():
                    break
                keys_tried += 1

            if self._key_idx in dead_keys:
                break  # todas las claves están muertas

            for attempt in range(3):
                try:
                    response = self._client.chat.completions.create(
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
                        "Rate limit clave #{} (intento {}/3) | esperando {}s",
                        self._key_idx + 1, attempt + 1, wait,
                    )
                    time.sleep(wait)

                except APIError as e:
                    err_str = str(e)
                    # 401/403 = clave inválida/suspendida → marcar muerta y rotar
                    if any(c in err_str for c in ("401", "403", "invalid_api_key", "Invalid API Key", "Forbidden")):
                        logger.warning("Clave #{} muerta ({}) — marcando y rotando", self._key_idx + 1, err_str[:40])
                        dead_keys.add(self._key_idx)
                        break  # sale del loop de intentos, rota clave
                    logger.error("Fallo Groq API (no recuperable): {}", err_str)
                    return ""

                except (TimeoutError, OSError, ConnectionError) as e:
                    wait = (2 ** attempt) * 5
                    logger.warning(
                        "Timeout/conexión clave #{} (intento {}/3) | esperando {}s",
                        self._key_idx + 1, attempt + 1, wait,
                    )
                    time.sleep(wait)

            # 3 intentos fallidos o 401 con esta clave — intentar la siguiente
            if not self._rotate_key():
                break
            keys_tried += 1

        logger.error("Groq falló con todas las claves disponibles — retornando string vacío")
        return ""
