"""
agent.py
========
Loop agéntico ReAct (Reason + Act) con DeepSeek-R1 14B via Ollama.

Por qué DeepSeek-R1 para retrieval: DCI propone que un LLM razonador puede
sustituir el retrieval vectorial one-shot con búsqueda iterativa guiada por
razonamiento explícito.

Por qué ReAct sin framework: no dependemos de LangChain, LlamaIndex ni
similares para mantener el código reproducible y sin acoplamientos externos.

Separación de roles: DeepSeek-R1 (Ollama) solo hace retrieval/razonamiento.
Groq (GroqLLM) hace la generación final.

Diseño stateless-per-iteration: cada iteración del loop es una llamada
independiente a Ollama (contexto fresco). El contexto acumulado (términos
buscados, resultados previos) se pasa como texto en el mensaje de usuario,
no como historial de chat. Esto evita que el modelo confunda turnos anteriores
y declare terminación prematura sin haber usado herramientas.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Optional

from loguru import logger

from src.dci.tools import DCITools

# ── Prompt del sistema ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a document search agent with access to a corpus of academic papers.
Your ONLY job is to output ONE tool call per response using EXACTLY this format:

ACTION: tool_name
ARGS: {"key": "value"}

Available tools:
- grep_corpus: search for a keyword in the corpus
  ARGS: {"pattern": "keyword"}   ← "pattern" is REQUIRED
- read_lines: read specific lines from a paper section (USE AFTER grep_corpus)
  ARGS: {"paper_id": "...", "section": "...", "start": 5, "end": 20}
- read_file: read a full section of a paper
  ARGS: {"paper_id": "...", "section": "..."}
- list_sections: list available sections for a paper
  ARGS: {"paper_id": "..."}
- list_papers: list all available papers
  ARGS: {}

RULES:
1. Output EXACTLY ONE tool call per response — no exceptions.
2. Do NOT answer from memory — ONLY use tools.
3. Do NOT write any explanation. Output ONLY the ACTION and ARGS lines.

HOW TO READ GREP RESULTS:
grep_corpus returns lines like:
  [paper_id=nlpeer_ARR-22_abc123 | section=methodology | line=42] some matched text

→ Use paper_id, section, and line directly in read_lines.

OPTIMAL WORKFLOW:
Step 1: grep_corpus to find which papers mention the topic
Step 2: read_lines using paper_id, section, line from grep result
Step 3: grep_corpus with another keyword if needed
(Repeat until enough evidence found)

EXAMPLE:
After grep returns: [paper_id=nlpeer_F1000-22_10-170 | section=results | line=42] accuracy improves...
Call:
ACTION: read_lines
ARGS: {"paper_id": "nlpeer_F1000-22_10-170", "section": "results", "start": 38, "end": 52}"""

# ── Regex de parsing ──────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# Acepta ACTION: o **ACTION:** (markdown bold del modelo)
_ACTION_RE = re.compile(r"\*{0,2}ACTION:\*{0,2}\s*(\w+)", re.IGNORECASE)
_ARGS_RE   = re.compile(r"\*{0,2}ARGS:\*{0,2}\s*(\{.*?\})", re.DOTALL)


def _extract_think_block(text: str) -> str:
    match = _THINK_RE.search(text)
    return match.group(1).strip() if match else ""


def _extract_action(text: str) -> Optional[dict]:
    """
    Parsea ACTION/ARGS del texto del LLM.

    Maneja tanto formato plain (ACTION:) como markdown bold (**ACTION:**).
    Si no hay ARGS, retorna args vacío para herramientas sin parámetros.
    """
    action_match = _ACTION_RE.search(text)
    if not action_match:
        return None

    tool_name = action_match.group(1).strip()
    after_action = text[action_match.end():]

    args_match = _ARGS_RE.search(after_action)
    if args_match:
        try:
            args = json.loads(args_match.group(1))
            return {"tool": tool_name, "args": args}
        except json.JSONDecodeError:
            pass

    return {"tool": tool_name, "args": {}}


def _call_ollama(
    base_url: str,
    model: str,
    messages: list[dict],
    temperature: float,
    timeout: int,
) -> str:
    """Llama al endpoint /api/chat de Ollama via HTTP (sin dependencias extra)."""
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"]
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama no está corriendo en {base_url}. Ejecuta: ollama serve &"
        ) from e
    except (TimeoutError, OSError) as e:
        raise RuntimeError(f"Ollama timeout tras {timeout}s: {e}") from e
    except (KeyError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Respuesta inesperada de Ollama: {e}") from e


def _build_user_message(
    query: str,
    iteration: int,
    searched_terms: list[str],
    last_result: str,
    evidence_count: int,
    grep_findings: list[str],
) -> str:
    """
    Construye el mensaje de usuario para la iteración actual.

    Diseño stateless: en lugar de pasar el historial completo de chat,
    construimos un resumen compacto del progreso que el modelo lee de nuevo
    en cada iteración — sin ver sus propias respuestas anteriores.
    """
    lines = [f"Research question: {query}", ""]

    if searched_terms:
        lines.append(f"Already searched for: {', '.join(repr(t) for t in searched_terms)}")

    if evidence_count > 0:
        lines.append(f"Evidence chunks collected: {evidence_count}")

    if grep_findings:
        lines.append("")
        lines.append("Previous grep results (use paper_id and line numbers to read more):")
        lines.extend(grep_findings[-15:])  # últimas 15 líneas de grep

    if last_result and iteration > 0:
        lines.append("")
        lines.append(f"Last tool result preview:")
        lines.append(last_result[:400])

    lines.append("")
    if iteration == 0:
        lines.append("Call grep_corpus now with a key term from the research question.")
    else:
        lines.append("Call the next most useful tool to find more evidence.")

    return "\n".join(lines)


class DCIAgent:
    """
    Agente de retrieval iterativo basado en DeepSeek-R1 via Ollama.

    Implementa loop ReAct con diseño stateless-per-iteration: cada llamada
    a Ollama es un contexto fresco (system + user). El estado acumulado
    (términos buscados, resultados grep) se pasa como texto en el mensaje
    de usuario, no como historial de chat multi-turno.

    Este diseño evita el problema de modelos 14B que declaran terminación
    prematura al ver tokens de stop en su propio historial previo.
    """

    def __init__(self, config: dict):
        rag3_cfg = config.get("rag3", {})
        ollama_cfg = rag3_cfg.get("ollama", {})
        agent_cfg = rag3_cfg.get("agent", {})

        self.base_url: str = ollama_cfg.get("base_url", "http://localhost:11434")
        self.model: str = ollama_cfg.get("model", "deepseek-r1:14b")
        self.temperature: float = ollama_cfg.get("temperature", 0.0)
        self.timeout: int = ollama_cfg.get("timeout_seconds", 120)
        self.max_iterations: int = agent_cfg.get("max_iterations", 5)
        self.max_evidence_chunks: int = agent_cfg.get("max_evidence_chunks", 10)

        self.tools = DCITools(config)

        logger.info(
            "DCIAgent inicializado | model={} | max_iterations={} | timeout={}s",
            self.model, self.max_iterations, self.timeout,
        )

    def check_ollama(self) -> None:
        """Verifica que Ollama esté corriendo antes de empezar."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=10):
                pass
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama no está corriendo en {self.base_url}. Ejecuta: ollama serve &"
            ) from e

    def run_agent(self, query: str) -> dict:
        """
        Ejecuta el loop agéntico ReAct para una query.

        Diseño stateless-per-iteration:
        - Cada iteración es una llamada independiente a Ollama.
        - El contexto acumulado (searched_terms, grep_findings) se pasa
          como texto plano en el mensaje de usuario, no como historial.
        - No hay señal de stop (EVIDENCIA_FINAL) — el loop corre
          max_iterations y toda evidencia acumulada se usa.

        La evidencia se acumula de llamadas a read_file y read_lines.
        grep_corpus provee navegación — sus líneas de resultado se usan
        para orientar las llamadas de read_lines en iteraciones siguientes.

        Args:
            query: Pregunta de investigación.
        Returns:
            Dict con evidence_raw, iterations, tool_calls, think_blocks, stopped_reason.
        """
        evidence_raw: list[dict] = []
        tool_calls: list[dict] = []
        think_blocks: list[dict] = []
        searched_terms: list[str] = []
        grep_findings: list[str] = []
        last_result: str = ""
        seen_reads: set[tuple] = set()
        stopped_reason = "max_iterations"

        for i in range(self.max_iterations):
            if len(evidence_raw) >= self.max_evidence_chunks:
                stopped_reason = "max_evidence"
                break

            user_msg = _build_user_message(
                query, i, searched_terms, last_result, len(evidence_raw), grep_findings
            )
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ]

            t0 = time.time()
            try:
                response = _call_ollama(
                    self.base_url, self.model, messages,
                    self.temperature, self.timeout,
                )
            except RuntimeError as e:
                logger.error("Ollama falló en iteración {}: {}", i, e)
                stopped_reason = "timeout"
                break

            elapsed = time.time() - t0

            think = _extract_think_block(response)
            if think:
                think_blocks.append({"iteration": i, "think": think})

            action = _extract_action(response)
            if action is None:
                logger.debug(
                    "Iteración {}: no se detectó acción válida (elapsed={:.1f}s)", i, elapsed
                )
                logger.debug("Respuesta raw (primeros 300 chars): {}", response[:300])
                # Sin historial que corromper — simplemente saltamos la iteración
                last_result = f"[iter {i}: no valid action detected]"
                continue

            tool_name = action["tool"]
            args = action["args"]

            result = self.tools.dispatch(tool_name, args)
            last_result = result

            tool_calls.append({
                "iteration": i,
                "tool": tool_name,
                "args": args,
                "result_preview": result[:300],
                "elapsed_s": round(elapsed, 2),
            })

            logger.debug(
                "Iteración {} | tool={} | args={} | evidence_acc={} | elapsed={:.1f}s",
                i, tool_name, args, len(evidence_raw), elapsed,
            )

            if tool_name == "grep_corpus":
                pattern = args.get("pattern", "")
                if pattern:
                    searched_terms.append(pattern)
                if result and not result.startswith("Error") and not result.startswith("No se encontraron"):
                    # Guardar líneas del grep para próximas iteraciones
                    new_lines = result.splitlines()
                    grep_findings.extend(new_lines)
                    grep_findings = grep_findings[-40:]  # mantener últimas 40 líneas

            elif tool_name in ("read_file", "read_lines"):
                if result and not result.startswith("Error"):
                    key = (
                        tool_name,
                        args.get("paper_id", ""),
                        args.get("section", ""),
                        str(args.get("start", "")),
                        str(args.get("end", "")),
                    )
                    if key not in seen_reads:
                        seen_reads.add(key)
                        evidence_raw.append({
                            "tool": tool_name,
                            "args": args,
                            "text": result,
                        })

        logger.info(
            "Agente terminó | iterations={} | evidence={} | tool_calls={} | reason={}",
            min(i + 1, self.max_iterations), len(evidence_raw), len(tool_calls), stopped_reason,
        )

        return {
            "evidence_raw": evidence_raw,
            "iterations": min(i + 1, self.max_iterations),
            "tool_calls": tool_calls,
            "think_blocks": think_blocks,
            "stopped_reason": stopped_reason,
        }
