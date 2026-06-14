"""
tools.py
========
Herramientas de búsqueda lexical del agente DCI sobre el corpus de texto plano.

Estas herramientas reemplazan el retrieval vectorial de RAG 1 y 2 con búsqueda
directa sobre archivos .txt organizados en corpus/{paper_id}/{seccion}.txt.
El agente DeepSeek-R1 llama estas herramientas en un loop ReAct para encontrar
evidencia relevante a una query.

Por qué búsqueda lexical: DCI (Document-Centric Intelligence) propone que un
agente razonador puede navegar y filtrar texto plano con mayor precisión que el
retrieval vectorial one-shot, especialmente para términos técnicos y cifras
específicas que los embeddings comprimen.

Cada herramienta captura excepciones y retorna mensajes de error como string —
el agente puede leer el error y adaptar su estrategia sin colapsar el loop.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from loguru import logger


class DCITools:
    """
    Conjunto de herramientas de búsqueda sobre el corpus de texto plano.

    Implementa las cinco herramientas que el agente DCI puede invocar.
    Toda la configuración (ruta del corpus, límites) viene del config.yaml
    bajo la sección rag3: para no hardcodear nada.
    """

    def __init__(self, config: dict):
        """
        Inicializa las herramientas con los parámetros del corpus.

        Args:
            config: Configuración global del proyecto (config.yaml).
        """
        rag3_cfg = config.get("rag3", {})
        corpus_cfg = rag3_cfg.get("corpus", {})
        agent_cfg = rag3_cfg.get("agent", {})

        corpus_path = corpus_cfg.get("path", "./corpus")
        self.corpus_dir = Path(corpus_path).resolve()
        self.max_results: int = agent_cfg.get("max_results_per_grep", 20)
        self.context_lines: int = agent_cfg.get("context_window_lines", 5)

        logger.debug(
            "DCITools inicializado | corpus={} | max_results={} | context_lines={}",
            self.corpus_dir, self.max_results, self.context_lines,
        )

    # ── Herramientas públicas ─────────────────────────────────────────────────

    def grep_corpus(self, pattern: str = "", section_filter: str | None = None,
                    query: str = "", keyword: str = "", term: str = "", **_kw) -> str:
        # Normaliza nombres de argumento alternativos que distintos modelos pueden generar
        pattern = pattern or query or keyword or term
        """
        Busca un patrón en el corpus completo o en secciones específicas.

        Retorna los matches en formato estructurado para que el agente pueda
        extraer directamente paper_id, section y line_number sin parsear paths
        completos ni separadores de contexto de grep.

        Args:
            pattern: Término o frase a buscar (case-insensitive).
            section_filter: Si se especifica, busca solo en esa sección.
        Returns:
            Matches en formato:
              [paper_id=NAME | section=SECTION | line=N] matched text
        """
        if not self.corpus_dir.exists():
            return f"Error: corpus no encontrado en {self.corpus_dir}. Corre: python main.py --pipeline rag3 --mode ingest"

        include_glob = f"{section_filter}.txt" if section_filter else "*.txt"

        # Sin -C (context) para obtener output limpio sin separadores -- y líneas de contexto
        cmd = [
            "grep", "-r", "-n", "-i",
            f"--include={include_glob}",
            pattern,
            str(self.corpus_dir),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, encoding="utf-8"
            )
            output = result.stdout.strip()
            if not output:
                section_hint = f" in section '{section_filter}'" if section_filter else ""
                return f"No matches found for '{pattern}'{section_hint}"

            # Parsear el output de grep y reformatear como structured snippets
            structured: list[str] = []
            for raw_line in output.splitlines():
                if len(structured) >= self.max_results:
                    structured.append(f"... (truncated to {self.max_results} results)")
                    break
                # Formato grep: /abs/path/paper_id/section.txt:lineno:text
                try:
                    # Separar path del resto en el primer ':'
                    path_part, rest = raw_line.split(":", 1)
                    lineno_str, text = rest.split(":", 1)
                    path = Path(path_part)
                    paper_id = path.parent.name      # directorio = paper_id
                    section = path.stem               # nombre sin .txt = section
                    structured.append(
                        f"[paper_id={paper_id} | section={section} | line={lineno_str}] {text.strip()}"
                    )
                except (ValueError, AttributeError):
                    # Línea no parseable (separadores --, líneas de contexto) — ignorar
                    continue

            if not structured:
                return f"No matches found for '{pattern}'"

            header = f"Search results for '{pattern}' ({len(structured)} matches):\n"
            return header + "\n".join(structured)

        except subprocess.TimeoutExpired:
            return "Error: grep exceeded 30s timeout"
        except FileNotFoundError:
            return "Error: 'grep' command not available"
        except Exception as e:
            return f"Error in grep_corpus: {e}"

    def read_file(self, paper_id: str, section: str) -> str:
        """
        Lee el contenido completo de una sección de un paper.

        Útil cuando el agente quiere leer el texto completo de una sección
        después de encontrar matches con grep_corpus.

        Args:
            paper_id: Identificador del paper (nombre del directorio en corpus/).
            section: Nombre de la sección (sin extensión .txt).
        Returns:
            Contenido completo de la sección como string.
        """
        file_path = self.corpus_dir / paper_id / f"{section}.txt"
        if not file_path.exists():
            paper_dir = self.corpus_dir / paper_id
            if paper_dir.exists():
                available = [p.stem for p in paper_dir.glob("*.txt")]
                return (
                    f"Error: sección '{section}' no existe para paper_id='{paper_id}'. "
                    f"Secciones disponibles: {', '.join(sorted(available)) or 'ninguna'}"
                )
            return (
                f"Error: paper_id='{paper_id}' no encontrado en el corpus. "
                "Usa list_papers() para ver los disponibles."
            )

        try:
            text = file_path.read_text(encoding="utf-8").strip()
            return f"[{paper_id} / {section}]\n{text}"
        except Exception as e:
            return f"Error leyendo {file_path}: {e}"

    def read_lines(self, paper_id: str = "", section: str = "", start: int = 1, end: int = 20,
                   line: int = 0, **_kw) -> str:
        # line=N es el alias que el agente puede recibir del grep estructurado
        if line and not (start != 1 or end != 20):
            start = max(1, line - 5)
            end   = line + 15
        """
        Lee líneas específicas de una sección con ventana de contexto configurable.

        Útil para extraer el fragmento exacto alrededor de un match de grep,
        sin leer toda la sección. Las líneas son 1-indexadas.

        Args:
            paper_id: Identificador del paper.
            section: Nombre de la sección (sin extensión .txt).
            start: Primera línea a leer (1-indexado).
            end: Última línea a leer (1-indexado, inclusivo).
        Returns:
            Líneas seleccionadas con contexto adicional (context_window_lines).
        """
        file_path = self.corpus_dir / paper_id / f"{section}.txt"
        if not file_path.exists():
            return f"Error: {paper_id}/{section}.txt no existe"

        try:
            all_lines = file_path.read_text(encoding="utf-8").splitlines()
            total = len(all_lines)

            ctx = self.context_lines
            s = max(0, start - 1 - ctx)
            e = min(total, end + ctx)

            selected = all_lines[s:e]
            header = f"[{paper_id} / {section} — líneas {s + 1}-{s + len(selected)} de {total}]"
            return header + "\n" + "\n".join(selected)

        except Exception as e:
            return f"Error en read_lines({paper_id}, {section}, {start}, {end}): {e}"

    def list_papers(self) -> str:
        """
        Lista todos los paper_ids disponibles en el corpus.

        Returns:
            String con los nombres de directorio de cada paper, uno por línea.
        """
        if not self.corpus_dir.exists():
            return f"Error: corpus no encontrado en {self.corpus_dir}. Corre: python main.py --pipeline rag3 --mode ingest"

        papers = sorted(p.name for p in self.corpus_dir.iterdir() if p.is_dir())
        if not papers:
            return "El corpus está vacío. Corre: python main.py --pipeline rag3 --mode ingest"

        return f"Papers disponibles ({len(papers)}):\n" + "\n".join(papers)

    def list_sections(self, paper_id: str) -> str:
        """
        Lista las secciones disponibles para un paper específico.

        Args:
            paper_id: Identificador del paper (nombre del directorio).
        Returns:
            String con los nombres de sección disponibles.
        """
        paper_dir = self.corpus_dir / paper_id
        if not paper_dir.exists():
            return (
                f"Error: paper_id='{paper_id}' no encontrado. "
                "Usa list_papers() para ver los disponibles."
            )

        sections = sorted(p.stem for p in paper_dir.glob("*.txt"))
        if not sections:
            return f"No hay secciones para paper_id='{paper_id}'"

        return f"Secciones de '{paper_id}':\n" + "\n".join(sections)

    def dispatch(self, tool_name: str, args: dict) -> str:
        """
        Despacha una llamada a herramienta por nombre.

        Usado por DCIAgent para ejecutar acciones parseadas del LLM.

        Args:
            tool_name: Nombre de la herramienta.
            args: Argumentos como dict.
        Returns:
            Resultado de la herramienta como string.
        """
        dispatch_map = {
            # nombres canónicos
            "grep_corpus":   self.grep_corpus,
            "read_file":     self.read_file,
            "read_lines":    self.read_lines,
            "list_papers":   lambda **kw: self.list_papers(),
            "list_sections": self.list_sections,
            # aliases que modelos alternativos (Granite, etc.) pueden generar
            "search_corpus": self.grep_corpus,
            "search":        self.grep_corpus,
            "grep":          self.grep_corpus,
            "read_paper":    self.read_lines,
            "read_section":  self.read_lines,
            "read":          self.read_lines,
        }
        fn = dispatch_map.get(tool_name)
        if fn is None:
            known = ", ".join(dispatch_map.keys())
            return f"Error: herramienta '{tool_name}' no reconocida. Disponibles: {known}"
        try:
            return fn(**args)
        except TypeError as e:
            return f"Error: argumentos incorrectos para '{tool_name}': {e}"
