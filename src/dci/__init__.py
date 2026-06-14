"""
src/dci/__init__.py
===================
Módulo DCI (Document-Centric Intelligence) para RAG 3.

Exporta las tres clases principales del pipeline de retrieval agéntico:
  - DCITools: herramientas de búsqueda lexical sobre el corpus de texto plano
  - DCIAgent: loop ReAct con DeepSeek-R1 via Ollama
  - EvidenceCollector: formatea la evidencia agéntica al esquema estándar de chunk
"""

from src.dci.tools import DCITools
from src.dci.agent import DCIAgent
from src.dci.evidence_collector import EvidenceCollector

__all__ = ["DCITools", "DCIAgent", "EvidenceCollector"]
