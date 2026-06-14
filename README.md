# SAIL-RAG Comparative Study

Estudio comparativo de siete variantes de sistemas **RAG (Retrieval-Augmented Generation)** para respuesta a preguntas sobre artículos científicos, evaluadas sobre el dataset **PeerQA**.

---

## Sistemas evaluados

| Sistema | Descripción | A.MRR | B.cov | C.ROUGE-L |
|---|---|---|---|---|
| **RAG 1** | Baseline: BGE-M3 + ANN + Llama-3.3-70B | 0.221 | — | 0.102 |
| **RAG 2.1** | + Oráculo Jaccard (sin reranker) | 0.214 | 0.247 | 0.069 |
| **RAG 2.3** | + Reranker cross-encoder (sin oráculo) | 0.301 | — | 0.121 |
| **RAG 2.2** | + Oráculo + Reranker (SAIL-RAG completo) | **0.456** | **0.305** | 0.119 |
| **RAG 3** | Agente DCI (DeepSeek-R1 14B) + corpus plano | 0.282 | 0.000 | 0.069 |
| **RAG 3.1** | Agente DCI (Granite 3.1 8B) + corpus Docling | 0.154 | 0.109 | 0.034 |
| **RAG 3.2** | Agente DCI (DeepSeek-R1 14B) + corpus Docling | 0.310 | 0.149 | 0.077 |

> **A** = pre-curación (retrieval puro) · **B** = post-curación (oracle coverage) · **C** = post-LLM (generación)

---

## Arquitectura

```
Pregunta ──► Embedding (BGE-M3) ──► Qdrant ANN ──► Reranker / Oracle ──► Llama-3.3-70B ──► Respuesta
                                                          ↑
                                          (RAG 3.x) Agente DCI ReAct
                                          DeepSeek-R1 14B via Ollama
                                          grep iterativo sobre corpus_docling/
```

### Hallazgos principales

- **El reranker cross-encoder es el componente individual más valioso**: +36% MRR sobre baseline, sin necesitar ground-truth (deployable).
- **Docling desbloquea la curación oracle en DCI**: sin Docling, B.cov=0 por incompatibilidad Jaccard; con Docling, B.cov=0.149.
- **DeepSeek-R1 14B supera a Granite 3.1 8B en +101% MRR** para la tarea DCI bajo las mismas condiciones de corpus.
- **RAG 3.2 (DCI+Docling) alcanza MRR comparable al reranker** (0.310 vs 0.301) sin embedding vectorial previo.

---

## Figuras

| Figura | Descripción |
|---|---|
| `results/final_comparison_rag1_rag2_rag3.png` | Comparación completa de los 7 sistemas |
| `results/docling_impact_analysis.png` | Impacto de Docling en el agente DCI (4 paneles) |
| `results/faithfulness_comparison_full.png` | Fidelidad al contexto por sistema |

---

## Documentación

- **[RESULTADOS.md](RESULTADOS.md)** — Métodos matemáticos, métricas con justificación, tablas completas de resultados, interpretaciones y limitaciones.
- **[REPRODUCIBILIDAD.md](REPRODUCIBILIDAD.md)** — Guía paso a paso para reproducir todos los experimentos desde cero.

---

## Inicio rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar API key de Groq en .env
echo "GROQ_API_KEY=gsk_..." > .env

# 3. Ejecutar baseline
python main.py --pipeline rag1 --mode ingest
python main.py --pipeline rag1 --mode evaluate

# 4. Ejecutar con reranker (mejor sistema deployable)
python main.py --pipeline rag2_3 --mode evaluate

# 5. Comparación visual
python -m src.evaluation.visualizer_rag3_dci
```

Para RAG 3.x (agente DCI) se necesita **Ollama** con `deepseek-r1:14b`. Ver [REPRODUCIBILIDAD.md](REPRODUCIBILIDAD.md) para instrucciones completas.

---

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Embedding | [BGE-M3](https://huggingface.co/BAAI/bge-m3) (BAAI, 568M params) |
| Reranker | [BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| Vector DB | [Qdrant](https://qdrant.tech/) (on-disk, HNSW) |
| Generación | [Llama-3.3-70B-Versatile](https://groq.com/) via Groq API |
| Agente DCI | [DeepSeek-R1 14B](https://ollama.com/library/deepseek-r1) via Ollama |
| Dataset | [PeerQA](https://huggingface.co/datasets/mteb/PeerQA) (MTEB) |
| PDF parsing | [Docling](https://github.com/DS4SD/docling) (IBM Research) |

---

## Requisitos

- Python 3.10+
- GPU con ≥4 GB VRAM (embedding/reranker) · ≥8 GB VRAM para RAG 3.x (DeepSeek-R1 14B)
- Cuenta Groq gratuita: [console.groq.com](https://console.groq.com)
- Ollama (solo RAG 3.x): [ollama.ai](https://ollama.ai)
