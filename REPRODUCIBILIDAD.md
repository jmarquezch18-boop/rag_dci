# Guía de Reproducibilidad — SAIL-RAG Comparative Study

Esta guía permite reproducir todos los experimentos del estudio desde cero.

---

## Requisitos del Sistema

| Componente | Mínimo | Recomendado |
|---|---|---|
| GPU VRAM | 4 GB (solo RAG 1-2.x) | 16 GB (RAG 3.x con DeepSeek-R1 14B) |
| RAM | 16 GB | 32 GB |
| Disco | 20 GB | 50 GB |
| Python | 3.10+ | 3.13 |
| CUDA | 11.8+ | 12.1+ |
| OS | Linux | Ubuntu 22.04+ |

**Servicios externos necesarios**:
- Cuenta de Groq (gratuita): [console.groq.com](https://console.groq.com)
- Ollama instalado localmente (para RAG 3.x)

---

## 1. Configuración del Entorno

### 1.1 Clonar el repositorio

```bash
git clone <repo-url>
cd rag_dci
```

### 1.2 Instalar dependencias Python

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 1.3 Configurar API keys

Crear archivo `.env` en la raíz del proyecto:

```bash
cp .env.example .env   # si existe template
# O crear directamente:
```

Contenido mínimo de `.env`:

```ini
# Groq API Key — obtener en console.groq.com (plan gratuito suficiente para pruebas)
GROQ_API_KEY=gsk_xxxxxxxxxxxx

# Opcional: claves adicionales para mayor throughput (rotación automática)
# GROQ_API_KEY2=gsk_yyyyyyyyyyyy
# GROQ_API_KEY3=gsk_zzzzzzzzzzzz
```

> **Seguridad**: Nunca comitear `.env` al repositorio. El archivo `.gitignore` ya lo excluye.

### 1.4 Instalar Ollama (solo para RAG 3.x)

```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Verificar instalación
ollama --version
```

Descargar modelos necesarios:

```bash
# Para RAG 3 y RAG 3.2 (DeepSeek-R1 14B — requiere ~8GB VRAM)
ollama pull deepseek-r1:14b

# Para RAG 3.1 (Granite 3.1 8B — requiere ~5GB VRAM)
ollama pull granite3.1-dense:8b
```

---

## 2. Preparación del Dataset

El dataset PeerQA se descarga automáticamente desde HuggingFace Hub durante el primer `ingest`.

Si quieres pre-descargarlo manualmente:

```python
from datasets import load_dataset
ds = load_dataset("mteb/PeerQA", cache_dir="./data/peerqa")
```

---

## 3. Ejecutar los Pipelines

Todos los pipelines se ejecutan con `main.py`. La estructura general es:

```bash
python main.py --pipeline <nombre> --mode <modo>
```

### 3.1 RAG 1 — Baseline Dense Retrieval

```bash
# Paso 1: Indexar chunks en Qdrant
python main.py --pipeline rag1 --mode ingest

# Paso 2: Evaluar retrieval + generación
python main.py --pipeline rag1 --mode evaluate

# Paso 3 (opcional): Visualizar resultados
python main.py --pipeline rag1 --mode visualize
```

**Tiempo estimado**: ~15 min (ingest con GPU) + ~20 min (evaluate con 136 queries)

**Salida**: `results/rag1/metrics_post_llm.json`

### 3.2 RAG 2.1 — Oracle Only (sin reranker)

```bash
# RAG 2.1 reutiliza el índice vectorial de RAG 1 (no re-indexar)
python main.py --pipeline rag2_1 --mode evaluate
```

**Tiempo estimado**: ~20 min

**Salida**: `results/rag2_1/metrics_post_llm.json`

### 3.3 RAG 2.3 — Reranker Only (sin oracle)

```bash
python main.py --pipeline rag2_3 --mode evaluate
```

**Tiempo estimado**: ~25 min (reranker añade latencia por batch)

**Salida**: `results/rag2_3/metrics_post_llm.json`

### 3.4 RAG 2.2 — SAIL-RAG Completo (Oracle + Reranker)

RAG 2.2 usa un proceso de dos rondas (activo learning):

```bash
# Paso 1: Ingestar y crear chunks vectoriales
python main.py --pipeline rag2 --mode ingest

# Paso 2: Evaluación ronda 0 (sin entrenamiento previo)
python main.py --pipeline rag2 --mode evaluate_r0

# Paso 3: Entrenar el modelo de curación con feedback
python main.py --pipeline rag2 --mode train

# Paso 4: Evaluación ronda N (con modelo entrenado)
python main.py --pipeline rag2 --mode evaluate_rn

# Paso 5 (opcional): Visualizar curva de aprendizaje
python main.py --pipeline rag2 --mode visualize
```

**Tiempo estimado**: ~45 min total

**Salida**: `results/rag2/round_0/` y `results/rag2/round_n/`

### 3.5 RAG 3 — Agente DCI + Corpus Plano

```bash
# Paso 1: Iniciar servidor Ollama (en terminal separada)
ollama serve &

# Paso 2: Construir corpus de texto plano
python main.py --pipeline rag3 --mode ingest

# Paso 3: Ejecutar agente DCI + evaluación
python main.py --pipeline rag3 --mode evaluate
```

**Tiempo estimado**: ~8-12 horas (136 queries × 5 iteraciones × ~30-120s/iter)

**Salida**: `results/rag3/metrics_dci.json`

> **Nota**: Qdrant no se usa en RAG 3. El agente busca mediante grep en el corpus de archivos de texto.

### 3.6 RAG 3.1 — Agente DCI + Corpus Docling + Granite

```bash
# Ollama debe estar corriendo con granite3.1-dense:8b descargado

# Paso 1: Construir corpus Docling (a partir de rag2_chunks.json)
python main.py --pipeline rag3_1 --mode ingest

# Paso 2: Ejecutar agente DCI con Granite
python main.py --pipeline rag3_1 --mode evaluate
```

**Tiempo estimado**: ~6-10 horas (Granite 8B es más rápido que DeepSeek 14B)

**Salida**: `results/rag3_1/metrics_dci.json`

### 3.7 RAG 3.2 — Agente DCI + Corpus Docling + DeepSeek

```bash
# RAG 3.2 reutiliza el corpus Docling construido por RAG 3.1
# Si no se ejecutó RAG 3.1, construir el corpus primero:
python main.py --pipeline rag3_1 --mode ingest

# Ejecutar agente DCI con DeepSeek-R1 14B
python main.py --pipeline rag3_2 --mode evaluate
```

**Tiempo estimado**: ~10-14 horas (DeepSeek-R1 14B es más lento, mayor calidad)

**Salida**: `results/rag3_2/metrics_dci.json`

---

## 4. Visualización de Resultados

### 4.1 Comparación final de todos los sistemas

```bash
python -m src.evaluation.visualizer_rag3_dci
```

**Salida**: `results/final_comparison_rag1_rag2_rag3.png`

### 4.2 Análisis de impacto de Docling (RAG 3 vs RAG 3.2)

```bash
python -m src.evaluation.visualizer_docling_impact
```

**Salida**: `results/docling_impact_analysis.png`

### 4.3 Comparación de faithfulness

```bash
python -m src.evaluation.visualizer_faithfulness
```

**Salida**: `results/faithfulness_comparison.png`

---

## 5. Configuración Avanzada

### 5.1 Archivo de configuración

Todas las opciones están en `config/config.yaml`:

```yaml
# Tamaño del subset (null = dataset completo)
data:
  subset_size: null    # Cambiar a 20 para pruebas rápidas

# Parámetros de chunking
chunking:
  chunk_size_tokens: 384
  chunk_overlap_tokens: 64

# Parámetros de recuperación
retrieval:
  k_candidates: 20    # Chunks candidatos antes del reranker/oracle
  top_n_final: 5      # Chunks finales para el LLM

# Oráculo RAG 2
rag2:
  oracle_similarity_threshold: 0.75    # Jaccard threshold
  oracle_containment_threshold: 0.5   # Containment threshold (RAG 3.x)
  oracle_metric: "jaccard"            # "jaccard" o "containment"

# Agente DCI RAG 3
rag3:
  ollama:
    model: "deepseek-r1:14b"
    timeout_seconds: 120
  agent:
    max_iterations: 5
    max_evidence_chunks: 10
```

### 5.2 Prueba rápida con subset pequeño

Para verificar que todo funciona sin esperar horas:

```yaml
# En config/config.yaml:
data:
  subset_size: 10   # Solo 10 queries
```

```bash
# Luego ejecutar cualquier pipeline normalmente
python main.py --pipeline rag1 --mode evaluate
```

### 5.3 Re-indexar desde cero

Si necesitas forzar la re-creación del índice vectorial:

```bash
python main.py --pipeline rag1 --mode ingest --force-reindex
```

---

## 6. Estructura del Proyecto

```
rag_dci/
├── main.py                          # Punto de entrada CLI
├── config/
│   └── config.yaml                  # Configuración global
├── src/
│   ├── pipelines/                   # Un archivo por sistema RAG
│   │   ├── rag1_baseline.py
│   │   ├── rag2_1_oracle_only.py
│   │   ├── rag2_3_reranker_only.py
│   │   ├── rag2_sail_r0.py          # RAG 2.2 ronda 0
│   │   ├── rag2_sail_rn.py          # RAG 2.2 ronda N
│   │   ├── rag3_dci_sail.py         # RAG 3 (DCI + corpus plano)
│   │   ├── rag3_1_granite_docling.py
│   │   └── rag3_2_deepseek_docling.py
│   ├── retrieval/
│   │   ├── embedder.py              # BGE-M3 wrapper
│   │   ├── indexer.py               # Qdrant indexing
│   │   └── reranker.py              # BGE-reranker-v2-m3
│   ├── curation/
│   │   └── oracle.py                # Jaccard + containment oracle
│   ├── dci/
│   │   ├── agent.py                 # Loop ReAct stateless
│   │   ├── tools.py                 # grep_corpus, read_lines, etc.
│   │   └── evidence_collector.py   # Acumulador de evidencia DCI
│   ├── generation/
│   │   └── llm.py                   # GroqLLM con rotación de claves
│   └── evaluation/
│       ├── metrics.py               # MRR, Recall, NDCG, ROUGE-L, BERTScore
│       ├── visualizer_rag3_dci.py  # Figura comparación final 7 sistemas
│       └── visualizer_docling_impact.py  # Figura impacto Docling
├── corpus/                          # Corpus plano (generado por ingest rag3)
├── corpus_docling/                  # Corpus Docling (generado por ingest rag3_1)
├── checkpoints/                     # Embeddings y chunks guardados
├── results/                         # Métricas y figuras de salida
├── data/                            # PeerQA dataset (descargado automáticamente)
├── requirements.txt
├── RESULTADOS.md                    # Documentación de métodos y resultados
└── REPRODUCIBILIDAD.md              # Este archivo
```

---

## 7. Solución de Problemas Comunes

### Error: `Ollama no está corriendo`

```bash
# Iniciar Ollama en background
ollama serve &

# Verificar que corre
curl http://localhost:11434/api/tags
```

### Error: `GROQ_API_KEY no encontrada`

```bash
# Verificar que .env existe y tiene la clave
cat .env | grep GROQ_API_KEY

# Verificar que dotenv lo carga
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('GROQ_API_KEY', 'NO ENCONTRADA')[:10])"
```

### Error: `CUDA out of memory`

Reducir batch size en `config/config.yaml`:

```yaml
hardware:
  batch_size_embed: 4    # Default: 8
  batch_size_rerank: 2   # Default: 4
  fp16: true             # Mantener en true
```

### Rate limit de Groq (429)

El sistema automáticamente rota entre claves GROQ_API_KEY, GROQ_API_KEY2, etc. Si hay rate limit persistente, añadir más claves al `.env`:

```ini
GROQ_API_KEY2=gsk_...
GROQ_API_KEY3=gsk_...
```

### DeepSeek-R1 muy lento / timeout

Aumentar el timeout en config:

```yaml
rag3:
  ollama:
    timeout_seconds: 300   # Default: 120
```

### Checkpoints corruptos

```bash
# Eliminar checkpoints y re-indexar
rm -rf checkpoints/
python main.py --pipeline rag1 --mode ingest --force-reindex
```

---

## 8. Verificación de Resultados

Para verificar que los resultados son correctos, los valores esperados de los experimentos completos son:

| Sistema | A.MRR esperado | Tolerancia |
|---|---|---|
| RAG 1 | 0.2211 | ±0.005 |
| RAG 2.1 | 0.2135 | ±0.005 |
| RAG 2.3 | 0.3008 | ±0.005 |
| RAG 2.2 | 0.4559 | ±0.010 |
| RAG 3 | 0.2819 | ±0.020 |
| RAG 3.1 | 0.1544 | ±0.030 |
| RAG 3.2 | 0.3100 | ±0.020 |

> Variaciones mayores pueden deberse a diferencias en la versión de los modelos Ollama o en el orden de rotación de claves Groq.

Para verificar rápidamente los resultados existentes:

```bash
python -c "
import json, glob
for f in sorted(glob.glob('results/*/metrics*.json')):
    d = json.load(open(f))
    agg = d.get('aggregate', {})
    mrr = agg.get('point_a', {}).get('mrr', 'N/A')
    print(f'{f}: MRR={mrr:.4f}' if isinstance(mrr, float) else f'{f}: {mrr}')
"
```

---

*Este proyecto requiere acceso a Groq API (gratuito) y Ollama para los experimentos DCI. Los modelos de embedding y reranking se descargan automáticamente desde HuggingFace Hub en el primer uso.*
