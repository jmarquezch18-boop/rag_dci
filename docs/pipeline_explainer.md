# SAIL-RAG Pipeline Explainer

Este documento explica cómo funcionan los dos pipelines RAG del proyecto, cómo se evalúan y qué significa cada métrica.

---

## Contexto general

**RAG** (Retrieval-Augmented Generation) es un sistema que, dado una pregunta, busca los fragmentos de texto más relevantes en una base de conocimiento y se los pasa a un LLM para que genere una respuesta fundamentada en esa evidencia.

Este proyecto compara dos versiones de RAG sobre el dataset **PeerQA** (preguntas sobre papers académicos):

| | RAG 1 | RAG 2 |
|---|---|---|
| Tipo | Baseline puro | SAIL-RAG (con curación y aprendizaje activo) |
| Chunking | Simple (tamaño fijo) | Por secciones de paper (Abstract, Results, etc.) |
| Retrieval | Solo embedding (bi-encoder) | Embedding + Reranker (cross-encoder) |
| Curación | Ninguna | Oráculo que aprueba/rechaza chunks |
| Aprendizaje | Estático | Fine-tuning activo del reranker |
| LLM | Groq llama-3.3-70b | Groq llama-3.3-70b |

---

## Dataset: PeerQA

- Preguntas sobre papers científicos con respuestas anotadas
- Cada pregunta tiene `answer_evidence_mapped`: los IDs de los chunks exactos que contienen la evidencia
- Esto permite evaluar si el sistema recuperó los chunks correctos (ground truth)
- Se usa un subconjunto estratificado de **136–200 queries** para evaluación

---

## RAG 1 — Baseline

### Cómo funciona

```
Pregunta
   |
   v
[BGE-M3]  <-- modelo de embedding (bi-encoder, GPU)
   |  genera vector de 1024 dimensiones
   v
[Qdrant]  <-- búsqueda ANN (Approximate Nearest Neighbor) en memoria
   |  devuelve top-5 chunks más similares
   v
[Groq LLM]  <-- llama-3.3-70b-versatile, temperatura 0
   |  genera respuesta con los chunks como contexto
   v
Respuesta generada
```

### Chunking (preparación del índice)

Los papers se dividen en fragmentos de **384 tokens** con **64 tokens de overlap** entre fragmentos consecutivos. No se tiene en cuenta la estructura del paper (sección, título, etc.) — son trozos de texto planos.

### Ingest (indexación)

1. Descarga PeerQA desde HuggingFace
2. Divide cada paper en chunks
3. Genera embeddings con BGE-M3 en GPU (fp16, ~0.9 GB VRAM)
4. Guarda embeddings y chunks en checkpoints (no se re-indexa si ya existen)

### Evaluate

1. Carga checkpoints
2. Reconstruye Qdrant en memoria desde los embeddings
3. Para cada query: embebe la pregunta → busca top-5 → genera respuesta con Groq
4. Calcula métricas de retrieval y generación
5. Guarda todo en `results/rag1/metrics_post_llm.json`

### Evaluación — 2 Puntos de medición

**Punto A** (pre-LLM, mide la calidad del retrieval):
- ¿Están los chunks correctos entre los top-K recuperados?

**Punto B** (post-LLM, mide la calidad de la respuesta generada):
- ¿Qué tan parecida es la respuesta al texto de referencia?

---

## RAG 2 — SAIL-RAG

RAG 2 extiende RAG 1 añadiendo tres capas: **reranker**, **oráculo de curación** y **aprendizaje activo**.

### Cómo funciona

```
Pregunta
   |
   v
[BGE-M3]  <-- embedding de la query (GPU, luego se descarga)
   |
   v
[Qdrant]  <-- búsqueda ANN, devuelve top-20 candidatos
   |
   v
[BGE-Reranker-v2-m3]  <-- cross-encoder: puntúa cada par (query, chunk)
   |  reordena los 20 candidatos, se queda con top-7
   v
[CurationOracle]  <-- simula curación humana con ground truth
   |  aprueba chunks que contienen evidencia real, rechaza el resto
   |  genera tripletas (query, chunk_positivo, chunk_negativo) para entrenamiento
   v
[Groq LLM]  <-- genera respuesta con los chunks APROBADOS como contexto
   |  (si 0 aprobados, usa fallback: top-3 rerankeados)
   v
Respuesta generada
```

### Chunking con secciones (Docling)

A diferencia de RAG 1, los papers se parsean con **Docling** que detecta la estructura:
- Abstract, Introduction, Related Work, Methodology, Results, Discussion, Conclusion, etc.

Cada chunk sabe a qué sección pertenece. Esto permite filtrar por sección si se desea (ej: buscar solo en "Results").

### Gestión de VRAM (RTX 3050 4GB)

BGE-M3 y el reranker **no pueden estar en GPU simultáneamente** (4 GB limitado). El pipeline los carga y descarga explícitamente:

```
Fase 1: Cargar BGE-M3 → embeber todas las queries → DESCARGAR BGE-M3
Fase 2: Cargar Reranker → reranking de todos los resultados → DESCARGAR Reranker
Fase 3: Groq (API, no usa GPU)
Fase 4: BERTScore + Faithfulness en CPU (no necesita GPU)
```

### El Oráculo de Curación

El `CurationOracle` simula lo que haría un humano revisando los chunks recuperados:

- **Aprueba** un chunk si su overlap con el ground truth (Jaccard similarity ≥ 0.75) indica que contiene evidencia real
- **Rechaza** chunks que no son relevantes
- Por cada par (aprobado, rechazado) genera una **tripleta** para entrenamiento: `{query, chunk_positivo, chunk_negativo}`

Esto reduce el contexto enviado al LLM — en lugar de pasar 7 chunks, solo pasa los aprobados.

### Aprendizaje Activo (Active Learning)

El bucle de aprendizaje activo **mejora el reranker** con las decisiones del oráculo:

```
Para cada query:
  - El oráculo genera tripletas
  - Se acumulan en FeedbackStore (persistido en disco)
  - Cuando se acumulan >= 50 tripletas:
      - Fine-tuning contrastivo del reranker (CPU, Adafactor)
      - Guarda checkpoint reranker_round_N.pt
      - Registra MRR y loss en active_learning_curve.json
      - Vacía el store para el siguiente ciclo
```

**Loss contrastiva (MarginRankingLoss)**: el score del chunk positivo debe superar al negativo por al menos 0.5 de margen.

**Fine-tuning en CPU**: para evitar OOM en 4 GB VRAM. Usa Adafactor (optimizador de bajo consumo de RAM) en lugar de AdamW.

### Evaluación — 3 Puntos de medición

**Punto A** (pre-curación): ¿Recuperó bien el reranker?

**Punto B** (post-curación): ¿Qué pasó después del filtro del oráculo?

**Punto C** (post-LLM): ¿Qué tan buena es la respuesta final?

---

## Métricas explicadas

Todas las métricas van de **0.0 a 1.0** (1.0 = perfecto).

### Métricas de Retrieval (Puntos A y B)

| Métrica | Qué mide | 1.0 significa |
|---|---|---|
| **MRR** (Mean Reciprocal Rank) | Posición promedio del primer chunk correcto | El chunk correcto siempre es el #1 |
| **Recall@K** | ¿Está el chunk correcto entre los top-K? | Siempre está en los primeros K resultados |
| **Precision@5** | De los top-5, ¿qué fracción son relevantes? | Los 5 resultados son todos correctos |
| **NDCG@5** | Ranking de relevancia ponderado por posición | Los chunks más relevantes siempre están primero |
| **Coverage** (solo RAG 2) | Fracción del ground truth capturada tras curación | Todos los chunks de evidencia fueron aprobados |

### Métricas de Generación (Punto B/C)

| Métrica | Qué mide | Nota |
|---|---|---|
| **ROUGE-L** | Overlap de palabras entre respuesta generada y referencia | Bajo es normal en QA abierta — el modelo usa sus propias palabras |
| **BERTScore-F1** | Similitud semántica (significado) entre respuesta y referencia | 0.85 es muy bueno — las respuestas tienen el significado correcto aunque usen palabras diferentes |
| **Faithfulness** (solo RAG 2) | ¿Cada oración de la respuesta está respaldada por algún chunk aprobado? | Se calcula por coseno de embeddings, umbral 0.75. Valores bajos indican alucinación |

---

## Comandos para correr cada pipeline

### RAG 1

```bash
# 1. Indexar (solo la primera vez)
python main.py --pipeline rag1 --mode ingest

# 2. Evaluar
python main.py --pipeline rag1 --mode evaluate

# 3. Generar figuras (sin modelos, solo lee JSON)
python main.py --pipeline rag1 --mode visualize
```

### RAG 2

```bash
# 1. Indexar con secciones (solo la primera vez)
python main.py --pipeline rag2 --mode ingest

# 2. Evaluar ronda 0 (reranker base, sin fine-tuning)
python main.py --pipeline rag2 --mode evaluate_r0

# 3. Entrenar (bucle de aprendizaje activo) -- tarda ~2-4 horas en CPU
python main.py --pipeline rag2 --mode train

# 4. Evaluar ronda N (con reranker fine-tuneado)
python main.py --pipeline rag2 --mode evaluate_rn

# 5. Generar todas las figuras
python main.py --pipeline rag2 --mode visualize
```

### Ver tabla comparativa en terminal

```bash
python compare_metrics.py
```

---

## Dónde se guardan los resultados

```
sail_rag_project/
  checkpoints/
    rag1_embeddings.npy          # embeddings precalculados RAG 1
    rag1_chunks.json             # metadata de chunks RAG 1
    rag2_embeddings.npy          # embeddings precalculados RAG 2
    rag2_chunks.json             # chunks con info de sección
    reranker/
      reranker_round_1.pt        # checkpoint del reranker fine-tuneado
      reranker_round_N.pt        # checkpoint más reciente
      active_learning_curve.json # MRR y loss por ronda de fine-tuning

  results/
    rag1/
      metrics_post_llm.json      # todas las métricas RAG 1
      figures/                   # 5 gráficas PNG

    rag2/
      round_0/
        metrics_pre_curation.json   # Punto A
        metrics_post_curation.json  # Punto B
        metrics_post_llm.json       # Puntos A+B+C completos
        figures/                    # 5 gráficas PNG

      round_n/
        metrics_post_llm.json       # igual pero con reranker fine-tuneado
        figures/                    # 6 gráficas (incluye curva de aprendizaje)

    comparison_rag1_vs_rag2.png  # gráfica comparativa final
```

---

## Resumen visual del flujo completo

```
                        RAG 1                          RAG 2
                        -----                          -----

INGEST          Chunks planos (384 tok)        Chunks con secciones (Docling)
                       |                                  |
                  BGE-M3 embed                      BGE-M3 embed
                       |                                  |
                    Qdrant                             Qdrant
                       
EVALUATE        Query -> BGE-M3 embed          Query -> BGE-M3 embed
                       |                                  |
                  Qdrant top-5                     Qdrant top-20
                       |                                  |
                       |                          Reranker top-7   <-- NUEVO
                       |                                  |
                       |                         Oracle: aprueba/rechaza  <-- NUEVO
                       |                                  |
                  Groq LLM                           Groq LLM
                       |                                  |
                  Respuesta                          Respuesta

METRICAS        Punto A: MRR, Recall@K         Punto A: MRR, Recall@K, NDCG@5
                Punto B: ROUGE-L, BERTScore     Punto B: MRR, Coverage, NDCG@5
                                                Punto C: ROUGE-L, BERTScore, Faithfulness

MEJORA          Ninguna (estatico)              Fine-tuning del reranker  <-- NUEVO
                                                cada 50 tripletas (activo)
```

---

## Por qué los scores de generación son bajos

ROUGE-L de 0.18 puede parecer bajo, pero es normal en este tipo de tarea:

- PeerQA tiene respuestas largas y técnicas con terminología específica
- El LLM responde correctamente **en significado** pero raramente usa las mismas palabras exactas que la referencia → BERTScore alto (0.85), ROUGE-L bajo
- Faithfulness bajo (0.22) indica que el modelo a veces genera información que no está explícitamente en los chunks recuperados (alucinación o conocimiento propio del modelo)
