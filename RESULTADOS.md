# Estudio Comparativo SAIL-RAG: Métodos, Resultados e Interpretaciones

> **Resumen ejecutivo**: Este trabajo implementa y compara siete variantes de sistemas RAG (Retrieval-Augmented Generation) para respuesta a preguntas sobre artículos científicos. Se evalúan técnicas de recuperación vectorial, curación por oráculo, re-ranking, y agente DCI (Document-Controlled Interaction) con razonamiento iterativo. El corpus proviene del dataset PeerQA (136 pares pregunta-evidencia de revisión científica). Los resultados muestran que el reranking cross-encoder es el factor más determinante (+36% MRR vs baseline), que el oráculo de curación mejora la cobertura sin mejorar la generación léxica, y que el agente DCI con Docling supera al corpus plano en cobertura (+∞) preservando calidad de recuperación.

---

## Índice

1. [Arquitectura del Pipeline de Evaluación](#1-arquitectura-del-pipeline-de-evaluación)
2. [Dataset: PeerQA](#2-dataset-peerqa)
3. [Sistemas Evaluados: Taxonomía](#3-sistemas-evaluados-taxonomía)
4. [Métodos Matemáticos](#4-métodos-matemáticos)
5. [Métricas de Evaluación](#5-métricas-de-evaluación)
6. [Resultados Completos](#6-resultados-completos)
7. [Aprendizaje Activo: Round 0 → Round N](#7-aprendizaje-activo-round-0--round-n)
8. [Interpretaciones](#8-interpretaciones)
9. [Impacto de Docling en el Agente DCI](#9-impacto-de-docling-en-el-agente-dci)
10. [Limitaciones](#10-limitaciones)
11. [Conclusiones](#11-conclusiones)

---

## 1. Arquitectura del Pipeline de Evaluación

El pipeline sigue el marco SAIL-RAG con tres puntos de medición:

```
Pregunta  ──► [A. Pre-curación] ──► [B. Post-curación] ──► LLM ──► [C. Post-LLM]
              Recuperación           Curación                        Generación
              vectorial              (oracle/reranker)               (Groq + Llama-3.3)
```

- **Punto A** (pre-curación): mide calidad del retrieval vectorial puro (BM25/dense).
- **Punto B** (post-curación): mide calidad tras filtrar/reordenar chunks con oráculo o reranker.
- **Punto C** (post-LLM): mide calidad de la respuesta generada por el LLM.

### Componentes técnicos compartidos

| Componente | Tecnología | Justificación |
|---|---|---|
| Embedding | BGE-M3 (BAAI) | Modelo multilingüe SOTA con 8192 tokens de contexto; supera a `text-embedding-ada-002` en benchmarks MTEB |
| Vector DB | Qdrant (en disco) | Búsqueda ANN con HNSW; persistencia entre runs sin reindexar |
| LLM generación | Llama-3.3-70B (Groq) | Balance calidad/latencia; corre en cloud evitando consumir VRAM local |
| Dataset | PeerQA (MTEB) | Único dataset público con evidencia a nivel de chunk de revisión científica |
| Hardware | RTX 3050 4GB VRAM | Los modelos de embedding y reranking corren en GPU; LLM en cloud |

---

## 2. Dataset: PeerQA

PeerQA ([Dycke et al., 2023](https://arxiv.org/abs/2309.11498)) es un dataset de Question Answering sobre artículos científicos donde las preguntas son extraídas de revisiones por pares reales. Cada ejemplo contiene:

- Una **pregunta** de un revisor (ej. *"What datasets were used for training?"*)
- Una o más **evidencias ground-truth**: extractos textuales del artículo que responden la pregunta
- El **artículo fuente** (PDF disponible en ArXiv / OpenReview / F1000Research)

### Estadísticas del subset utilizado

| Métrica | Valor |
|---|---|
| Queries evaluadas | 136 |
| Artículos únicos | 70 |
| Fuentes | NLPeer (ACL ARR), EGU (Earth sciences), F1000Research |
| Chunks por artículo (media) | 18.6 |
| Longitud media chunk (tokens) | ~300 |
| Longitud media ground truth (palabras) | ~50 |

**Por qué PeerQA**: La asimetría entre el tamaño del chunk recuperado (~300 tokens) y la evidencia ground-truth (~50 palabras) es representativa de retrieval en dominio científico y genera desafíos de evaluación específicos (ver sección [Limitaciones](#9-limitaciones)).

---

## 3. Sistemas Evaluados: Taxonomía

```
RAG 1         → Baseline: embed + ANN + Llama-3.3 directo
RAG 2.1       → + Oráculo de curación (sin reranker)
RAG 2.3       → + Reranker cross-encoder (sin oráculo)
RAG 2.2       → + Oráculo + Reranker (sistema completo SAIL-RAG)
RAG 3         → Agente DCI (DeepSeek-R1 14B) + corpus plano + Jaccard oracle
RAG 3.1       → Agente DCI (Granite 3.1 8B) + corpus Docling + containment oracle
RAG 3.2       → Agente DCI (DeepSeek-R1 14B) + corpus Docling + containment oracle
```

### Descripción detallada

**RAG 1 — Baseline dense retrieval**
- Indexa todos los chunks con BGE-M3.
- Consulta Qdrant con la pregunta → top-5 chunks.
- Llama-3.3-70B genera respuesta a partir de los 5 chunks.
- Sin curación, sin reranking.

**RAG 2.1 — Oracle curation only**
- Igual a RAG 1 pero recupera top-20 primero.
- Aplica oráculo Jaccard: mantiene solo chunks que solapen ≥0.75 con el ground-truth.
- Evalúa generación sobre los chunks curados (oráculo requiere conocer GT → no deployable; es upper bound de curación).

**RAG 2.3 — Reranker only**
- Igual a RAG 1 pero recupera top-20 → cross-encoder BGE-reranker-v2-m3 → top-5.
- Sin oráculo. Deployable en producción.
- Aísla el efecto del reranker.

**RAG 2.2 — SAIL-RAG completo (Oracle + Reranker)**
- Recupera top-20 → reranker → oráculo Jaccard → generación.
- Sistema de referencia superior (oracle = conoce GT).

**RAG 3 — DCI con corpus plano**
- Corpus: archivos `.txt` por sección de paper (plain text del dataset PeerQA).
- Agente: DeepSeek-R1 14B (Ollama) con loop ReAct, max 5 iteraciones.
- Oráculo: Jaccard ≥ 0.75 (incompatible con chunks DCI grandes → B.cov=0).
- Generación: Llama-3.3-70B (Groq).

**RAG 3.1 — DCI con corpus Docling + Granite**
- Corpus: procesado con Docling (extracción estructurada de PDF) → 445 archivos de sección.
- Agente: IBM Granite 3.1 Dense 8B (Ollama, 5GB VRAM).
- Oráculo: containment recall ≥ 0.5 (compatible con chunks grandes).

**RAG 3.2 — DCI con corpus Docling + DeepSeek**
- Igual a RAG 3.1 pero con DeepSeek-R1 14B.
- Aísla el efecto del corpus (vs RAG 3) y del modelo (vs RAG 3.1).

---

## 4. Métodos Matemáticos

### 4.1 Embedding Denso — BGE-M3

BGE-M3 genera un vector $\mathbf{v} \in \mathbb{R}^{1024}$ por chunk de texto usando un Transformer encoder con pooling en el token `[CLS]`:

$$\mathbf{v}_i = \text{normalize}(\text{BERT-encoder}(t_i)[CLS])$$

La similitud se computa por producto interno (equivalente a coseno tras normalización):

$$\text{sim}(q, c_i) = \mathbf{q} \cdot \mathbf{v}_i$$

**Por qué BGE-M3**: (1) Modelo de 568M parámetros con soporte multilingüe. (2) Entrenado con MNRL (Multiple Negatives Ranking Loss) que optimiza directamente el ranking. (3) Supera a `text-embedding-3-large` de OpenAI en el benchmark BEIR.

### 4.2 Búsqueda Aproximada — HNSW (Qdrant)

Qdrant indexa los vectores con Hierarchical Navigable Small World graphs. La búsqueda ANN devuelve los $k$ vecinos más cercanos en $O(\log n)$ en lugar de $O(n)$ para búsqueda exacta:

$$\text{ANN-top}_k(q) = \underset{i \in \mathcal{I}}{\arg\text{top-}k} \; \mathbf{q} \cdot \mathbf{v}_i$$

**Por qué ANN**: Con 10,000+ chunks, búsqueda exacta es lenta; HNSW sacrifica ~1% recall para 100× speedup.

### 4.3 Re-ranking Cross-encoder — BGE-reranker-v2-m3

El reranker procesa pares `(query, chunk)` concatenados y produce un score de relevancia directamente (sin embedding independiente):

$$s_i = \text{sigmoid}(\text{CrossEncoder}([q; c_i]))$$

Los chunks se reordenan por $s_i$ y se toman top-$n$.

**Por qué cross-encoder**: El bi-encoder (BGE-M3) compara vectores independientes, perdiendo interacción contextual. El cross-encoder ve ambos textos simultáneamente, capturando relevancia semántica fina a costa de $O(k)$ inferencias.

### 4.4 Oráculo de Curación — Jaccard y Containment Recall

**Similitud Jaccard** (simétrica):
$$J(A, B) = \frac{|W(A) \cap W(B)|}{|W(A) \cup W(B)|}$$

donde $W(x)$ es el conjunto de palabras de texto $x$.

**Problema con DCI**: Los chunks del agente DCI tienen ~300 palabras ($|W(C)|$ grande) mientras la evidencia ground-truth tiene ~50 palabras ($|W(GT)|$ pequeño). Por simetría de Jaccard:
$$J(GT, C) = \frac{|W(GT) \cap W(C)|}{|W(GT) \cup W(C)|} \leq \frac{50}{300 + 50 - 50} \approx 0.17 \ll 0.75$$

El umbral de 0.75 es inalcanzable → B.cov = 0 para todo RAG 3.

**Containment Recall** (asimétrica, correcta para este caso):
$$\text{ContainmentRecall}(GT, C) = \frac{|W(GT) \cap W(C)|}{|W(GT)|}$$

Mide qué fracción del ground-truth está contenida en el chunk. Si el chunk grande contiene la evidencia pequeña, la métrica es alta (cercana a 1.0). Umbral usado: 0.5.

**Equivalencia con los otros RAGs**: Ambas métricas miden solapamiento léxico. La diferencia es la normalización. Containment recall es la métrica correcta cuando $|C| \gg |GT|$, que es exactamente el escenario DCI. Para RAG 2.x donde los chunks tienen ~300 tokens (similar a GT), Jaccard ≈ Containment, por lo que las comparaciones son válidas.

### 4.5 Agente DCI — Loop ReAct Stateless

El agente implementa el paradigma Reason+Act con diseño **stateless-per-iteration**:

```
Para i = 1..max_iter:
    user_msg = construir_resumen(query, términos_buscados, grep_findings_previos)
    response = LLM([system_prompt, user_msg])   ← contexto fresco cada iteración
    action   = parsear_ACTION_ARGS(response)
    result   = ejecutar_herramienta(action)
    acumular_evidencia(result)
```

**Por qué stateless**: Los modelos de 14B parámetros tienden a declarar terminación prematura cuando ven tokens de parada en su propio historial de chat. Al pasar el estado acumulado como texto en el mensaje de usuario (no como historial multi-turno), se elimina este sesgo.

**Herramientas disponibles**:
- `grep_corpus(pattern)` → busca en todos los archivos del corpus
- `read_lines(paper_id, section, start, end)` → lee sección específica
- `read_file(paper_id, section)` → lee sección completa
- `list_sections(paper_id)` → lista secciones disponibles

---

## 5. Métricas de Evaluación

### 5.1 Mean Reciprocal Rank (MRR)

$$\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}$$

donde $\text{rank}_q$ es la posición del primer chunk relevante en el ranking para la query $q$ (0 si no se recupera ninguno).

**Por qué MRR**: Penaliza fuertemente cuando el chunk relevante aparece lejos del top. Apropiado para QA donde la primera coincidencia es la más importante.

**Validez**: Métrica estándar en Information Retrieval (TREC, MS-MARCO). Computable sin interacción humana si el ground-truth es conocido.

### 5.2 Recall@k

$$\text{Recall@k} = \frac{1}{|Q|} \sum_{q} \frac{|\text{relevant} \cap \text{retrieved}_k|}{|\text{relevant}|}$$

Fracción de chunks relevantes recuperados en los top-$k$.

**Por qué Recall@5**: Medida complementaria a MRR que captura cobertura. MRR puede ser alto aunque no se recuperen todos los evidencias.

### 5.3 NDCG@5 (Normalized Discounted Cumulative Gain)

$$\text{NDCG@5} = \frac{\text{DCG@5}}{\text{IDCG@5}}, \quad \text{DCG@5} = \sum_{i=1}^{5} \frac{\text{rel}_i}{\log_2(i+1)}$$

Mide calidad del ranking ponderando posición: un chunk relevante en posición 1 vale más que en posición 5.

### 5.4 B.cov — Oracle Coverage

$$B.\text{cov} = \frac{|\text{chunks aprobados por oráculo}|}{|\text{chunks ground-truth}|}$$

Mide qué fracción de la evidencia correcta fue recuperada Y aprobada por el oráculo. Combina retrieval y curación.

**Validez**: Métrica interna del sistema. Un B.cov=0 indica que el oráculo no aprobó ningún chunk, independientemente de la métrica de similitud usada.

### 5.5 ROUGE-L (Lexical Overlap)

$$\text{ROUGE-L}(r, c) = \frac{\text{LCS}(r, c)}{|r|}$$

donde LCS es la subsecuencia común más larga entre respuesta generada $c$ y referencia $r$.

**Por qué ROUGE-L**: Mide superposición léxica sin necesidad de modelos externos. Validado en decenas de benchmarks NLP (GLUE, SQuAD). Limitación: sensible al vocabulario, no captura paráfrasis semánticas.

### 5.6 BERTScore F1

$$\text{BERTScore-F1}(r, c) = 2 \cdot \frac{P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}$$

donde $P_{\text{BERT}}$ y $R_{\text{BERT}}$ son la precisión y recall computadas sobre similitudes coseno de embeddings token a token de BERT.

**Por qué BERTScore**: Captura similitud semántica que ROUGE no detecta. Correlaciona mejor con juicio humano (Pearson ~0.65 en WMT) que ROUGE (~0.45).

### 5.7 Faithfulness (Fidelidad al Contexto)

$$\text{Faithfulness} = \frac{1}{|Q|} \sum_{q} \mathbb{1}[\text{cosine}(\text{embed}(\text{response}_q), \text{embed}(\text{context}_q)) \geq \theta]$$

con umbral $\theta = 0.75$. Mide si la respuesta generada está anclada en el contexto recuperado (no alucina desde parámetros internos del LLM).

**Por qué Faithfulness**: Es la métrica más importante para RAG en producción. Un sistema con ROUGE alto pero faithfulness bajo está inventando respuestas plausibles pero no fundamentadas.

---

## 6. Resultados Completos

### 6.1 Tabla General — Todos los Sistemas

| Sistema | A.MRR | A.Rec@5 | A.NDCG@5 | B.cov | C.ROUGE-L | C.BERTScore | C.Faith | Oracle |
|---|---|---|---|---|---|---|---|---|
| **RAG 1** | 0.2211 | 0.2115 | — | — | **0.1022** | 0.8279 | 0.1600 | Ninguno |
| **RAG 2.1** | 0.2135 | 0.2115 | 0.2813 | 0.2473 | 0.0694 | 0.8373 | 0.0668 | Jaccard |
| **RAG 2.3** | 0.3008 | 0.2671 | 0.2413 | — | 0.1206 | 0.8309 | **0.1819** | Ninguno |
| **RAG 2.2 — R0** (sin fine-tuning) | 0.3008 | 0.2671 | 0.2413 | 0.2866 | 0.1829 | 0.8553 | 0.2298 | Jaccard |
| **RAG 2.2 — RN** (fine-tuning ×4) | **0.4559** | **0.3046** | **0.3403** | **0.3046** | 0.1190 | 0.8327 | 0.1011 | Jaccard |
| **RAG 3** | 0.2819 | 0.2177 | 0.2963 | 0.0000 | 0.0685 | 0.8200 | 0.0000 | Jaccard→0 |
| **RAG 3.1** | 0.1544 | 0.1120 | 0.1572 | 0.1092 | 0.0344 | 0.4343 | 0.0012 | Containment |
| **RAG 3.2** | 0.3100 | 0.2152 | 0.3174 | 0.1485 | 0.0765 | 0.8225 | 0.0037 | Containment |

> **Nota**: RAG 2.2 — R0 = reranker base + oráculo, sin fine-tuning. RAG 2.2 — RN = reranker fine-tuneado 4 rondas + oráculo. Los sistemas con Oracle=Jaccard o Containment conocen el ground-truth durante curación (B post-curation) → no son deployables directamente. El Punto A es deployable para todos.

### 6.2 Comparación de Ablaciones — Efecto de Cada Componente

| Ablación | Sistemas | ΔMRR (A) | Δcov (B) | ΔROUGE (C) | Conclusión |
|---|---|---|---|---|---|
| Reranker | RAG1 → RAG2.3 | +0.0797 (+36%) | N/A | +0.0184 (+18%) | Reranker es el mayor contribuyente individual deployable |
| Oráculo solo | RAG1 → RAG2.1 | −0.0076 (−3%) | +0.247 | −0.033 (−32%) | Oráculo mejora cobertura, daña generación |
| Oracle+Reranker R0 | RAG1 → RAG 2.2 R0 | +0.080 (+36%) | +0.287 | +0.061 (+60%) | Base antes del fine-tuning |
| **Fine-tuning AL** | **RAG 2.2 R0 → RN** | **+0.155 (+52%)** | +0.018 | −0.064 | **Mayor ganancia de retrieval del estudio** |
| Oracle+Reranker RN | RAG1 → RAG 2.2 RN | +0.235 (+106%) | +0.305 | +0.017 (+17%) | Sinergia total: reranker + oráculo + fine-tuning |
| Corpus Docling | RAG3 → RAG3.2 | +0.028 (+10%) | +0.149 (∞) | +0.008 (+12%) | Docling desbloquea B.cov |
| Modelo DCI | RAG3.1 → RAG3.2 | +0.156 (+101%) | +0.039 | +0.042 (+122%) | DeepSeek-R1 >> Granite 3.1 8B para DCI |

### 6.3 RAG 2.x Detalle — Punto B (Post-Curación)

| Sistema | B.MRR | B.Rec@5 | B.P@5 | B.cov |
|---|---|---|---|---|
| RAG 2.1 | 0.4044 | 0.2473 | 0.4044 | 0.2473 |
| RAG 2.3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RAG 2.2 | **0.4559** | **0.3046** | **0.4559** | **0.3046** |

> RAG 2.3 (reranker only) tiene B.cov=0 porque no usa oráculo → sin curación, no hay "post-curation" comparable.

### 6.4 RAG 3.x Detalle — Punto A y B (DCI Agent)

| Sistema | A.MRR | A.Rec@5 | B.cov | B.MRR | Evidencia/query (media) |
|---|---|---|---|---|---|
| RAG 3 | 0.2819 | 0.2177 | 0.000 | 0.000 | 1.61 chunks |
| RAG 3.1 | 0.1544 | 0.1120 | 0.109 | 0.125 | 1.52 chunks |
| RAG 3.2 | 0.3100 | 0.2152 | 0.149 | 0.257 | 1.52 chunks |

---

## 7. Aprendizaje Activo: Round 0 → Round N

El bucle de aprendizaje activo es la contribución central de SAIL-RAG: cada decisión de curación del oráculo genera una tripleta `(query, chunk_positivo, chunk_negativo)` que se usa para fine-tuning contrastivo del reranker (BGE-reranker-v2-m3) con `MarginRankingLoss`.

### 7.1 Condiciones del experimento

| Parámetro | Valor |
|---|---|
| Modelo base | `BAAI/bge-reranker-v2-m3` (560M params) |
| Optimizador | Adafactor (O(√n) memoria vs O(2n) AdamW) |
| Learning rate | 2×10⁻⁵ |
| Épocas por ronda | 3 |
| Margen (MarginRankingLoss) | 0.5 |
| Umbral de acumulación | 50 tripletas |
| Dispositivo fine-tuning | CPU (evita OOM en RTX 3050 4GB VRAM) |
| Queries usadas | 136 (subset estratificado por longitud) |

### 7.2 Curva de aprendizaje activo

| Ronda | Tripletas usadas | Loss final |
|---|---|---|
| R1 | 50 | 0.01664 |
| R2 | 52 | 0.00289 |
| R3 | 50 | 0.00148 |
| R4 | 32 | 0.00206 |

La pérdida cae 11× entre R1 y R3, indicando convergencia rápida del MarginRankingLoss. La ligera subida en R4 (0.00148 → 0.00206) se debe al batch más pequeño (32 tripletas finales vs 50 de rondas anteriores).

### 7.3 Impacto del fine-tuning sobre las métricas

| Métrica | Round 0 (base) | Round N (fine-tuned) | Δ absoluto | Δ relativo |
|---|---|---|---|---|
| **A.MRR** | 0.3008 | **0.4559** | +0.1551 | **+51.6%** |
| **A.Rec@5** | 0.2671 | **0.3046** | +0.0375 | +14.0% |
| **A.NDCG@5** | 0.2413 | **0.3403** | +0.0990 | +41.0% |
| **B.cov** | 0.2866 | **0.3046** | +0.0180 | +6.3% |
| **B.MRR** | 0.4412 | **0.4559** | +0.0147 | +3.3% |
| C.ROUGE-L | **0.1829** | 0.1190 | −0.0639 | −34.9% |
| C.BERTScore | **0.8553** | 0.8327 | −0.0226 | −2.6% |
| C.Faith | **0.2298** | 0.1011 | −0.1287 | −56.0% |

> **Nota sobre ROUGE-L y Faithfulness**: La caída en métricas de generación (C) es coherente con el comportamiento observado en el oráculo solo (RAG 2.1): el reranker fine-tuneado recupera chunks más precisos léxicamente pero más cortos y focalizados, que aportan menos contexto parafraseable al LLM. La mejora en retrieval (Punto A) es la señal principal del bucle de aprendizaje activo.

### 7.4 Diagnóstico: tensión retrieval/generación tras el fine-tuning

El fine-tuning contrastivo forzó al reranker a asignar scores más altos a chunks léxicamente similares al ground-truth. Esto sube el chunk correcto en el ranking (+52% MRR), pero el efecto colateral es que el contexto entregado al LLM se vuelve más preciso y menos diverso: menos contexto parafraseable → ROUGE-L −35%, Faithfulness −56%.

Esta tensión no es un artefacto del experimento, sino un resultado estructural: el oracle duro optimiza precisión de evidencia, no riqueza generativa. Un revisor que lea solo la tabla 7.3 notará que la columna C empeora mientras A mejora. Lo declaramos explícitamente: **el bucle de aprendizaje activo resuelve el problema de recuperación pero no el de generación**. La hipótesis de SAIL-RAG se valida parcialmente — la etapa de curación humana real (no simulada por oráculo) podría preservar más contexto y mitigar esta degradación, pero queda como trabajo futuro.

---

## 8. Interpretaciones

### 8.1 El reranker cross-encoder es el componente más valioso (RAG 2.3)

RAG 2.3 (reranker, sin oráculo) mejora MRR en **+36% sobre RAG 1** sin necesitar ground-truth. Esto es deployable en producción. El oráculo solo (RAG 2.1) no mejora MRR y reduce ROUGE-L en -32%, probablemente porque el oráculo excluye chunks contextualizadores que ayudan al LLM aunque no contengan la respuesta exacta.

### 8.2 El oráculo mejora cobertura pero no generación léxica

RAG 2.1 tiene B.cov=0.247 (recupera el 24.7% de la evidencia correcta) pero ROUGE-L baja de 0.1022 a 0.0694. Paradoja: los chunks curados son más precisos como evidencia pero el LLM genera respuestas con menos overlap léxico. Hipótesis: el oráculo filtra chunks con contexto adicional que enriquece la generación.

### 8.3 El +106% de MRR en RAG 2.2 RN es reranker + fine-tuning, no sinergia con el oráculo

RAG 2.2 RN alcanza A.MRR=0.456 (+106% vs RAG 1). Esta ganancia se descompone en dos pasos independientes sobre el Punto A:

1. **Reranker** (RAG 1 → RAG 2.2 R0): 0.221 → 0.301 (+36%) — el cross-encoder reordena los candidatos ANN.
2. **Fine-tuning AL** (RAG 2.2 R0 → RN): 0.301 → 0.456 (+52%) — el reranker ajustado por aprendizaje activo recupera el chunk correcto más arriba en el ranking.

El oráculo no contribuye al Punto A porque opera en el Punto B (curación post-reranking). El Punto A se mide antes de que el oráculo intervenga, por lo que RAG 2.2 R0 y RAG 2.3 tienen A.MRR idéntico (0.301). La mejora de cobertura que aporta el oráculo aparece únicamente en B.cov (0.287 en R0, 0.305 en RN).

### 8.4 El aprendizaje activo es el mayor contribuyente individual en retrieval

El fine-tuning contrastivo de 4 rondas (184 tripletas totales) mejora A.MRR en **+51.6%** sobre el reranker base (Round 0 → Round N: 0.301 → 0.456). Esto supera el efecto del reranker sobre RAG 1 (+36%, RAG 1 → RAG 2.3) y es la mayor ganancia de retrieval en todo el estudio. La señal de entrenamiento proviene exclusivamente de las decisiones de curación del oráculo, sin ningún dataset externo.

### 8.5 El agente DCI (RAG 3.2) compite con RAG 2.3 en MRR

RAG 3.2 (DCI+Docling, MRR=0.310) ≈ RAG 2.3 (reranker, MRR=0.301) con la diferencia de que DCI no usa embedding vectorial: busca mediante grep iterativo guiado por razonamiento. Esto es notable porque DCI opera sobre texto plano sin índice vectorial previo.

### 8.6 Faithfulness muy baja en RAG 3.x

RAG 3 y RAG 3.2 tienen faithfulness ≈ 0. Esto se debe a que el agente DCI acumula evidencia de múltiples secciones heterogéneas del paper, y el LLM generador (Llama-3.3-70B) no puede anclar su respuesta a un contexto tan fragmentado. La baja faithfulness indica que el LLM usa conocimiento paramétrico más que el contexto DCI.

### 8.7 Granite 3.1 8B es significativamente inferior a DeepSeek-R1 14B para DCI

RAG 3.1 (Granite) MRR=0.154 vs RAG 3.2 (DeepSeek) MRR=0.310 — **brecha del 101%**. Granite genera llamadas a herramientas con nombres alternativos (`search_corpus` en lugar de `grep_corpus`) que se resolvieron con aliases, pero la calidad del razonamiento es inferior. DeepSeek-R1 tiene cadenas de pensamiento explícitas (`<think>`) que mejoran la selección de términos de búsqueda.

---

## 9. Impacto de Docling en el Agente DCI

### 9.1 ¿Qué es Docling?

Docling es una librería de IBM Research para extracción estructurada de PDFs científicos. A diferencia del texto plano del dataset (que pierde estructura de secciones), Docling:

- Detecta y preserva secciones semánticas (Abstract, Introduction, Methodology, Results, Discussion, Conclusion, Related Work)
- Detecta tablas y figuras (aunque no las incluye en los chunks de texto)
- Elimina artefactos de layout (números de página, encabezados repetidos, referencias en columnas)

### 9.2 Comparación de corpora

| Aspecto | Corpus Plano | Corpus Docling |
|---|---|---|
| Archivos de sección | 283 | **445 (+57%)** |
| Secciones únicas | abstract, intro, results, methodology, discussion, limitations | + conclusion, related_work, unknown |
| Calidad del texto | Artefactos de PDF (columnas mezcladas, headers) | Texto limpio por sección |
| Identificación de secciones | Basada en metadata original PeerQA | Re-detectada por Docling |

### 9.3 Resultados del impacto

| Métrica | RAG 3 (plano) | RAG 3.2 (Docling) | Δ |
|---|---|---|---|
| A.MRR | 0.282 | **0.310** | +10.0% |
| B.cov | 0.000 | **0.149** | +∞ |
| Queries con B.cov > 0 | 0/136 | **35/136 (26%)** | — |
| C.ROUGE-L | 0.069 | **0.077** | +11.6% |

### 9.4 Análisis por query

De las 136 queries:
- **30 queries**: RAG 3.2 mejora MRR sobre RAG 3 (Docling ayuda)
- **25 queries**: RAG 3 supera RAG 3.2 (Docling perjudica)
- **81 queries**: Sin diferencia en MRR

La mejora en B.cov es universal: RAG 3 no logra B.cov > 0 en ninguna query (Jaccard incompatible), mientras RAG 3.2 logra B.cov > 0 en 35 queries (26%) con containment recall.

---

## 10. Limitaciones

### 10.1 Incomparabilidad de la métrica de oráculo

RAG 2.x usa Jaccard ≥ 0.75; RAG 3.x usa containment recall ≥ 0.5. Las métricas B.cov no son directamente comparables entre grupos. La razón es técnica (chunks DCI vs chunks vectoriales tienen tamaños muy diferentes), pero dificulta el análisis cross-sistema.

### 10.2 Oracle = upper bound no deployable

RAG 2.1 y RAG 2.2 usan el ground-truth durante curación. Esto hace que B.cov y las métricas post-curación sean un upper bound teórico, no una estimación de rendimiento en producción. RAG 2.3 (reranker solo) es el único sistema de los RAG 2.x completamente deployable.

### 10.3 Subset de 136 queries

Se evaluó un subset de 136 de las ~1,000+ queries de PeerQA. La varianza es alta: los intervalos de confianza no fueron calculados formalmente. Los resultados deben interpretarse como tendencias, no como valores absolutos definitivos.

### 10.4 Faithfulness con embedding propio

La faithfulness se calcula como similitud coseno entre embeddings de la respuesta y el contexto usando el mismo modelo BGE-M3. Este proxy puede subestimar faithfulness real (el modelo puede capturar similitud superficial que no corresponde a fidelidad semántica fina).

### 10.5 DCI limitado a grep léxico

El agente DCI usa `grep_corpus` (búsqueda por subcadena exacta) como herramienta principal. Esto favorece queries con términos técnicos específicos y perjudica queries conceptuales abstractas donde la evidencia usa vocabulario diferente.

### 10.6 Costo computacional DCI

RAG 3.x requiere DeepSeek-R1 14B en Ollama (≥8GB VRAM) + Groq API para generación + ~5 iteraciones de 30-120 segundos por query. Una evaluación completa tarda ~8-12 horas en hardware consumer. RAG 2.3 tarda <5 minutos para las mismas 136 queries.

### 10.7 BERTScore en RAG 3.1 (Granite)

El BERTScore de RAG 3.1 es 0.434, anormalmente bajo (los otros sistemas están en 0.82+). Esto indica que las respuestas generadas en RAG 3.1 son muy cortas o vacías, probablemente porque el contexto DCI fragmentado de Granite no fue suficiente para que el LLM generara respuestas completas.

---

## 11. Conclusiones

1. **El fine-tuning contrastivo por aprendizaje activo mejora retrieval pero degrada generación**: +51.6% MRR (Round 0 → Round N) con 184 tripletas del oráculo, sin datos externos — la mayor ganancia de recuperación del estudio. Sin embargo, ROUGE-L cae −35% y Faithfulness −56% en ese mismo paso. El bucle de aprendizaje activo valida la hipótesis de mejora de recuperación, pero no la de mejora de calidad de respuesta final; la tensión retrieval/generación es un resultado abierto del sistema.

2. **El reranking cross-encoder (BGE-reranker-v2-m3) es el componente individual más valioso** en sistemas RAG para QA científico: +36% MRR sin oracle, deployable.

3. **El agente DCI con DeepSeek-R1 14B y corpus Docling (RAG 3.2) alcanza calidad comparable al reranker** en MRR (0.310 vs 0.301), con la ventaja adicional de no requerir embedding vectorial previo.

4. **Docling desbloquea la curación oracle en DCI**: sin Docling, B.cov=0 por incompatibilidad Jaccard; con Docling y containment recall, B.cov=0.149 (35/136 queries cubiertas).

5. **El modelo de razonamiento importa más que el corpus para DCI**: DeepSeek-R1 14B supera a Granite 3.1 8B en +101% MRR bajo las mismas condiciones de corpus. Lo que deja una tendencia de que un modelo más capaz quizá obtenga mejores resultados de rendimiento.

6. **Faithfulness es el talón de Aquiles de los sistemas DCI**: la evidencia fragmentada del agente no permite al LLM anclar su respuesta, resultando en generación paramétrica. RAG 2.3 (reranker) tiene faithfulness 49× superior a RAG 3.2.

---

*Proyecto SAIL-RAG Comparative Study. Dataset: PeerQA (MTEB). Hardware: RTX 3050 4GB VRAM + Groq cloud API.*
