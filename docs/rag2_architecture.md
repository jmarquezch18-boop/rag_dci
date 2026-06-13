# RAG 2 Architecture — SAIL-RAG con Curación y Aprendizaje Activo

## Pipeline completo con 3 puntos de evaluación

```mermaid
flowchart TD
    subgraph INGEST["INGEST (rag2 --mode ingest)"]
        A[PeerQALoader\nQA pairs + corpus IDs] --> B[DoclingParser\nDescarga PDFs ArXiv]
        B --> C[SectionChunker\nChunks con sección]
        C --> D[BGEEmbedder\nEmbeddings BGE-M3]
        D --> E[(QdrantStoreRAG2\npeerqa_chunks_rag2)]
        A --> F[rag2_qa_pairs.json]
        D --> G[rag2_embeddings.npy\nrag2_chunks.json]
    end

    subgraph EVALUATE["EVALUATE (evaluate_r0 / evaluate_rn)"]
        H[Query] --> I[BGEEmbedder\nembed_query]
        I --> J["ANN Search\n(k=20, filtro sección opcional)"]
        J --> K["BGEReranker\nBGE-reranker-v2-m3\n(top 7)"]
        K -->|"PUNTO A\n📊 MRR, Recall@k\nNDCG@5, P@5"| L{{"CurationOracle\nanswer_evidence_mapped"}}
        L -->|"PUNTO B\n📊 MRR, Recall@5\nCoverage, NDCG@5"| M["GroqLLM\nllama-3.3-70b"]
        M -->|"PUNTO C\n📊 ROUGE-L, BERTScore\nFaithfulness coseno"| N[Respuesta + traza]
        L --> O["FeedbackStore\ntripletas pos/neg"]
    end

    subgraph ACTIVE_LEARNING["TRAIN (--mode train)"]
        O --> P{{"len ≥ threshold?"}}
        P -->|sí| Q["BGEReranker\nfine_tune_contrastive\nMarginRankingLoss"]
        Q --> R[reranker_round_N.pt]
        Q --> S[active_learning_curve.json]
        P -->|no| T[acumular...]
    end

    E --> J
    G --> J
    R -.->|load_checkpoint| K
```

## Tabla comparativa RAG 1 vs RAG 2

| Etapa | RAG 1 (Baseline) | RAG 2 (SAIL) | Por qué cambia |
|---|---|---|---|
| **Ingesta de corpus** | Corpus pre-chunkeado de MTEB | Docling descarga PDFs, preserva secciones | RAG 2 necesita metadato de sección para filtrar y para el oráculo |
| **Chunking** | Ventana deslizante plana (Chunker) | Ventana deslizante con etiqueta de sección (SectionChunker) | La sección es la unidad semántica natural en papers científicos |
| **Vector store** | `peerqa_chunks` (QdrantStore) | `peerqa_chunks_rag2` (QdrantStoreRAG2) | Colección separada con payload de sección; filtrado ANN por sección |
| **Retrieval** | BGE-M3 ANN k=20 → top-5 directo | BGE-M3 ANN k=20 → BGEReranker top-7 | El cross-encoder reordena por relevancia conjunta (query+chunk) |
| **Curación** | No existe | CurationOracle simula decisión humana | SAIL estudia el impacto de la intervención humana en la cadena |
| **Contexto al LLM** | Top-5 chunks del ANN | Solo chunks aprobados por el oráculo | La curación reduce ruido y tokens; fallback si 0 aprobados |
| **Puntos de evaluación** | 2 puntos (A=retrieval, B=post-LLM) | 3 puntos (A=pre-curación, B=post-curación, C=post-LLM) | El Punto B es el corazón del estudio: mide el valor de la curación |
| **Faithfulness** | RAGAS judge (LLM-as-judge, costoso) | Coseno BGE-M3 en CPU (local, determinístico) | RAGAS consume tokens API; coseno es reproducible y gratuito |
| **NDCG** | No disponible | NDCG@5 en Puntos A y B | NDCG penaliza ranking tardío de relevantes; más informativo que Recall solo |
| **Coverage** | No disponible | Fracción ground truth en chunks aprobados | Métrica clave del Punto B: ¿cuánta evidencia sobrevivió la curación? |
| **Aprendizaje activo** | No existe | Fine-tuning contrastivo del reranker | Las decisiones del oráculo mejoran el reranker ronda a ronda |
| **Gestión de VRAM** | BGE-M3 load/unload explícito | BGE-M3 → unload → Reranker → unload → CPU | RTX 3050 4GB no soporta ambos modelos simultáneamente |
| **Token limits** | Stratified sample + judge subset | Misma estrategia + contexto reducido por curación | La curación natural reduce tokens por llamada al LLM |

## Bucle de aprendizaje activo

```mermaid
flowchart LR
    A[evaluate_r0\nRonda 0] -->|"oracle.curate()\ntripletas pos/neg"| B[FeedbackStore\nfeedback_store.json]
    B --> C{{"len ≥ accumulation_threshold\n(default: 50)"}}
    C -->|sí| D["BGEReranker\nfine_tune_contrastive\nMarginRankingLoss(margin=0.5)"]
    D --> E[reranker_round_N.pt]
    D --> F[active_learning_curve.json\nMRR + loss por ronda]
    E -->|"load_checkpoint"| G[evaluate_rn\nRonda N]
    G -->|nuevas tripletas| B
    C -->|no| H[acumular hasta\npróxima query]
```

### Detalle del fine-tuning contrastivo

**Tripleta:** `(query, chunk_aprobado, chunk_rechazado)`

El oráculo genera tripletas donde:
- **Positivo:** chunk que contiene evidencia del `answer_evidence_mapped` (aprobado)
- **Negativo duro:** el mejor chunk rechazado (mayor `rerank_score` entre los no aprobados)

**Pérdida:** `MarginRankingLoss(pos_score, neg_score, margin=0.5)`

El reranker aprende que `score(query, chunk_relevante) > score(query, chunk_irrelevante) + margin`.

**Acumulación:** las tripletas se persisten en `feedback_store.json` para sobrevivir reinicios.

**Umbral:** configurable en `config.yaml` → `rag2.active_learning.accumulation_threshold` (default: 50 tripletas).

## Estructura de archivos generados

```
results/
├── rag1/
│   ├── metrics_post_llm.json
│   └── figures/                    ← 5 figuras RAG 1
├── rag2/
│   ├── round_0/
│   │   ├── metrics_pre_curation.json    ← Punto A
│   │   ├── metrics_post_curation.json   ← Punto B
│   │   ├── metrics_post_llm.json        ← Punto C + todo
│   │   └── figures/                     ← 6 figuras RAG 2 R0
│   └── round_n/
│       ├── metrics_pre_curation.json
│       ├── metrics_post_curation.json
│       ├── metrics_post_llm.json
│       ├── active_learning_curve.json   ← copiado desde checkpoints/
│       └── figures/                     ← 7 figuras RAG 2 RN
└── comparison_rag1_vs_rag2.png          ← generado si existen ambos

checkpoints/
├── rag1_embeddings.npy
├── rag1_chunks.json
├── rag1_qa_pairs.json
├── rag2_embeddings.npy
├── rag2_chunks.json
├── rag2_qa_pairs.json
└── reranker/
    ├── reranker_round_1.pt
    ├── reranker_round_2.pt
    ├── feedback_store.json
    └── active_learning_curve.json
```
