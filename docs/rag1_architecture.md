# RAG 1 Baseline — Architecture

## Pipeline Diagram

```mermaid
flowchart TD
    subgraph INGEST["🔄 Mode: ingest"]
        A[PeerQALoader\npeerqa_loader.py] -->|papers + qa_pairs| B[Chunker\nchunker.py]
        B -->|chunks list[dict]| C[BGEEmbedder\nembedder.py\nload → embed → unload]
        C -->|embeddings float32| D[(QdrantStore\nin-memory)]
        C -->|rag1_embeddings.npy| E[(checkpoints/)]
        B -->|rag1_chunks.json| E
        A -->|rag1_qa_pairs.json| E
    end

    subgraph EVALUATE["📊 Mode: evaluate"]
        F[(checkpoints/)] -->|load npy + json| G[QdrantStore\nreconstruida]
        F --> H[BGEEmbedder\nload]
        H -->|query_vec| G
        G -->|top-20 chunks| PA["⚑ PUNTO A\npre-LLM\nMRR · Recall@k · P@5"]
        G -->|top-5 chunks| I[GroqLLM\nllama-3.3-70b]
        I -->|generated_response| PB["⚑ PUNTO B\npost-LLM\nROUGE-L · BERTScore"]
        PA --> J[(results/rag1/\nmetrics_post_llm.json)]
        PB --> J
    end

    subgraph VISUALIZE["🎨 Mode: visualize"]
        K[(results/rag1/\nmetrics_post_llm.json)] --> L[RAG1Visualizer\nvisualizer.py]
        L --> M[5 PNG\nresults/rag1/figures/]
    end
```

**Punto A** — evaluación de recuperación: compara chunks recuperados vs `answer_evidence_mapped` de PeerQA usando match de substring y Jaccard sobre palabras.

**Punto B** — evaluación de generación: compara respuesta del LLM vs `answer` de referencia.

---

## Design Decisions

| Componente | Alternativas consideradas | Por qué esta elección |
|---|---|---|
| **Embedding: sentence-transformers** | FlagEmbedding, transformers directo | Python 3.13 compat garantizada; `model.encode()` abstrae batching y normalización; FlagEmbedding requiere Python ≤3.12 en algunas builds |
| **Vector DB: Qdrant in-memory** | ChromaDB, FAISS, Milvus | Qdrant tiene API limpia para RAG; in-memory elimina dependencias de servidor; checkpoint numpy resuelve la no-persistencia |
| **LLM: Groq llama-3.3-70b** | OpenAI GPT-4o, Ollama local, Anthropic | Groq es gratuito para experimentación; llama-3.3-70b es competitivo; corre en la nube sin consumir VRAM local |
| **Dataset: PeerQA** | SQuAD, NQ, BEIR | PeerQA tiene evidencia explícita mapeada al texto → evalúa retrieval con ground truth real; dominio académico relevante para SAIL |
| **BERTScore en CPU** | GPU, skip si OOM | BERTScore tarda ~2s/query en CPU y Groq es el cuello de botella real (red); forzar CPU evita competencia con embedder y simplifica VRAM management |
| **Checkpoint numpy + JSON** | Qdrant snapshot, pickle, HDF5 | Qdrant in-memory no serializa; numpy es portable y eficiente; JSON para metadata es legible y debuggeable |
| **load/unload explícitos en BGEEmbedder** | Context manager, lazy loading | RAG 2/3 necesitarán embedder + reranker; tener load/unload como métodos permite al pipeline decidir el orden explícitamente sin acoplamiento en constructores |
| **Chunking con tokenizer BGE-M3** | spaCy por oraciones, char-split | Garantiza que ningún chunk exceda el límite real del modelo; el overlap de 64 tokens preserva contexto en bordes |

---

## GPU Optimization Notes

### Hardware objetivo: RTX 3050 Laptop — 4 GB VRAM

| Componente | VRAM | Estrategia |
|---|---|---|
| BGE-M3 fp16 | ~0.9 GB | Cargado solo durante embedding, liberado antes de cualquier otro modelo |
| Qdrant in-memory | 0 (RAM) | Reconstruido desde checkpoint numpy en cada evaluate |
| Groq LLM | 0 (cloud) | No consume VRAM local |
| BERTScore | 0 (CPU forzado) | `device="cpu"` hardcodeado en compute_point_b |

### Patrón "un modelo a la vez en GPU"

```python
# ingest:
embedder.load()        # VRAM: ~0.9 GB (solo BGE-M3)
embedder.embed_batch() # procesa todos los chunks
embedder.unload()      # VRAM: 0 GB → libre para RAG 2/3

# evaluate (embedder para queries):
embedder.load()
for qa in qa_pairs:
    embedder.embed_query(qa["query"])
    # Groq corre en la nube — no hay conflicto de VRAM
embedder.unload()
```

### OOM Retry

Si `torch.cuda.OutOfMemoryError` al embeder:
1. Batch se reduce a la mitad
2. `torch.cuda.empty_cache()` + reintento
3. Si batch=1 sigue fallando → error propagado (modelo no cabe en GPU)

El batch mínimo efectivo para BGE-M3 fp16 en 4 GB es ~4 chunks de 384 tokens.
