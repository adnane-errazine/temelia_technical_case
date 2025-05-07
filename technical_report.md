## Technical Report

TemelIA is a specialized Retrieval‑Augmented Generation (RAG) application tailored for construction and building engineering documents. Leveraging hybrid search (dense + sparse embeddings), cross‑encoder reranking, and SOTA LLMs, it enables precise, reference‑anchored answers over complex PDFs, with a focus on French documentation. A Gradio frontend offers an intuitive user experience. Its modular architecture allows each component to be adapted to specific use cases and client requirements, and it provides a solid foundation for GDPR‑friendly local processing.

---

## System Architecture

### Ingestion & Parsing  
1. **Document Upload**: Users upload PDFs via Gradio.  
2. **Parser Layer**:  
   - **pdfplumber** for baseline text extraction (insufficient for some French‑specific formats)  
   - **Markitdown** for structured Markdown conversion (similarly limited on French technical layouts)  
   - **Mistral** (cloud‑based LLM parsing) good price‑performance ratio (≈ $1 per 1 000 pages)  
   - **Local VLM** (“OpenGVLab/InternVL3-1B”) for OCR and layout analysis on sensitive documents

### Chunking & Embedding  
- **Chunker**: `RecursiveCharacterTextSplitter` (256 token chunks, 60 token overlap, using the GPT‑4o tokenizer).  
  Alternative strategies, e.g., [Anthropic’s Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) can be explored depending on budget and requirements.  
  Additional context (section summaries, metadata) may be prepended to each chunk.  
- **Dense Embeddings**: [`intfloat/multilingual-e5-large-instruct`](https://huggingface.co/intfloat/multilingual-e5-large-instruct) (1 024 dimensions),  
  chosen for its multilingual support and performance/size trade‑off (see the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)).  
- **Sparse Embeddings**: BM25 via Qdrant’s `SparseTextEmbedding`. BM25’s hyperparameters can be tuned, though gains are often marginal;  
  emerging variants (e.g., BM42) could be evaluated in the future.

### Storage & Retrieval  
- **Vector Store**: Qdrant (self‑hosted or cloud; GDPR‑friendly for French data).  
- **Hybrid Search**: Reciprocal Rank Fusion (RRF) merges the top 40 dense and top 40 sparse matches.  
- **Reranking**: Cohere rerank‑multilingual‑v3.0, narrowing results to the top 20.

### Answer Generation  
- **Query Enhancer**: `gpt-4.1-nano` for optional question reformulation.  
- **LLM**: OpenAI GPT‑4.1 for final answer generation.  
- **Transparency**: Each response includes citations of the exact source passages.

---

## Frameworks & Workflow

- **LangChain**: Orchestrates the RAG pipeline.

---

## Frontend & User Experience

- **Gradio UI**: Two‑pane layout  
  - **Left pane**: document upload and question input  
  - **Right pane**: generated answer with highlighted source chunks  
- **Controls**:  
  - “Ingestion du document” initiates parsing  
  - “Envoyer” submits the query  
  - “Réinitialiser” clears the session

---

## Implementation Highlights

- **Environment**: Python 3.12+, managed via UV (`uv sync` from `pyproject.toml`).  
- **Dependencies** (for local VLM/OCR use):  
  - GPU‑accelerated libraries (flash‑attn, triton) often require manual build steps on Windows.  
  - Visual C++ Build Tools for certain Python wheels.  
- **Configuration**: `.env` file for OpenAI, Cohere, and Qdrant credentials, plus collection naming.

---

## Evaluation & Lessons Learned

- **What Worked**:  
  - Hybrid retrieval delivered robust relevance on French technical queries.  
  - The Cohere reranker significantly improved top‑k precision.  
  - Local VLM processing enabled GDPR‑compliant handling of sensitive documents.  
- **Challenges**:  
  - Installing GPU‑optimized libraries (flash‑attn/triton) on Windows was complex.  ( Prepared 1 package in 392m 15s Installed 1 package in 122ms + flash-attn==2.7.3)
  - Limited memory and compute resources constrained batch sizes and throughput.  
- **Future Directions**:  
  - Advanced query enrichment using LangGraph or similar agentic workflows.  
  - Enhanced parsing pipeline for images: detection → classification → information extraction → chunk integration.  
  - Semantic and context‑aware chunking (e.g., section embeddings, hierarchical splits).  
  - Production‑grade improvements: FastAPI backend, Dockerization, test suite, evaluation dataset, batch optimization, monitoring, and security hardening.

---

## Conclusion

The TemelIA prototype demonstrates a flexible, high‑precision RAG system for French construction engineering documents. Its modular design, powered by LangChain (and optionally LangGraph), and GDPR‑compliant options make it an excellent starting point for client‑specific extensions and production deployment.  
