# Temelion AI

A Retrieval-Augmented Generation (RAG) system specialized for construction and building engineering documents, built with state-of-the-art language models and hybrid search technology.

![Temelion Logo](https://cdn.prod.website-files.com/67e55e84075e7c49410bc67d/67e55f8cb52df7f0df4f0ca2_LOGO-01.jpg)

## Overview

TemelIA is an intelligent assistant that helps construction engineers and building professionals query technical documents. It uses advanced natural language processing and hybrid search techniques to provide precise answers from complex PDFs, with references to the source material.

### Key Features

- **Document Ingestion**: Parse and process PDF documents with high fidelity
- **Hybrid Search**: Combines dense and sparse embeddings for optimal retrieval
- **Cross-encoder Reranking**: Improves result relevance with Cohere's reranking API
- **Multilingual Support**: Optimized for French technical documentation
- **Source Transparency**: Shows the exact source passages used to generate answers
- **Interactive UI**: Clean, intuitive interface built with Gradio

## System Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Document  │────▶│    Parse    │────▶│    Chunk    │────▶│   Embed     │
│   Upload    │     │    PDF      │     │    Text     │     │   Chunks    │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                   │
                                                                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Generate  │◀────│   Rerank    │◀────│   Hybrid    │◀────│   Vector    │
│   Answer    │     │   Results   │     │   Search    │     │   Store     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### Components

- **Parser**: Extracts text from PDFs using various methods:
    - `pdfplumber`: Standard PDF text extraction.
    - `Markitdown`: Markdown conversion for structured text.
    - `Mistral`: LLM-based parsing (potentially OCR or structured extraction).
    - `VLM`: Vision Language Model for complex layouts/images.
- **Chunker**: Splits text into semantically coherent, token-aware chunks using `langchain.text_splitter.RecursiveCharacterTextSplitter`.
- **Embeddings**:
  - Dense: `intfloat/multilingual-e5-large-instruct` (1024 dimensions) via `src/embeddings/dense.py`
  - Sparse: Qdrant's built-in BM25/SPLADE implementation via `src/embeddings/sparse.py`
- **Vector Store**: Qdrant for efficient similarity search (`src/vectorStore/qdrant.py`)
- **Retrieval**: Hybrid search combining dense and sparse results with Reciprocal Rank Fusion (RRF) (`src/retrieval/retriever.py`)
- **Reranker**: Cohere's `rerank-v3.5` model for improved result relevance (`src/agent/rag_answerer.py`)
- **Query Reformulation**: Enhances user queries for better retrieval using a smaller LLM (`gpt-4.1-nano`) via Langchain (`src/agent/query_enhancer.py`)
- **LLM**: OpenAI models (`gpt-4.1` by default) for answer generation, orchestrated with Langchain (`src/agent/rag_answerer.py`)
- **Memory**: Manages conversation history using `langchain.memory.ConversationBufferMemory` (`src/agent/memory.py`).
- **Frontend**: Gradio-based interactive web interface (`app.py`)
- **Langchain Framework**: Used throughout the agent and parsing components for prompt templating (`ChatPromptTemplate`), LLM interaction (`ChatOpenAI`), text splitting, and memory management.

## Installation

### Prerequisites

- Python 3.12+
- [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (Required for building some dependencies like `flash-attn` or `triton` on Windows)
- CUDA support recommended for GPU acceleration

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/adnane-errazine/temelia 
   cd temelion
   ```
2. Install dependencies using [UV](https://github.com/astral-sh/uv):
   ```bash
   # Create a virtual environment (recommended)
   python -m venv .venv

   # Activate the virtual environment
   # On Windows
   .venv\\Scripts\\activate
   # On macOS/Linux
   # source .venv/bin/activate

   # Install UV if you don't have it
   pip install uv

   # Install project dependencies using uv sync (reads pyproject.toml)
   uv sync
   ```
   *Note: Some dependencies might require specific build tools (like VS C++ Build Tools on Windows) or have platform-specific installation steps. Refer to individual package documentation if `uv sync` encounters issues.*

3. Create a `.env` file in the project root with the following variables:
   ```dotenv
   OPENAI_API_KEY=your_openai_api_key
   COHERE_API_KEY=your_cohere_api_key
   QDRANT_URL=your_qdrant_url # e.g., http://localhost:6333 or cloud URL
   QDRANT_API_KEY=your_qdrant_api_key # Optional, depending on Qdrant setup
   QDRANT_COLLECTION_NAME=test_collection # Or your preferred collection name
   ```

## Usage

### Starting the Application

Run the main application:

```bash
python app.py
```

The UI will be available at http://localhost:7860 by default.

### Using the Interface

1. **Document Ingestion**:
   - Enter a username (used for collection management)
   - Upload a PDF document
   - Click "Ingestion du document" and wait for processing to complete

2. **Querying Documents**:
   - Type your question in the text input
   - Click "Envoyer" or press Enter
   - View the generated answer and the source chunks in the sidebar

3. **Resetting Conversation**:
   - Click "Réinitialiser" to start a new conversation

## Example Queries

The system is optimized for technical questions about construction specifications:

- "Quelles sont les exigences minimum requises pour les dispositifs d'isolation?"
- "Fournit un récapitulatif des dispositifs réglementaires applicables dans le document?"
- "A quel phase d'un projet d'isolation doit être remis le rapport d'étanchéité de l'air?"
- "Quels sont les différents livrables du management de la qualité en phase DCE?"
- "Quels sont les cibles prioritaires HQE pour les crèches?"

## Configuration

### Embedding Models

- Dense embeddings use `intfloat/multilingual-e5-large-instruct` by default
- To change the model, modify the `model_name` parameter in `src/embeddings/dense.py`

### LLM Settings

- The system uses `gpt-4.1` by default
- To change the model, modify the model name in `src/agent/rag_answerer.py`

### Retrieval Parameters

Adjust retrieval parameters in `src/agent/orchestrator.py`:

```python
orchestrator.ask_pipeline(
    user_question=question,
    qdrant_collection=collection_name,
    dense_limit=40,    # Number of dense results to retrieve
    sparse_limit=40,   # Number of sparse results to retrieve
    top_k_reranker=20  # Number of results after reranking
)
```

### Chunking Strategy

- **Method**: The text is split into chunks using `langchain.text_splitter.RecursiveCharacterTextSplitter` based on the `tiktoken` tokenizer for the specified model.
- **Default Parameters** (in `src/ingestion/ingest_pdf.py`):
    - `model_name`: "gpt-4o" (used for token counting)
    - `chunk_size`: 256 tokens
    - `chunk_overlap`: 60 tokens
- **Customization**: To change the chunking strategy, modify the parameters passed to the `chunk_texts` function within the `process_pdf` function in `src/ingestion/ingest_pdf.py`.

```python
# Example from src/ingestion/ingest_pdf.py
chunked_df = chunk_texts(
    parsed_df,
    model_name="gpt-4o", # Tokenizer model
    chunk_size=256,      # Max tokens per chunk
    chunk_overlap=60,    # Token overlap between chunks
)
```

