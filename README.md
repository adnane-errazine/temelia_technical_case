# Temelion AI

A Retrieval-Augmented Generation (RAG) system specialized for construction and building engineering documents, built with state-of-the-art language models and hybrid search technology.

![Temelion Logo](https://cdn.prod.website-files.com/67e55e84075e7c49410bc67d/67e55f8cb52df7f0df4f0ca2_LOGO-01.jpg)

## Overview

Temelion AI is an intelligent assistant that helps construction engineers and building professionals query technical documents. It uses advanced natural language processing and hybrid search techniques to provide precise answers from complex PDFs, with references to the source material.

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

- **Parser**: Extracts text from PDFs using pdfplumber or Vision Language Models
- **Chunker**: Splits text into semantically coherent, token-aware chunks
- **Embeddings**: 
  - Dense: Multilingual-E5-large-instruct (1024 dimensions)
  - Sparse: Qdrant/bm25 implementation
- **Vector Store**: Qdrant for efficient similarity search
- **Retrieval**: Hybrid search combining dense and sparse results with RRF fusion
- **Reranker**: Cohere's rerank-v3.5 model for improved result relevance
- **LLM**: OpenAI models (gpt-4o-mini) for answer generation
- **Frontend**: Gradio-based interactive web interface

## Installation

### Prerequisites

- Python 3.12+
- [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (for certain dependencies)
- CUDA support recommended for GPU acceleration

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/temelion.git
   cd temelion
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

4. Create a `.env` file in the project root with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   COHERE_API_KEY=your_cohere_api_key
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   QDRANT_COLLECTION_NAME=test_collection
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

- The system uses `gpt-4o-mini` by default
- To change the model, modify the model name in `src/agent/rag_answerer.py`

### Retrieval Parameters

Adjust retrieval parameters in `src/agent/orchestrator.py`:

```python
orchestrator.ask_pipeline(
    user_question=question,
    qdrant_collection=collection_name,
    dense_limit=20,    # Number of dense results to retrieve
    sparse_limit=20,   # Number of sparse results to retrieve
    top_k_reranker=15  # Number of results after reranking
)
```

