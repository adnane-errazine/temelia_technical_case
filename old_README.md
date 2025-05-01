

# Proposed solution


## Parsing phase
1. Input document (installed dependencies onloy for PDF and DOCX but it's possible to parse Images(OCR), HTML, etc)
2. document to text: Markitdown tool  (Quick win, 80/20 rule)
3. Embedding



## RAG
- Hybrid search Approach
- Dense embedder: Multilingual-E5-large-instruct (Supports French, Frugal, best ratio performance to size -> suitable for our chunking range, limits, it might be too small if we choose a more exhaustive chunking strategy)
- Sparse embedder: Qdrant/bm25 (Library: SparseTextEmbedding)
- Vector store: Qdrant (French -> RGPD)
- dense embeddings, vector store : qdrant
- sparse embeddings BM25, qdrant too?
- Reranker-> Cohere reranker (#TODO)


## Agentic workflow 
0. User query
1. rephrase / Enhance user question if necessary (Out of scope but a good option: data augmentation, Query enrichment through thesaurus / domain knowledge)
2. Retrieve Best 50 matches from the dense embeddings
3. Retrieve Best 50 matches from the sparse embeddings
4. Combine matches then use a reranker to output best 15 match
5. Invoke LLM to answer the query given the context and offer the references.
6. Output the answer


## Front
- Technology: Gradio
- visualize the chosen chunks on the side







## Out of scope, but interesting:
- Context chunking
- Advanced parsing (save images, Embbed images, non-text classifier, Extra verification by reordering)
- Batch optimization in models when possible.


-----------------------------------------------------
## assignment
The following are the instructions, 
I will be using langchain and perhaps langgraph to handle some tasks . 
I will use hybrid search, and store both dense and sparse embeddings in qdrant bm25 (if it's doable)

Project: 

1) Build a RAG app that lets a user upload a (complex) PDF document and ask questions about it. 

(2) Write a 1-page report explaining which methods you tried, what failed, what worked better, on which conditions: we are interested how you approached this case and any detail you find interesting.

## Tech Stack

Use the frameworks and languages you’re comfortable with, although we have a preference for:

- Backend: Python, with libraries/frameworks of your choice
- Frontend: streamlit or gradio
- APIs: you’re allowed to use commercial (free-tier) or open-source APIs
- Models: you can use any model you like, local or remote






Sample questions (answers are in the doc so you can test your app):

- Quelles sont les exigences minimum requises pour les dispositifs d’isolation?
- Fournit un récapitulatif des dispositifs réglementaires applicables dans le document (de préférence sous forme de matrice) ?
- A quel phase d’un projet d’isolation doit être remis le rapport d’étanchéité de l’air ?
- Quels sont les différents livrables du management de la qualité en phase DCE?
- Quels sont les cibles prioritaires HQE pour les crèches?
- Fournit un récapitulatif des comptages chauffage au 30 mai 2016 au format JSON.



During the demo, we will test your demo with some non-disclosed questions on the demo PDF file and also would like to test using another PDF document. We will also check document indexing speed, search speed, etc.

-----------------------------------------------------
-----------------------------------------------------












# TODO 

requirements: # uv bugs
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

uv pip install -U bitsandbytes

https://visualstudio.microsoft.com/visual-cpp-build-tools/
uv pip install flash-attn --no-build-isolation
https://github.com/Dao-AILab/flash-attention/issues/1038

Prepared 1 package in 392m 15s
Installed 1 package in 122ms
 + flash-attn==2.7.3

 pip install -U "triton-windows<3.4"



 https://python-client.qdrant.tech/qdrant_client.qdrant_fastembed.html#  (why is query used and not seach)


 https://qdrant.tech/articles/hybrid-search/



VLM   / parse output UUID transformer / explicitly add to prompt, if question is unrelated. output: out of bound







