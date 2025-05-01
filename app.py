import gradio as gr
import json
from typing import List, Dict, Any
import os
from pathlib import Path

from src.vectorStore.qdrant import get_qdrant_client
from src.embeddings.dense import DenseEmbedder
from src.retrieval.retriever import HybridRetriever
from src.agent.orchestrator import Orchestrator
from src.ingestion.ingest_pdf import process_pdf
from src.embeddings.sparse import SparseEmbedder
# Global orchestrator instance
_ORCHESTRATOR = None
_dense_embedder = None
_sparse_embedder = None

def get_dense_embedder():
    """Lazy initialization of the DenseEmbedder"""
    global _dense_embedder
    if _dense_embedder is None:
        _dense_embedder = DenseEmbedder()
    return _dense_embedder

def get_sparse_embedder():
    """Lazy initialization of the SparseEmbedder"""
    global _sparse_embedder
    if _sparse_embedder is None:
        _sparse_embedder = SparseEmbedder()
    return _sparse_embedder


def initialize_rag():
    """Initialize the RAG components (keep your existing implementation)"""
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        print("Initializing RAG components...")
        dense_embedder = get_dense_embedder()
        qdrant_client = get_qdrant_client()
        retriever = HybridRetriever(
            qdrant_client=qdrant_client,
            dense_embedder=dense_embedder
        )
        _ORCHESTRATOR = Orchestrator(retriever=retriever)
    return _ORCHESTRATOR

def ingest_pdf(file_obj, username: str):
    if file_obj is None:
        yield "Aucun fichier téléchargé."
        return
    if not username:
        yield "⚠️ Merci de saisir un nom d'utilisateur."
        return

    user_dir = Path("tmp") / username
    user_dir.mkdir(parents=True, exist_ok=True)
    yield f"🔄 Dossier créé : {user_dir}"

    src = Path(file_obj.name)
    dest = user_dir / src.name
    os.replace(src, dest)
    yield f"📁 Fichier enregistré : {dest}"

    try:
        dense_embedder = get_dense_embedder()
        sparse_embedder = get_sparse_embedder()
        for status in process_pdf(file_path=str(dest), username=username, dense_embedder=dense_embedder, sparse_embedder=sparse_embedder):
            yield status
        yield f"✅ Ingestion réussie pour {dest.name}."
    except Exception as e:
        print(f"Error during ingestion: {e}")
        yield f"❌ Erreur pendant l'ingestion : {e}"
        
def format_chunk_html(chunk: Dict[str, Any], index: int) -> str:
    """Simplified chunk formatting without the show more/less feature"""
    chunk_id = chunk.get('chunk_id', 'N/A')
    filename = chunk.get('filename', 'Unknown')
    page = chunk.get('page', '-')
    chunk_text = chunk.get('chunk_text', 'No text available')
    chunk_score = chunk.get('score', 0.0)
    
    # Determine color based on score
    if chunk_score > 0.7:  # Above 70%
        color_class = "high-relevance"
        score_label = f"Score: {chunk_score:.2f} (Élevé)"
    elif chunk_score > 0.3:  # Between 30% and 70%
        color_class = "medium-relevance"
        score_label = f"Score: {chunk_score:.2f} (Moyen)"
    else:  # Below 30%
        color_class = "low-relevance"
        score_label = f"Score: {chunk_score:.2f} (Faible)"
    
    html = f"""
    <div class="chunk-card {color_class}" id="chunk-{index}">
        <div class="chunk-header">
            <div class="source-title">
                <span class="source-badge">Source {index}</span>
                <span class="filename">{filename}</span>
                <span class="page">Page {page}</span>
            </div>
            <div class="chunk-meta">
                <span class="chunk-id">ID: {chunk_id[:6]}...</span>
                <span class="relevance-score">{score_label}</span>
            </div>
        </div>
        <div class="chunk-content">
            <div class="text-full">{chunk_text}</div>
        </div>
    </div>
    """
    return html

def chat(message: str, history: List[List[str]], chunks_state, username: str):
    if not username:
        warning = "⚠️ Merci de saisir un nom d'utilisateur avant de poser une question."
        # we keep the chat enabled so they can correct it:
        return gr.update(value=""), history + [[None, warning]], chunks_state
    if not message.strip():
        return gr.update(value=""), history, chunks_state

    orchestrator = initialize_rag()
    yield gr.update(value="", interactive=False), history + [[message, ""]], None

    try:
        streamed_answer = ""
        for partial, chunks in orchestrator.ask_pipeline(user_question=message, qdrant_collection=username, dense_limit=20, sparse_limit=20, top_k_reranker=15):
            streamed_answer += partial
            new_history = history + [[message, streamed_answer]]
            yield gr.update(value="", interactive=False), new_history, None

        # Sort chunks by score in descending order
        chunks = sorted(chunks, key=lambda x: x.get('score', 0.0), reverse=True)
        formatted_chunks = [format_chunk_html(c, i) for i, c in enumerate(chunks, 1)]
        chunks_state = json.dumps(formatted_chunks)

    except Exception as e:
        error = f"⚠️ Erreur: {str(e)}"
        yield gr.update(value="", interactive=True), history + [[message, error]], None
        return

    yield gr.update(value="", interactive=True), new_history, chunks_state


def reset_conversation():
    initial_message = "Bonjour ! Je suis TemelIA, l'assistant expert développé par Temelion. Comment puis-je vous aider aujourd'hui ?"
    return [[None, initial_message]], None, ""

custom_css = """
:root {
    --primary: #126EED;
    --primary-light: #3a8cff;
    --secondary: #f0f7ff;
    --background: white;
    --text: #2B3A67;
    --text-light: #666;
    --border: #e8e8e8;
    --success: #4CAF50;
    --error: #F44336;
    --warning: #FF9800;
    --high-relevance-bg: rgba(76, 175, 80, 0.1);
    --high-relevance-border: rgba(76, 175, 80, 0.5);
    --medium-relevance-bg: rgba(255, 152, 0, 0.1);
    --medium-relevance-border: rgba(255, 152, 0, 0.5);
    --low-relevance-bg: rgba(244, 67, 54, 0.1);
    --low-relevance-border: rgba(244, 67, 54, 0.5);
}

body {
    font-family: 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    background: var(--background);
    margin: 0;
    padding: 0;
    color: var(--text);
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 1.5rem;
    gap: 1.5rem;
}

.header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 1rem;
    background: var(--background);
    border-bottom: 1px solid var(--border);
}

.header img {
    height: 80px;
    margin-bottom: 1rem;
}

.main-row {
    display: flex;
    gap: 1.5rem;
    height: calc(100vh - 200px);
}

#chatbox {
    flex: 3;
    display: flex;
    flex-direction: column;
    background: white;
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.chat-input-container {
    padding: 1rem;
    background: white;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 0.5rem;
}

#sidebar {
  display: flex;
  flex-direction: column;  /* stack vertically */
  gap: 1.5rem;
  padding: 1.5rem;
}


#chunks-container {
  flex: 1 1 auto;          
  max-height: 600px;       
  overflow-y: auto;       
  overflow-x: hidden;
}


#chunks-container .chunk-card {
  box-sizing: border-box;
  width: 100%;
}

.gr-chat-message {
    max-width: 85%;
    padding: 0.8rem 1.2rem;
    border-radius: 18px;
    line-height: 1.5;
}

.gr-chat-message.user {
    background: var(--secondary);
    border: 1px solid var(--border);
    align-self: flex-end;
}

.gr-chat-message.bot {
    background: var(--primary);
    color: white;
    align-self: flex-start;
}

.gr-textbox {
    border-radius: 24px;
    padding: 0.8rem 1.2rem;
    border: 1px solid var(--border);
}

.gr-button {
    border-radius: 24px;
    transition: all 0.2s;
    padding: 0.5rem 1.5rem;
}

.gr-button-primary {
    background: var(--primary) !important;
    color: white !important;
}

.gr-button-primary:hover {
    background: var(--primary-light) !important;
}

.gr-button-secondary {
    background: var(--secondary) !important;
    color: var(--primary) !important;
    border: 1px solid var(--primary) !important;
}

.chunk-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.chunk-card.high-relevance {
    background-color: var(--high-relevance-bg);
    border-color: var(--high-relevance-border);
}

.chunk-card.medium-relevance {
    background-color: var(--medium-relevance-bg);
    border-color: var(--medium-relevance-border);
}

.chunk-card.low-relevance {
    background-color: var(--low-relevance-bg);
    border-color: var(--low-relevance-border);
}

.chunk-header {
    padding: 1rem;
    border-bottom: 1px solid var(--border);
    background: var(--secondary);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.source-badge {
    background: var(--primary);
    color: white;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8em;
    margin-right: 0.5rem;
}

.relevance-score {
    font-size: 0.85em;
    font-weight: 500;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
}

.high-relevance .relevance-score {
    color: var(--success);
}

.medium-relevance .relevance-score {
    color: var(--warning);
}

.low-relevance .relevance-score {
    color: var(--error);
}

.chunk-content {
    padding: 1rem;
}

/* Add responsive design and other styles from original code as needed */
"""

def add_js_to_chunks(state):
    """Simplified function to render chunks without JS toggles"""
    if not state:
        return "<p style='color: #666; text-align: center;'>Aucune source utilisée</p>"
    
    chunks_html = "".join(json.loads(state))
    
    return f"""
    <div id="chunks-wrapper">
        {chunks_html}
    </div>
    """

def main():
    initialize_rag()
    
    with gr.Blocks(
        css=custom_css,
        theme=gr.themes.Soft(primary_hue="blue"),
        title="Temel-IA",
        head=f'''
        <img src="https://cdn.prod.website-files.com/67e55e84075e7c49410bc67d/67e55f8cb52df7f0df4f0ca2_LOGO-01.jpg" alt="Logo Temelion">
        '''
    ) as demo:
        # Header with logo
        gr.HTML(f"""
        <div class="header">
        <center>
            <img src="https://cdn.prod.website-files.com/67e55e84075e7c49410bc67d/67e55f8cb52df7f0df4f0ca2_LOGO-01.jpg" alt="Logo Temelion">
        </center>
        </div>
        """)
        
        chunks_state = gr.State()
        
        with gr.Row(equal_height=False):
            with gr.Column():
                gr.Markdown("### 📄 Ingestion de documents")
                
                username_input = gr.Textbox(label="Nom d'utilisateur", placeholder="Ex: admin", interactive=True)
                
                file_upload = gr.File(
                    label="📄 Importer un fichier PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
                ingest_button = gr.Button("📥 Ingestion du document")
                ingestion_status = gr.Textbox(label="📢 Statut de l'ingestion", interactive=False)
            
            # Chat Column
            with gr.Column(scale=3, elem_id="chatbox"):
                chatbot = gr.Chatbot(
                    value=[[None, "Bonjour ! Je suis TemelIA, l'assistant expert développé par Temelion. Comment puis-je vous aider aujourd'hui ?"]],
                    height=600,
                    avatar_images=(
                        "src/assets/user_logo.png", 
                        "src/assets/bot.png"
                    ),
                    show_copy_button=True,
                    elem_id="chatbot",
                )
                
                with gr.Row(elem_classes="chat-input-container"):
                    msg = gr.Textbox(
                        placeholder="Écrivez votre question ici...",
                        autofocus=True,
                        max_lines=3,
                        container=False,
                        scale=8
                    )
                    submit_btn = gr.Button("Envoyer", variant="primary")
                    reset_btn = gr.Button("Réinitialiser", variant="secondary")
            
            # Sidebar Column
            with gr.Column(scale=2, elem_id="sidebar"):
                with gr.Row():
                    sidebar_header = gr.Markdown("### 📚 Sources utilisées")
                with gr.Row(elem_id="chunks-container"):
                    chunks_html = gr.HTML(
                        "<p style='color: #666; text-align: center;'>Les sources apparaîtront ici après question</p>"
                    )
                        
                        
        # Event Handlers
        msg.submit(
            chat,
            inputs=[msg, chatbot, chunks_state, username_input],
            outputs=[msg, chatbot, chunks_state],
            queue=True
        )
        
        submit_btn.click(
            chat,
            inputs=[msg, chatbot, chunks_state, username_input],
            outputs=[msg, chatbot, chunks_state],
            queue=True
        )
        
        reset_btn.click(
            reset_conversation,
            inputs=[],
            outputs=[chatbot, chunks_state, msg]
        )
        
        chunks_state.change(
            lambda x: add_js_to_chunks(x),
            inputs=[chunks_state],
            outputs=[chunks_html]
        )
        
        ingest_button.click(
            fn=ingest_pdf,
            inputs=[file_upload, username_input],
            outputs=[ingestion_status],
        )

    demo.queue().launch(server_name="0.0.0.0", share=False, show_error=True, debug=True)

if __name__ == "__main__":
    main()