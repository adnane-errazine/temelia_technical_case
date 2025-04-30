# src/agent/orchestrator.py
from src.agent.query_enhancer import enhance_query
from src.agent.answerer import generate_answer
from src.agent.memory import get_chat_history, save_memory
from typing import Tuple

def ask_pipeline(user_question: str) -> Tuple[str, list]:
    # 1. Enhance the query
    enhanced_q = enhance_query(user_question)

    # 2. Retrieve chunks
    hits = retrieve_chunks(enhanced_q, top_k=20)
    texts = [h.text for h in hits]
    metas = [h.metadata for h in hits]

    # 3. Generate answer
    chat_hist = get_chat_history()
    answer = generate_answer(enhanced_q, texts, chat_hist)

    # 4. Save to memory
    save_memory(user_question, answer)

    # Return answer + metadata for UI
    return answer, metas
