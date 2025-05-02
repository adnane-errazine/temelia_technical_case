from src.agent.query_enhancer import enhance_query
from src.agent.rag_answerer import main_rag_llm_answerer
from src.agent.memory import get_chat_history, save_memory
from src.retrieval.retriever import HybridRetriever


from src.config import COHERE_API_KEY
import cohere

from typing import Tuple, List, Dict, Any

## testing purpose only
from src.vectorStore.qdrant import get_qdrant_client
from src.embeddings.dense import DenseEmbedder
##


# Stateful agent orchestrator class, omnipresent in the pipeline
# This class orchestrates the entire pipeline, from query enhancement to answer generation and memory management.
class Orchestrator:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.cohere_client = cohere.Client(COHERE_API_KEY)
        self._enhanced_query = None

    def rerank_chunks(
        self, query: str, hits: List[Dict[str, Any]], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieved chunks using Cohere's reranking API.

        Args:
            query: The user's query (enhanced)
            hits: List of retrieved chunks from vector search
            top_k: Number of top chunks to return after reranking

        Returns:
            List of reranked chunks
        """
        try:
            # Extract documents for reranking
            documents = [hit["payload"]["chunk_text"] for hit in hits]

            # Use Cohere's rerank endpoint
            rerank_results = self.cohere_client.rerank(
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
                model="rerank-v3.5",
            )

            # Create a new list with reranked results
            reranked_hits = []
            for result in rerank_results.results:
                # Access the index property instead of document_index
                original_hit = hits[result.index]
                # Create a new hit with the updated relevance score
                reranked_hit = {
                    "id": original_hit["id"],
                    "payload": original_hit["payload"],
                    "score": result.relevance_score,
                }
                reranked_hits.append(reranked_hit)

            return reranked_hits

        except Exception as e:
            print(f"Reranking failed: {e}")
            # Fallback to original hits if reranking fails
            return hits

    def ask_pipeline(
        self,
        user_question: str,
        qdrant_collection: str,
        dense_limit: int = 30,
        sparse_limit: int = 30,
        top_k_reranker: int = 10,
        filters=None,
    ) -> Tuple[str, list]:
        # 1. Enhance the query
        self._enhanced_query = enhance_query(user_question)

        # 2. Retrieve chunks
        hits = self.retriever.retrieve(
            text=self._enhanced_query,
            qdrant_collection=qdrant_collection,
            dense_limit=dense_limit,
            sparse_limit=sparse_limit,
        )

        # BONUS: reranking relevant chunks through a cross-encoder model (COHERE for this POC)
        reranked_hits = self.rerank_chunks(
            self._enhanced_query, hits, top_k=top_k_reranker
        )

        # 2.1. Extract texts and metadata from hits
        chunks_payload_with_id = [
            {**hit["payload"], "chunk_id": hit["id"], "score": hit["score"]}
            for hit in reranked_hits
        ]

        # 3. Generate answer
        chat_hist = get_chat_history()

        response_gen_rag = main_rag_llm_answerer(
            self._enhanced_query, chunks_payload_with_id, chat_hist
        )

        # Stream chunks to caller
        final_answer = ""
        for partial in response_gen_rag:
            final_answer += partial
            yield partial, chunks_payload_with_id

        # 4. Save to memory
        save_memory(user_question, final_answer)


if __name__ == "__main__":
    # Example question
    dense_embedder = DenseEmbedder()
    qdrant_client = get_qdrant_client()
    COLLECTION_NAME = "test_collection"
    user_question_test = (
        "Quelles sont les exigences minimum requises pour les dispositifs d'isolation?"
    )
    retriever = HybridRetriever(
        qdrant_client=qdrant_client, dense_embedder=dense_embedder
    )
    orchestrator = Orchestrator(retriever=retriever)

    # Handle streaming output
    print("Réponse à la question suivante:", user_question_test)
    print("-" * 50)
    complete_answer = ""
    metadata = None

    for partial_answer, chunks_metadata in orchestrator.ask_pipeline(
        user_question_test
    ):
        print(partial_answer, end="", flush=True)  # Stream to console in real-time
        complete_answer += partial_answer
        if not metadata:  # Only need to capture metadata once
            metadata = chunks_metadata

    print("\n" + "-" * 50)
    # print("Metadata:", metadata)
