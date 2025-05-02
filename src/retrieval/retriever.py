from src.vectorStore.qdrant import get_qdrant_client
from src.embeddings.dense import DenseEmbedder
from qdrant_client import models


class HybridRetriever:
    def __init__(self, qdrant_client, dense_embedder: DenseEmbedder):
        self.client = qdrant_client
        self.dense_embedder = dense_embedder
        self.SPARSE_MODEL = "Qdrant/bm25"

    def retrieve(
        self,
        text: str,
        qdrant_collection: str,
        dense_limit: int = 50,
        sparse_limit: int = 50,
        final_limit: str = None,
    ) -> list[dict]:
        """
        Perform a hybrid retrieval of documents using sparse and dense embeddings
        fused via Reciprocal Rank Fusion (RRF).
        Args:
            text (str): The query string to retrieve relevant documents for.
            dense_limit (int, optional): Maximum number of dense embedding matches to fetch.
                Defaults to 50.
            sparse_limit (int, optional): Maximum number of sparse embedding matches to fetch.
                Defaults to 50.
            final_limit (int | None, optional): Maximum number of total results to return
                after fusion. If None, defaults to sparse_limit + dense_limit.
        Returns:
            list[dict]: A list of result dictionaries, each containing:
                - "id": Unique identifier of the document.
                - "payload": The stored document payload.
                - "score": Fusion score indicating relevance.
        """
        dense_emb = self.dense_embedder.embed_query(text).tolist()

        response = self.client.query_points(
            collection_name=qdrant_collection,
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=text, model=self.SPARSE_MODEL),
                    using="sparse",
                    limit=sparse_limit,
                ),
                models.Prefetch(query=dense_emb, using="dense", limit=dense_limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=final_limit if final_limit else sparse_limit + dense_limit,
        )

        hits = response.points

        return [{"id": p.id, "payload": p.payload, "score": p.score} for p in hits]


if __name__ == "__main__":
    dense_embedder = DenseEmbedder()
    qdrant_client = get_qdrant_client()
    COLLECTION_NAME = "test_collection"
    retriever = HybridRetriever(
        qdrant_client=qdrant_client, dense_embedder=dense_embedder
    )

    # Example query:
    question = (
        "Quelles sont les exigences minimum requises pour les dispositifs d’isolation?"
    )
    sparse_hits = retriever.retrieve(
        text=question,
        qdrant_collection=COLLECTION_NAME,
        # dense_limit=4,
        # sparse_limit=4,
    )
    print(len(sparse_hits), "hits found")
    print("Sparse results:\n")
    for hit in sparse_hits:
        print(f"ID: {hit['id']}, Score: {hit['score']}, Payload: {hit['payload']}")
        print("-" * 50)
