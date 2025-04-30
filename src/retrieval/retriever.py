from src.vectorStore.qdrant import get_qdrant_client
from src.embeddings.dense import DenseEmbedder
from qdrant_client import models

class HybridRetriever:
    def __init__(self, qdrant_client, collection_name: str, dense_embedder: DenseEmbedder):
        self.client = qdrant_client
        self.collection = collection_name
        self.dense_embedder = dense_embedder
        self.SPARSE_MODEL = "Qdrant/bm25"
        
        
    def retrieve(self, text: str,dense_limit:int = 50, sparse_limit:int = 50, final_limit:str=None) -> list[dict]:
        """
        Single-call hybrid retrieval: two Prefetches (sparse + dense)
        fused via RRF on the server.
        """
        dense_emb = self.dense_embedder.embed_query(text).tolist()

        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=text, model=self.SPARSE_MODEL),
                    using="sparse",
                    limit=sparse_limit
                ),
                models.Prefetch(
                    query=dense_emb,
                    using="dense",
                    limit=dense_limit
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=final_limit if final_limit else sparse_limit + dense_limit,
        )

        hits = response.points

        return [
            {"id": p.id, "payload": p.payload, "score": p.score}
            for p in hits
        ]

if __name__ == "__main__":
    dense_embedder = DenseEmbedder()
    qdrant_client = get_qdrant_client()
    COLLECTION_NAME = "test_collection"
    retriever = HybridRetriever(qdrant_client= qdrant_client, collection_name= "test_collection", dense_embedder=dense_embedder)

    # Example query:
    question = "Quelles sont les exigences minimum requises pour les dispositifs d’isolation?"
    sparse_hits = retriever.retrieve(
        text=question,
        #dense_limit=4,
        #sparse_limit=4,
    )
    print(len(sparse_hits), "hits found")
    print("Sparse results:\n")
    for hit in sparse_hits:
        print(f"ID: {hit['id']}, Score: {hit['score']}, Payload: {hit['payload']}")
        print("-" * 50)

