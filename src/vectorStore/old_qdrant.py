from qdrant_client import QdrantClient, models
from src.config import QDRANT_URL, QDRANT_API_KEY
import pandas as pd
from pathlib import Path

# Initialize a single Qdrant client and set embedding models
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)



def get_qdrant_client() -> QdrantClient:
    """Returns the Qdrant client instance."""
    return qdrant_client

def create_collection(collection_name: str, recreate: bool = False) -> None:
    """
    Creates a Qdrant collection with the specified name and configuration.
    """
    if recreate and qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)

    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

def upsert_embeddings_from_df(collection: str, df: pd.DataFrame) -> None:
    """
    Upserts text embeddings with metadata into a Qdrant collection.

    Args:
        collection (str): The target collection name.
        df (pd.DataFrame): DataFrame containing columns:
            - chunk_id: unique identifier for each chunk
            - chunk_text: text content to embed
            - filename: source file name
            - page: page number or other source indicator
            - token_count: token count of the chunk
            - metadata: any additional metadata dict or string
            - dense_embedding: dense embedding vector (1024)
            - sparse_embedding: sparse embedding vector (sparse format)
    """
    pass




if __name__ == "__main__":
    # test
    input_csv = Path("data/output_chunked_with_embeddings_and_sparse.csv")
    df = pd.read_csv(input_csv)
    print("Loaded DataFrame with embeddings from CSV.")
    
    







#qdrant_client.set_model("intfloat/multilingual-e5-large-instruct")
#qdrant_client.set_sparse_model("Qdrant/bm25")


# def initialize_qdrant_collection(collection_name: str, recreate: bool = False):
#     """
#     Initializes a Qdrant collection with the specified name and configuration.
#     """
#     if recreate and qdrant_client.collection_exists(collection_name):
#         qdrant_client.delete_collection(collection_name)

#     if not qdrant_client.collection_exists(collection_name):
#         qdrant_client.create_collection(
#             collection_name=collection_name,
#             vectors_config=qdrant_client.get_fastembed_vector_params(),
#             sparse_vectors_config=qdrant_client.get_fastembed_sparse_vector_params(),
#         )



# def add_embeddings(collection: str, df: pd.DataFrame) -> None:
#     """
#     Adds text embeddings with metadata into a Qdrant collection using the 'add' method.

#     Args:
#         collection (str): The target collection name.
#         df (pd.DataFrame): DataFrame containing columns:
#             - chunk_id: unique identifier for each chunk
#             - chunk_text: text content to embed
#             - filename: source file name
#             - page: page number or other source indicator
#             - token_count: token count of the chunk
#             - metadata: any additional metadata dict or string
#     """
#     # Prepare documents, ids, and metadata lists
#     documents = df["chunk_text"].tolist()
#     ids = df["chunk_id"].tolist()

#     # Build metadata dict per document
#     metadata_list = []
#     for _, row in df.iterrows():
#         meta = {
#             "filename": row.get("filename"),
#             "page": row.get("page"),
#             "token_count": row.get("token_count"),
#         }
#         extra = row.get("metadata")
#         if isinstance(extra, dict):
#             meta.update(extra)
#         elif extra is not None:
#             meta["metadata"] = extra
#         metadata_list.append(meta)

#     # Use the add method from QdrantFastembedMixin
#     qdrant_client.add(
#         collection_name=collection,
#         documents=documents,
#         ids=ids,
#         metadata=metadata_list,
#         batch_size=32,
#     )
