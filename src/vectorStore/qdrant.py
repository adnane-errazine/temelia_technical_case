from qdrant_client import QdrantClient, models
from src.config import QDRANT_URL, QDRANT_API_KEY 
import pandas as pd
from pathlib import Path
import ast     # Added to parse string representations of lists/dicts

try:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
    )
except Exception as e:
    print(f"Error initializing Qdrant client: {e}")
    raise


def get_qdrant_client() -> QdrantClient:
    """Returns the Qdrant client instance."""
    # Check if client was initialized successfully
    if qdrant_client is None:
         raise ConnectionError("Qdrant client is not initialized. Check configuration and connection.")
    return qdrant_client

def create_collection(collection_name: str, recreate: bool = False) -> None:
    """
    Creates a Qdrant collection with the specified name and configuration.
    """
    try:
        client = get_qdrant_client() # Get client, raises ConnectionError if not initialized
        collection_exists = client.collection_exists(collection_name)

        if recreate and collection_exists:
            print(f"Deleting existing collection: {collection_name}...")
            client.delete_collection(collection_name)
            print(f"Collection '{collection_name}' deleted.")
            collection_exists = False # Update status

        if not collection_exists:
            print(f"Creating collection: {collection_name}...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                },
                # Consider adding optimizers_config_diff for performance tuning if needed
            )
            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Collection '{collection_name}' already exists.")

    except ConnectionError as ce:
         print(f"ERROR: Could not connect to Qdrant for collection operations: {ce}", file=sys.stderr)
         raise # Re-raise connection error
    except Exception as e:
        print(f"ERROR: Failed during collection creation/check for '{collection_name}': {e}", file=sys.stderr)
        raise # Re-raise other exceptions

def upsert_embeddings_from_df(collection: str, df_input: pd.DataFrame, batch_size: int = 100) -> None:
    """
    Upserts text embeddings with metadata into a Qdrant collection from a DataFrame.
    Handles both dense and sparse embeddings.

    Args:
        collection (str): The target collection name.
        df (pd.DataFrame): DataFrame containing columns:
            - chunk_id: unique identifier (int or UUID string) for each chunk
            - chunk_text: text content to embed
            - filename: source file name
            - page: page number or other source indicator
            - token_count: token count of the chunk
            - metadata: additional metadata (dict or string representation of dict)
            - dense_embedding: dense embedding vector (list/string of floats, size 1024)
            - sparse_embedding: sparse embedding vector
            - sparse_indices: list of integer indices (optional)
            - sparse_values: list of float weights (optional)
        batch_size (int): Number of points to upsert in each batch. Defaults to 100.
    """
    qdrant_client = get_qdrant_client()
    print(f"Starting upsert process for collection '{collection}' with {len(df_input)} rows.")
    batch = []
    for _, row in df_input.iterrows():
        # Prepare dense list
        dense_vec = row["dense_embedding"]
        # Prepare sparse vectors
        sparse_vec = models.SparseVector(indices=list(row["sparse_indices"]),
                                         values=list(row["sparse_values"]))
        # Build point
        point = models.PointStruct(
            id=row["chunk_id"],
            vector={
                "dense": dense_vec,
                "sparse": sparse_vec
            },
            payload={
                "chunk_text": row.get("chunk_text", ""),
                "filename": row.get("filename", ""),
                "page": row.get("page", -1),
                "token_count": row.get("token_count", 0),
                "metadata": row.get("metadata", {}),
            }
        )
        batch.append(point)
        if len(batch) >= batch_size:
            qdrant_client.upsert(collection_name=collection, points=batch, wait=True)
            batch = []
    if batch:
        qdrant_client.upsert(collection_name=collection, points=batch, wait=True)
    print(f"Upserted {len(df_input)} points into '{collection}'")


if __name__ == "__main__":

    input_csv = Path("data/tempo_output/output_chunked_with_embeddings_and_sparse.csv")
    TEST_COLLECTION_NAME= "test_collection"
    df = pd.read_csv(input_csv, converters={
                "dense_embedding": ast.literal_eval,
                "sparse_indices":  ast.literal_eval,
                "sparse_values":   ast.literal_eval,
            })

    print(f"Loaded DataFrame with {len(df)} rows from {input_csv}.")

    # --- Qdrant Setup ---
    print(f"Attempting to create/recreate collection: {TEST_COLLECTION_NAME}")
    # Set recreate=True to ensure a clean state for testing
    create_collection(TEST_COLLECTION_NAME, recreate=True)

    # --- Perform Upsert ---
    print("Starting upsert...")
    # Use a smaller batch size for small test data to see batching logs
    upsert_embeddings_from_df(TEST_COLLECTION_NAME, df, batch_size=2)
    print("Upsert finished.")
    
    
    


