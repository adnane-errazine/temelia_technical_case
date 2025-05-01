from fastembed import SparseTextEmbedding
import pandas as pd
from pathlib import Path

# Initialize the model once globally or pass it to the function

class SparseEmbedder:
    def __init__(self, model_name: str = "Qdrant/bm25"):
        """
        Initialize the SparseEmbedder with the specified model.
        
        Args:
            model_name: The name of the sparse embedding model to use.
        """
        self.model = SparseTextEmbedding(model_name=model_name)

    def embed_df_chunked(self,df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Generates sparse embeddings for text chunks in a DataFrame and adds them
        as a new column.

        Args:
            df_input: Input DataFrame with a 'chunk_text' column.

        Returns:
            DataFrame with a new 'sparse_embedding' column containing the embeddings.
            
            # Note: added:
            - 'sparse_indices': list of integer indices
            - 'sparse_values' : list of float weights
        """
        # Ensure 'chunk_text' column exists and is of string type
        if "chunk_text" not in df_input.columns:
            raise ValueError("Input DataFrame must have a 'chunk_text' column")

        # Get the list of texts from the DataFrame column
        texts_to_embed = df_input["chunk_text"].astype(str).tolist()

        # Generate embeddings
        raw_embeddings = list(self.model.embed(texts_to_embed))

        df_input["sparse_embedding"] = raw_embeddings

        ########"
        # Extract indices and values from the sparse embeddings
        indices_list = []
        values_list = []

        for emb in raw_embeddings:
            # e.g. emb.indices is a numpy array of ints
            #      emb.values  is a numpy array of floats
            indices_list.append(emb.indices.tolist())
            values_list.append(emb.values.tolist())
        
        df_input["sparse_indices"] = indices_list
        df_input["sparse_values"] = values_list
        #########
        return df_input

if __name__ == "__main__":
    input_df = Path("data/tempo_output/output_chunked_with_embeddings.csv")
    # Load the DataFrame from a CSV file
    test_df = pd.read_csv(input_df)
    print("Loaded DataFrame with embeddings from CSV.")
    sparse_embedder = SparseEmbedder(model_name="Qdrant/bm25")
    # Generate sparse embeddings
    output_df = sparse_embedder.embed_df_chunked(test_df)
    print("Sparse embeddings generated.")
    # Save the DataFrame with embeddings to a new CSV file
    output_df.to_csv("data/tempo_output/output_chunked_with_embeddings_and_sparse.csv", index=False)
    print("Sparse embeddings added and saved to CSV.")