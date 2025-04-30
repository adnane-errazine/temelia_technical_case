import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from typing import List
from pathlib import Path

class DenseEmbedder:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large-instruct",
        device: str = None,
        max_length: int = 512,
        batch_size: int = 8,
    ):
        """
        Initialize the embedder with the E5 model.
        
        Args:
            model_name: The name of the model to use
            device: The device to run the model on ('cuda', 'cpu', etc.)
            max_length: Maximum token length for the model
            batch_size: Batch size for processing multiple texts at once
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Set device (use CUDA if available)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading model {model_name} on {self.device}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        
        print(f"Model loaded successfully!")
    
    def _average_pool(self, last_hidden_states, attention_mask):
        """
        Average pooling operation to get sentence embeddings        
        Args:
            last_hidden_states: The last hidden states from the transformer model
            attention_mask: The attention mask from tokenization
            
        Returns:
            Tensor of pooled embeddings
        """
        # Mask out padding tokens
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        
        # Sum and divide by the number of non-padding tokens
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def embed_query(self, query: str, task_description: str = None) -> np.ndarray:
        """
        Generate embeddings for a single query.
        
        Args:
            query: The query text
            task_description: Optional task description for the query
            
        Returns:
            Numpy array containing the embedding vector
        """
        
        # Use default task description if none provided
        if task_description is None:
            task_description = "Étant donné une requête en français, récupérez les passages pertinents qui répondent à la question"
        formatted_query = f"Instruct: {task_description}\nQuery: {query}"

        # Tokenize
        inputs = self.tokenizer(
            formatted_query,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**inputs)
        
        # Apply average pooling and normalize (using the exact method from model card)
        batch_embeddings = self._average_pool(model_output.last_hidden_state, inputs["attention_mask"])
        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        
        # Convert to numpy and store
        batch_embeddings = batch_embeddings.detach().cpu().numpy()
        return batch_embeddings[0]  # Return the first (and only) embedding in the batch
    
    def embed_df_chunked(
        self,
        df_chunked: pd.DataFrame,
        text_col: str = "chunk_text",
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Embed a DataFrame of text chunks, adding an 'embedding' column.

        Args:
            df_chunked: DataFrame with at least the columns
                        ['chunk_id','chunk_text','page','filename','token_count','metadata'].
            text_col:   Name of the column containing the text to embed.
            show_progress: Whether to display a tqdm progress bar.

        Returns:
            A new DataFrame with an additional 'embedding' column, where each entry
            is a numpy array (dtype=float32) of shape (hidden_size,).
        """
        texts = df_chunked[text_col].astype(str).tolist()
        embeddings: List[np.ndarray] = []

        # Create batch indices
        batch_starts = list(range(0, len(texts), self.batch_size))
        iterator = batch_starts
        if show_progress:
            iterator = tqdm(batch_starts, desc="Generating embeddings", unit="batch")

        for start in iterator:
            batch_texts = texts[start : start + self.batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Pool, normalize, convert to numpy
            pooled = self._average_pool(outputs.last_hidden_state, inputs["attention_mask"])
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            batch_embeddings = normalized.cpu().numpy()

            embeddings.extend(batch_embeddings)

        # sanity check
        assert len(embeddings) == len(df_chunked), \
            f"Expected {len(df_chunked)} embeddings, got {len(embeddings)}"

        # Attach embeddings column
        df_out = df_chunked.copy()
        df_out["dense_embedding"] = embeddings

        return df_out


if __name__ == "__main__":
    # Initialize the embedder
    embedder = DenseEmbedder()
    
    # Example query with task description (as per model card)
    query = "Quelles sont les exigences minimum requises pour les dispositifs d’isolation?"
    task_description = "Étant donné une requête en français, récupérez les passages pertinents qui répondent à la question"
    # task_description = "Given a query in any language, retrieve relevant passages that answer the query"
    # task_description = "Given a web search query, retrieve relevant passages that answer the query"
    # task_description = "Given a query, retrieve relevant passages that answer the query"
    embedding = embedder.embed_query(query, task_description)
    print(f"Query: {query}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding: {embedding}")
    print(f"Embedding dtype: {embedding.dtype}")
    print("-" * 50)
    # Embed the DataFrame
    input_csv = Path("data/tempo_output/output_chunked.csv")
    df = pd.read_csv(input_csv)
    df_chunked = embedder.embed_df_chunked(df, text_col="chunk_text", show_progress=True)
    # 2) Convert embeddings to lists
    df_chunked["dense_embedding"] = df_chunked["dense_embedding"].apply(lambda arr: arr.tolist())
    # 3) Save as CSV 
    output_csv = "data/tempo_output/output_chunked_with_embeddings.csv"
    df_chunked.to_csv(output_csv, index=False)
    print(f"Embeddings saved to {output_csv}")