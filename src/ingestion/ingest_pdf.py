from src.parsing.parser_pdfplumber import parse_document as pdfplumber_parse
from src.parsing.parser_VLM import parse_document as vlm_parse
from src.parsing.chunker import chunk_texts
from src.vectorStore.qdrant import create_collection, upsert_embeddings_from_df

from pathlib import Path
import pandas as pd
import uuid
import os
import ast
import time
from typing import Generator


def process_pdf(file_path: str, username: str,dense_embedder,sparse_embedder, gpu:bool=False) -> Generator:
    """
    Process a PDF document through the entire pipeline:
    1. Parse the PDF to extract text
    2. Chunk the text into smaller segments
    3. Generate dense and sparse embeddings
    4. Store in Qdrant vector database
    
    Args:
        file_path: Path to the PDF file
        username: Username for collection management/tracking
        
    Yields:
        Status messages throughout the processing
    """
    try:
        start_time = time.time()
        file_path = Path(file_path)
        yield f"Starting processing of: {file_path.name}"
        print(f"Starting processing of: {file_path.name}")
        
        # Dictionary to store execution times
        execution_times = {}
        
        # Create user-specific collection name
        collection_name = username
        
        # Step 1: Parse PDF - choose parser based on file size or complexity
        parse_start = time.time()
        yield f"Parsing PDF using {'VLM' if gpu else 'pdfplumber'}..."
        if gpu:
            # Use VLM parser for larger documents or complex parsing
            parsed_df = vlm_parse(file_path)
        else:
            # Use pdfplumber for smaller documents or simpler parsing
            parsed_df = pdfplumber_parse(file_path)
        
        parse_time = time.time() - parse_start
        execution_times['parsing'] = parse_time
        yield f"Successfully parsed {len(parsed_df)} pages in {parse_time:.2f}s"
        print(f"PDF parsing complete: {parse_time:.2f}s")
        
        # Step 2: Chunk the text
        chunk_start = time.time()
        yield f"Chunking the text with appropriate sizing..."
        model_name = "gpt-4o"  # For token counting
        chunked_df = chunk_texts(
            parsed_df, 
            model_name=model_name,
            chunk_size=256,  # Adjust based on your needs
            chunk_overlap=30
        )
        chunk_time = time.time() - chunk_start
        execution_times['chunking'] = chunk_time
        yield f"Created {len(chunked_df)} text chunks in {chunk_time:.2f}s"
        print(f"Text chunking complete: {chunk_time:.2f}s")
        
        # Step 3: Generate dense embeddings
        dense_start = time.time()
        yield f"Generating dense embeddings..."
        df_with_dense = dense_embedder.embed_df_chunked(chunked_df)
        dense_time = time.time() - dense_start
        execution_times['dense_embeddings'] = dense_time
        yield f"Dense embeddings complete in {dense_time:.2f}s"
        print(f"Dense embeddings complete: {dense_time:.2f}s")
        
        # Step 4: Generate sparse embeddings
        sparse_start = time.time()
        yield f"Generating sparse embeddings..."
        df_with_sparse = sparse_embedder.embed_df_chunked(df_with_dense)
        sparse_time = time.time() - sparse_start
        execution_times['sparse_embeddings'] = sparse_time
        yield f"Sparse embeddings complete in {sparse_time:.2f}s"
        print(f"Sparse embeddings complete: {sparse_time:.2f}s")
        
        # Step 5: Prepare vector database
        db_start = time.time()
        yield f"Preparing Qdrant collection: {collection_name}"
        create_collection(collection_name, recreate=False)
        
        # Step 6: Store embeddings
        yield f"Storing document embeddings in vector database..."
        upsert_embeddings_from_df(
            collection=collection_name,
            df_input=df_with_sparse,
            batch_size=50,
        )
        db_time = time.time() - db_start
        execution_times['vectorstore_operations'] = db_time
        yield f"Database operations complete in {db_time:.2f}s"
        print(f"Database operations complete: {db_time:.2f}s")
        
        # Optional: Save processed data to CSV for debugging/analysis in tmp\username\data
        save_start = time.time()
        output_dir = Path("tmp") / username / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{file_path.stem}_processed.csv"
        df_with_sparse.to_csv(output_file, index=False)
        save_time = time.time() - save_start
        execution_times['csv_export'] = save_time
        print(f"CSV export complete: {save_time:.2f}s")
        
        total_time = time.time() - start_time
        yield f"✅ Document successfully processed and stored in collection: {collection_name}"
        yield f"Total processing time: {total_time:.2f}s"
        
        # Detailed execution time breakdown
        print(f"✅ Total processing time: {total_time:.2f}s")
        print("Execution time breakdown:")
        print(f"  - PDF parsing: {execution_times['parsing']:.2f}s ({(execution_times['parsing']/total_time*100):.1f}%)")
        print(f"  - Text chunking: {execution_times['chunking']:.2f}s ({(execution_times['chunking']/total_time*100):.1f}%)")
        print(f"  - Dense embeddings: {execution_times['dense_embeddings']:.2f}s ({(execution_times['dense_embeddings']/total_time*100):.1f}%)")
        print(f"  - Sparse embeddings: {execution_times['sparse_embeddings']:.2f}s ({(execution_times['sparse_embeddings']/total_time*100):.1f}%)")
        print(f"  - Database operations: {execution_times['vectorstore_operations']:.2f}s ({(execution_times['vectorstore_operations']/total_time*100):.1f}%)")
        print(f"  - CSV export: {execution_times['csv_export']:.2f}s ({(execution_times['csv_export']/total_time*100):.1f}%)")
        
    except Exception as e:
        #import traceback
        yield f"❌ Error during processing: {str(e)}"
        #yield traceback.format_exc()
        raise