from pathlib import Path
import pandas as pd
import uuid
from typing import Callable

from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken


def num_tokens_from_string(string: str, model_name: str)-> int:
    """
    Returns the number of tokens in a text string using the provided encoder.
    """
    # Get the encoder for the specified model
    encoder = tiktoken.encoding_for_model(model_name)
    # Encode the string and return the token count
    return len(encoder.encode(string)) if string else 0


def chunk_texts(
    df: pd.DataFrame,
    model_name: str,
    chunk_size: int = 256,
    chunk_overlap: int = 30
) -> pd.DataFrame:
    """
    Token-aware chunking of a DataFrame of texts.
    
    Args:
        df: DataFrame with columns ["page", "text", "filename"].
        model_name: Name of the model whose tokenizer to use (e.g. "gpt-4o", "claude-2.0").
        chunk_size: max tokens per chunk.
        chunk_overlap: tokens to overlap between chunks.
    
    Returns:
        DataFrame with columns ["chunk_id","chunk_text","page","filename","token_count"].
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    
    # 1) Get encoder for specified model

    # 2) Build a token-based recursive splitter
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 3) Iterate and build chunks
    all_chunks = []
    for _, row in df.iterrows():
        text = (row.get("text") or "").strip()
        if not text:
            continue
        splits = splitter.split_text(text)
        for fragment in splits:
            clean = fragment.strip()
            all_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "chunk_text": clean,
                "page": row["page"],
                "filename": row["filename"],
                "token_count": num_tokens_from_string(clean, model_name),
                "metadata": row.get("metadata", {}),
            })

    return pd.DataFrame(all_chunks)

if __name__ == "__main__":
    # 1) Load parsed PDF (page-level) DataFrame
    parsed = pd.read_csv("data/tempo_output/output_pdfplumber.csv", sep=",")

    # 2) Choose your model (e.g. "gpt-4o", "claude-2.0")
    model_name_tokenizer = "gpt-4o"         # for GPT-4 Nano
    # model_to_test = "claude-2.0"   # for Anthropic Claude V2

    # 3) Chunk and count tokens
    chunks_df = chunk_texts(parsed, model_name=model_name_tokenizer, chunk_size=256, chunk_overlap=30)

    # 4) Save
    out_path = Path("data/tempo_output/output_chunked.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_df.to_csv(out_path, index=False, sep=",", encoding="utf-8")
    print(f"Saved {len(chunks_df)} chunks using {model_name_tokenizer} tokenizer to {out_path}")




# Notes:

# def chunk_full_document(
#     df: pd.DataFrame,
#     model_name: str,
#     chunk_size: int = 256,
#     chunk_overlap: int = 30
# ) -> pd.DataFrame:
    
#     For each filename, concatenate its pages in order (inserting a page-break token),
#     split into ~chunk_size tokens, then record which pages each chunk covers.
    
#     Returns a DataFrame with:
#       - chunk_id
#       - chunk_text
#       - pages: list of page numbers this chunk spans
#       - filename
#       - token_count
    
#     encoder = get_encoder_for_model(model_name)
#     splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         model_name=model_name,
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "",],  # only paragraph or raw
#     )

#     all_chunks = []
#     for fname, sub in df.groupby("filename", sort=False):
#         # Build a single big text with page markers
#         pages = []
#         full_text = []
#         for _, row in sub.sort_values("page").iterrows():
#             pages.append(row["page"])
#             # insert an explicit marker so we can detect page boundaries later
#             full_text.append(f"[PAGE_{row['page']}] " + (row["text"] or ""))
#         joined = "\n\n".join(full_text)

#         # split on the full text
#         fragments = splitter.split_text(joined)
#         for frag in fragments:
#             clean = frag.strip()
#             if not clean:
#                 continue

#             # figure out which pages it covers by seeing which markers appear
#             covered = []
#             for m in re.finditer(r"\[PAGE_(\d+)\]", clean):
#                 covered.append(int(m.group(1)))
#             # if no explicit marker (i.e. fragment falls entirely inside a page),
#             # fall back to picking the single page whose marker occurs nearest the frag's start
#             if not covered:
#                 # get the index of this fragment in the original joined text
#                 start = joined.find(clean[:30])
#                 # find the last page marker before that
#                 prev = re.finditer(r"\[PAGE_(\d+)\]", joined[:start])
#                 last = None
#                 for mo in prev:
#                     last = int(mo.group(1))
#                 covered = [last] if last is not None else []

#             all_chunks.append({
#                 "chunk_id": str(uuid.uuid4()),
#                 "chunk_text": clean,
#                 "pages": covered,
#                 "filename": fname,
#                 "token_count": num_tokens_from_string(clean, encoder),
#             })

#     return pd.DataFrame(all_chunks)
