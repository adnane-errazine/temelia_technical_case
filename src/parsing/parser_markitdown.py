from pathlib import Path
from typing import Dict
import pandas as pd
from markitdown import MarkItDown

# initialize the converter once
_md = MarkItDown()

def parse_document(path: Path) -> pd.DataFrame:
    """
    Parse a single document (PDF, DOCX, HTML, images, etc.) into a DataFrame with page, text, and filename.
    Raises FileNotFoundError if the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    result = _md.convert(str(path))

    # MarkItDown usually returns full text as one block; we'll treat it as page 1
    texts = [result.text_content]
    print(len(texts))

    df = pd.DataFrame(texts, columns=["text"])
    df["page"] = df.index + 1  # page number (always 1 unless you split)
    df = df[["page", "text"]]  # rearrange columns
    df["text"] = df["text"].str.replace("\n", " ", regex=False)  # Replace newlines with spaces
    df["text"] = df["text"].str.replace(" +", " ", regex=True)  # Replace multiple spaces with a single space
    df["text"] = df["text"].str.strip()  # Remove leading/trailing spaces
    df = df[df["text"] != ""]  # Remove empty rows
    df = df.reset_index(drop=True)
    df["filename"] = path.name  # Add filename column

    return df

if __name__ == "__main__":
    # Example usage
    output_path = Path("data/tempo_output/output_markitdown.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    document_path = Path("data/document_1_small.pdf")
    parsed_text_df = parse_document(document_path)

    # Save as tab-separated CSV
    parsed_text_df.to_csv(output_path, index=False, sep="\t", encoding="utf-8")
    print(f"Parsed text saved to {output_path}")
