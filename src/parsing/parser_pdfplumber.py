from pathlib import Path
import pdfplumber
import pandas as pd
from tqdm import tqdm

def parse_document(path: Path) -> pd.DataFrame:
    """
    Parse a PDF document into a DataFrame with page, text, and filename columns.
    Designed to handle French text (accents, special characters).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    all_texts = []

    with pdfplumber.open(path) as pdf:
        for page_num, page in tqdm(enumerate(pdf.pages, start=1), total=len(pdf.pages), desc=f"Parsing {path.name}"):
            text = page.extract_text() or ""  # Handle empty pages
            # Clean text
            cleaned_text = text.strip()
            cleaned_text = ' '.join(cleaned_text.split())  # collapse multiple spaces
            all_texts.append({
                "page": page_num,
                "text": cleaned_text,
                "filename": path.name
            })

    df = pd.DataFrame(all_texts)
    df = df[df["text"] != ""]  # Remove empty pages
    df = df.reset_index(drop=True)
    df["metadata"]= "metadata placeholder"  # Placeholder for metadata, can be customized

    return df

if __name__ == "__main__":
    output_path = Path("data/tempo_output/output_pdfplumber.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document_path = Path("data/document_1.pdf")
    parsed_text_df = parse_document(document_path)

    # Save csv file with UTF-8 encoding
    parsed_text_df.to_csv(output_path, index=False, sep=",", encoding="utf-8")
    print(f"Parsed text saved to {output_path}")
