from pathlib import Path
import base64
import pandas as pd
from tqdm import tqdm
import os
from mistralai import Mistral
from src.config import MISTRAL_API_KEY


def encode_pdf(pdf_path):
    """Encode the pdf to base64."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: The file {pdf_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def parse_document(path: Path) -> pd.DataFrame:
    """
    Parse a PDF document into a DataFrame with page, text, and filename columns.
    Uses Mistral OCR API to extract text from PDFs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    # Encode PDF to base64
    base64_pdf = encode_pdf(path)
    if base64_pdf is None:
        raise Exception(f"Failed to encode PDF: {path}")

    # Initialize Mistral client
    client = Mistral(api_key=MISTRAL_API_KEY)

    # Process document with OCR
    print(f"Processing {path.name} with Mistral OCR...")
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{base64_pdf}",
        },
    )

    # Extract text by page using markdown content
    all_texts = []
    if hasattr(ocr_response, "pages") and ocr_response.pages:
        for page in tqdm(ocr_response.pages, desc=f"Extracting text from {path.name}"):
            page_num = page.index + 1  # Convert 0-based index to 1-based page number

            # Extract markdown text from the page
            markdown_text = page.markdown if hasattr(page, "markdown") else ""

            # Clean text
            cleaned_text = markdown_text.strip()
            cleaned_text = " ".join(cleaned_text.split())  # collapse multiple spaces

            # Create metadata with page dimensions if available
            metadata = {}
            if hasattr(page, "dimensions"):
                metadata.update(
                    {
                        "dpi": page.dimensions.dpi
                        if hasattr(page.dimensions, "dpi")
                        else None,
                        "height": page.dimensions.height
                        if hasattr(page.dimensions, "height")
                        else None,
                        "width": page.dimensions.width
                        if hasattr(page.dimensions, "width")
                        else None,
                    }
                )

            # Convert metadata to string for consistency with original code
            metadata_str = str(metadata) if metadata else "metadata placeholder"

            all_texts.append(
                {
                    "page": page_num,
                    "text": cleaned_text,
                    "filename": path.name,
                    "metadata": metadata_str,
                }
            )
    else:
        # Handle case where OCR response doesn't have page structure
        print(f"Warning: OCR response format unexpected. Creating single entry.")
        all_texts.append(
            {
                "page": 1,
                "text": "",
                "filename": path.name,
                "metadata": "metadata placeholder",
            }
        )

    # Create DataFrame
    df = pd.DataFrame(all_texts)
    df = df[df["text"] != ""]  # Remove empty pages
    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    output_path = Path("data/tempo_output/output_mistral_ocr.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document_path = Path("data/document_1_small.pdf")

    parsed_text_df = parse_document(document_path)

    # Print usage info if available
    print(f"Document stats: {parsed_text_df.shape[0]} pages extracted")

    # Save csv file with UTF-8 encoding
    parsed_text_df.to_csv(output_path, index=False, sep=",", encoding="utf-8")
    print(f"Parsed text saved to {output_path}")

    # Display sample of the extracted text
    if not parsed_text_df.empty:
        print("\nSample of extracted text (first page):")
        sample_text = parsed_text_df.iloc[0]["text"]
        print(sample_text[:500] + "..." if len(sample_text) > 500 else sample_text)
