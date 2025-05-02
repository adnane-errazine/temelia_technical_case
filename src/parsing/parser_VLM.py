from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import torch
import torchvision.transforms as T
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import time
import pandas as pd
from tqdm import tqdm
import gc  # For garbage collection
import os
import psutil
import GPUtil

# Set environment variables for better performance
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Helpful for debugging

# Constants - set to be compatible with InternVL3-1B
IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)  # CLIP standard mean
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)  # CLIP standard std
INPUT_SIZE = 448  # InternVL3 requires 448x448 images
MAX_IMAGES = 1  # Process one image per page for simplicity


# Helper function to monitor system resources
def log_resources():
    # Log CPU usage
    cpu_percent = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()

    # Log GPU usage if available
    gpu_info = ""
    if torch.cuda.is_available():
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_info = f", GPU: {gpu.load * 100:.1f}%, GPU Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB"
        except:
            gpu_info = ", GPU metrics unavailable"

    print(f"CPU: {cpu_percent}%, RAM: {memory_info.percent}%{gpu_info}")


# Simplified preprocessing function that follows InternVL3's expected format
def preprocess_image(image, input_size=448):
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize the image to the required input size
    image = image.resize((input_size, input_size))

    # Convert to tensor and normalize with CLIP standard values
    transform = T.Compose(
        [T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    )

    # Apply transformation
    pixel_values = transform(image)

    # Add batch dimension
    pixel_values = pixel_values.unsqueeze(0)

    return pixel_values


def parse_document(
    path: Path, model_path="OpenGVLab/InternVL3-1B", dpi=200
) -> pd.DataFrame:
    """Parse a PDF document using InternVL3-1B model with a fixed preprocessing approach"""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Log initial system resources
    print("Initial system resources:")
    log_resources()

    # Configure quantization - for InternVL3
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Use 8-bit for more stability
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    # Load model and tokenizer for InternVL3
    print(f"Loading model from {model_path}...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=device == "cuda",
        device_map="auto",
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )

    # Fix pad_token_id warning
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Log resources after model loading
    print("System resources after model loading:")
    log_resources()

    # PDF to images with sufficient DPI for text recognition
    print(f"Converting PDF to images with DPI={dpi}...")
    images = convert_from_path(str(path), dpi=dpi)

    # Process each page separately and combine results
    all_texts = []
    generation_config = dict(
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.1,
    )

    pbar = tqdm(enumerate(images), total=len(images), desc="Processing pages")
    for i, img in pbar:
        pbar.set_description(f"Processing page {i + 1}/{len(images)}")

        try:
            start_time = time.time()

            # Free up memory before processing each page
            torch.cuda.empty_cache() if device == "cuda" else None
            gc.collect()

            # Simple preprocessing - one image at a time with proper size
            pixel_values = preprocess_image(img, input_size=INPUT_SIZE)
            pixel_values = pixel_values.to(device)

            if device == "cuda":
                pixel_values = pixel_values.to(torch.float16)

            # Monitor resources before model inference
            if i == 0:  # Only log for first page to avoid clutter
                print(f"\nResources before processing page {i + 1}:")
                log_resources()
                print(f"Pixel values shape: {pixel_values.shape}")

            # Create prompt for OCR
            question = "<image>\nExtract all text content visible in this image."

            # Process with the model
            response, _ = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                history=None,
                return_history=True,
            )

            # Add the processed text
            all_texts.append(response)

            # Print time and resources used
            end_time = time.time()
            elapsed = end_time - start_time
            pbar.set_postfix({"Time": f"{elapsed:.2f}s"})

            if i == 0:  # Only log for first page to avoid clutter
                print(f"\nResources after processing page {i + 1}:")
                log_resources()
                print(f"Page processing time: {elapsed:.2f}s")

        except Exception as e:
            print(f"Error processing page {i + 1}: {str(e)}")
            import traceback

            traceback.print_exc()
            all_texts.append(f"[ERROR PROCESSING PAGE {i + 1}]")

    # Clean up memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Return the parsed text as a dataframe with 2 columns: page and text
    df = pd.DataFrame(all_texts, columns=["text"])
    df["page"] = df.index + 1
    df = df[["page", "text"]]
    df["text"] = df["text"].str.replace(
        "\n", " ", regex=False
    )  # Replace newlines with spaces
    df["text"] = df["text"].str.replace(
        " +", " ", regex=True
    )  # Replace multiple spaces with a single space
    df["text"] = df["text"].str.strip()  # Strip leading/trailing spaces
    df = df[df["text"] != ""]  # Remove empty rows
    df = df.reset_index(drop=True)  # Reset index after filtering
    df["filename"] = path.name  # Add filename column
    return df


if __name__ == "__main__":
    output_path = Path("data/tempo_output/output.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document_path = Path("data/document_1_small.pdf")

    # Parse with optimized settings
    parsed_text = parse_document(
        document_path,
        model_path="OpenGVLab/InternVL3-1B",
        dpi=200,  # Higher DPI for better text recognition
    )

    # Save the parsed dataframe to a CSV file
    parsed_text.to_csv(output_path, index=False, sep="\t", encoding="utf-8")
    print(f"Parsed text saved to {output_path}")
