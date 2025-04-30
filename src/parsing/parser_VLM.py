from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import time
import pandas as pd

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 448
MAX_NUM = 12

# Helper functions for image preprocessing (from official documentation)
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def parse_document(path: Path, model_path="OpenGVLab/InternVL2-1B") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=True,
    ).eval()#.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    # PDF to images
    print(f"Converting PDF to images...")
    images = convert_from_path(str(path), dpi=300)
    
    # Process each page separately and combine results
    all_texts = []
    generation_config = dict(max_new_tokens=4096, do_sample=False)
    
    for i, img in enumerate(images):
        print(f"Processing page {i+1}/{len(images)}")
        # time it 
        
        # Process with InternVL2 approach
        try:
            start_time = time.time()
            # Convert image to pixel values using their preprocessing method
            pixel_values = load_image(img, input_size=INPUT_SIZE, max_num=MAX_NUM)
            pixel_values = pixel_values.to(device)
            
            if device == "cuda":
                pixel_values = pixel_values.to(torch.bfloat16)
                
            # Create prompt for OCR
            question = "<image>\nExtract all text content visible in this image."
            
            # Use the chat function as shown in their example
            response = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config
            )
            
            all_texts.append(response)

            # Print the time taken for this page
            end_time = time.time()
            print(f"Time taken for page {i+1}: {end_time - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error processing page {i+1}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_texts.append(f"[ERROR PROCESSING PAGE {i+1}]")

    # return the parsed text as a dataframe with 2 columns: page and text
    df = pd.DataFrame(all_texts, columns=["text"])
    df["page"] = df.index + 1
    df = df[["page", "text"]]
    df["text"] = df["text"].str.replace("\n", " ", regex=False)  # Replace newlines with spaces
    df["text"] = df["text"].str.replace(" +", " ", regex=True)  # Replace multiple spaces with a single space
    df["text"] = df["text"].str.strip()  # Strip leading/trailing spaces
    df = df[df["text"] != ""]  # Remove empty rows
    df = df.reset_index(drop=True)  # Reset index after filtering
    df["filename"] = path.name  # Add filename column
    return df

if __name__ == "__main__":
    output_path = Path("data/tempo_output/output.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    # Ensure the directory exists
   
    document_path = Path("data/document_1_small.pdf")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(f"Using device: {device}")
    parsed_text = parse_document(document_path)
    # Save the parsed dataframe to a CSV file
    parsed_text.to_csv(output_path, index=False, sep="\t", encoding="utf-8")
    print(f"Parsed text saved to {output_path}")