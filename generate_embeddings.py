import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import argparse

def load_image(path):
    """Load an image, return None if failed"""
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"⚠️ Failed to load {path}: {e}")
        return None

def find_image_path(sticker_id, image_folder):
    """Find image path for a sticker ID in the folder"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']

    possible_files = [
        f"{sticker_id}.jpg",
    ]

    for filename in possible_files:
        full_path = os.path.join(image_folder, filename)
        if os.path.exists(full_path):
            return full_path
    
    return None

def get_sticker_ids_from_csv(csv_path):
    """Extract unique sticker IDs from CSV file"""
    print(f"📊 Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    sticker_ids_1 = df['sticker_id'].unique().tolist()
    sticker_ids_2 = df['sticker_id_2'].unique().tolist()
    docids = df['docid']
    
    all_sticker_ids = list(set(sticker_ids_1 + sticker_ids_2))
    print(f"📈 Found {len(all_sticker_ids)} unique sticker IDs in CSV")
    
    return all_sticker_ids

def main():
    parser = argparse.ArgumentParser(description='Generate CLIP embeddings for sticker IDs from CSV')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file with sticker IDs')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to image folder')
    parser.add_argument('--output', type=str, required=True, help='Output path for embeddings')
    parser.add_argument('--model_path', type=str, default="openai-clip-vit-large-patch14", 
                       help='Path to CLIP model')
    parser.add_argument('--checkpoint_path', type=str, default='', 
                       help='Path to fine-tuned checkpoint')
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    print("🔄 Loading CLIP model...")
    model = CLIPModel.from_pretrained(args.model_path)
    processor = CLIPProcessor.from_pretrained(args.model_path)
    model = model.to(device)

    if os.path.exists(args.checkpoint_path):
        print(f"🔄 Loading fine-tuned checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("⚠️ No checkpoint found, using pre-trained weights")

    model.eval()
    
    sticker_ids = get_sticker_ids_from_csv(args.csv)
    print(f"🔍 Searching for images in: {args.image_folder}")
    image_paths = []
    found_sticker_ids = []
    
    for sticker_id in tqdm(sticker_ids, desc="Finding images"):
        image_path = find_image_path(sticker_id, args.image_folder)
        if image_path:
            image_paths.append(image_path)
            found_sticker_ids.append(sticker_id)
        else:
            print(f"⚠️ Could not find image for sticker ID: {sticker_id}")
    
    print(f"📊 Found {len(image_paths)} images out of {len(sticker_ids)} sticker IDs")
    
    if len(image_paths) == 0:
        print("❌ No images found, exiting...")
        return
    
    embeddings = {}
    skipped_count = 0
    
    print("🎯 Generating embeddings...")
    for img_path, sticker_id in tqdm(zip(image_paths, found_sticker_ids), total=len(image_paths), desc="Processing images"):
        image = load_image(img_path)
        if image is None:
            skipped_count += 1
            continue
        
        try:
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            
            embeddings[str(sticker_id)] = {
                'embedding': emb.cpu(),
                'image_path': img_path
            }
            
        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")
            skipped_count += 1

    print(f"💾 Saving {len(embeddings)} embeddings to {args.output}")
    print(f"⏩ Skipped {skipped_count} images due to errors")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    torch.save({
        'embeddings': embeddings,
        'model_path': args.model_path,
        'checkpoint_path': args.checkpoint_path if os.path.exists(args.checkpoint_path) else None,
        'source_csv': args.csv,
        'image_folder': args.image_folder,
        'total_sticker_ids': len(sticker_ids),
        'found_images': len(embeddings),
        'skipped_images': skipped_count
    }, args.output)
    
    print("\n" + "="*50)
    print("📋 GENERATION SUMMARY")
    print("="*50)
    print(f"Total sticker IDs in CSV: {len(sticker_ids)}")
    print(f"Images found: {len(embeddings)}")
    print(f"Images skipped: {skipped_count}")
    print(f"Success rate: {len(embeddings)/len(sticker_ids)*100:.1f}%")
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
