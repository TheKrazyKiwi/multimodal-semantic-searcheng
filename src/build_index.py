
import os
import torch
import argparse
import clip
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.model import MultimodalSearchEngine
from src.index import VectorIndex
from src.data.dataset import MultimodalDataset

def build_index(metadata_path, batch_size=32, device="cuda"):
    print(f"Building index from {metadata_path} on {device}...")
    
    # 1. Load Model (Zero-Shot)
    model = MultimodalSearchEngine(device=device)
    model.eval()

    # 2. Get CLIP Preprocess
    # We need the transform that matches the training one (CLIP's default)
    _, preprocess = clip.load("ViT-B/32", device="cpu") 
    
    # 3. Load Dataset
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
    dataset = MultimodalDataset(metadata_path, image_transform=preprocess)
    # Important: shuffle=False to align with metadata list index
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0) 
    
    # 4. Initialize Index
    index = VectorIndex()
    
    print(f"Dataset size: {len(dataset)}")
    
    # Track global index to retrieve metadata
    global_idx = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Indexing"):
            # Move inputs to device
            text = batch['text']
            image = batch['image'].to(device)
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            
            # Masks: dictionary of tensors (BatchSize, )
            mask = batch['mask'] 
            mask_text = mask['text'] # usually all 1s
            mask_image = mask['image']
            mask_audio = mask['audio']
            mask_video = mask['video']
            
            # Forward Pass (compute all embeddings)
            # Text
            text_emb = model.forward_text(text).cpu().numpy()
            
            # Image
            if mask_image.sum() > 0:
                img_emb = model.forward_image(image).cpu().numpy()
            else:
                img_emb = None # Optimization: skip if batch has no images
                
            # Audio
            if mask_audio.sum() > 0:
                aud_emb = model.forward_audio(audio).cpu().numpy()
            else:
                aud_emb = None
                
            # Video
            if mask_video.sum() > 0:
                 vid_emb = model.forward_video(video).cpu().numpy()
            else:
                 vid_emb = None

            # Add to Index
            current_batch_size = len(text)
            
            for i in range(current_batch_size):
                # Retrieve original metadata for this item
                item_meta = dataset.data[global_idx + i]
                # We want to store meaningful metadata for search results.
                # Common fields: id, text (caption/description), original paths
                
                base_meta = {
                    "id": item_meta.get("id", "unknown"),
                    "text": item_meta.get("text", ""),
                    "image_path": item_meta.get("image_path", ""),
                    "video_path": item_meta.get("video_path", ""),
                    "audio_path": item_meta.get("audio_path", "")
                }
                
                # 1. Text Vector (Always present/valid as anchor)
                # We assume text is always valid in this dataset design
                meta_text = base_meta.copy()
                meta_text['type'] = 'text'
                meta_text['content'] = item_meta.get("text", "") # Display content
                # Add normalized vector
                index.add(text_emb[i:i+1], [meta_text])
                
                # 2. Image Vector
                if img_emb is not None and mask_image[i] == 1:
                    meta_img = base_meta.copy()
                    meta_img['type'] = 'image'
                    meta_img['content'] = item_meta.get("image_path", "") # Display path
                    index.add(img_emb[i:i+1], [meta_img])
                    
                # 3. Audio Vector
                if aud_emb is not None and mask_audio[i] == 1:
                    meta_aud = base_meta.copy()
                    meta_aud['type'] = 'audio'
                    meta_aud['content'] = item_meta.get("audio_path", "") # Display path
                    index.add(aud_emb[i:i+1], [meta_aud])
                    
                # 4. Video Vector
                if vid_emb is not None and mask_video[i] == 1:
                    meta_vid = base_meta.copy()
                    meta_vid['type'] = 'video'
                    meta_vid['content'] = item_meta.get("video_path", "") # Display path
                    index.add(vid_emb[i:i+1], [meta_vid])
            
            global_idx += current_batch_size
            
    # Save the index
    index.save()
    print("Indexing Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="data/user_dataset_metadata.json", help="Path to metadata JSON")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    # Check if user metadata exists
    if args.metadata == "data/user_dataset_metadata.json" and not os.path.exists(args.metadata):
        print(f"User metadata not found at {args.metadata}, trying toy data...")
        args.metadata = "data/toy_dataset_metadata.json"

    build_index(args.metadata, batch_size=args.batch_size)
