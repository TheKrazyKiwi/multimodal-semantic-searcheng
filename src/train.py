import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import clip # Import CLIP

from src.model import MultimodalSearchEngine
from src.data.dataset import MultimodalDataset

# --- Hyperparameters ---
BATCH_SIZE = 4       # Very small for 4GB VRAM
ACCUM_STEPS = 4      # Effective batch size = 16
LEARNING_RATE = 1e-4
EPOCHS = 5
TEMP = 0.07          # Contrastive temperature

def contrastive_loss(v1, v2, temperature=0.07):
    """
    Computes InfoNCE (Contrastive) Loss between two batches of vectors.
    v1: (B, D)
    v2: (B, D)
    """
    # Normalize vectors
    v1 = nn.functional.normalize(v1, dim=1)
    v2 = nn.functional.normalize(v2, dim=1)
    
    # Cosine similarity matrix: (B, B)
    logits = torch.matmul(v1, v2.T) / temperature
    
    # Labels: 0, 1, 2, ... B-1 (diagonal elements are positives)
    labels = torch.arange(v1.size(0), device=v1.device)
    
    loss_v1 = nn.functional.cross_entropy(logits, labels)
    loss_v2 = nn.functional.cross_entropy(logits.T, labels)
    
    return (loss_v1 + loss_v2) / 2

def train(epochs=EPOCHS, batch_size=BATCH_SIZE, metadata_file="data/toy_dataset_metadata.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}...")
    print(f"Using metadata file: {metadata_file}")
    
    # Get CLIP transform
    # We load just the model structure to get the transform, then discard valid model if we want, 
    # but clip.load returns both. We put it on CPU to save GPU RAM for the main training loop.
    _, preprocess = clip.load("ViT-B/32", device="cpu")

    # Data
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
    dataset = MultimodalDataset(metadata_file, image_transform=preprocess)
    # Quick sanity check: Drop last batch if it's 1 sample to avoid BN errors (though we use LN usually)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    if len(dataloader) == 0:
        print("Dataset too small for batch size! Using batch_size=2 and drop_last=False")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=False)

    # Model
    model = MultimodalSearchEngine(device=device)
    model.train()
    
    # Optimizer (Only optimize projection heads)
    params = list(model.text_proj.parameters()) + \
             list(model.image_proj.parameters()) + \
             list(model.video_proj.parameters()) + \
             list(model.audio_proj.parameters())
             
    optimizer = optim.AdamW(params, lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler() # Mixed Precision

    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(loop):
            # Move data to device
            # Retrieve masks
            mask = batch['mask'] # dict of tensors (B,)
            
            # Move to device
            text = batch['text']
            image = batch['image'].to(device)
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            
            # Masks to device (float for easier calculation or bool for indexing)
            mask_image = mask['image'].to(device)
            mask_audio = mask['audio'].to(device)
            mask_video = mask['video'].to(device)

            with torch.cuda.amp.autocast():
                # Forward Pass
                # 1. Text Embeddings (Anchor)
                text_emb = model.forward_text(text)
                
                # 2. Other Modalities
                img_emb = model.forward_image(image)
                aud_emb = model.forward_audio(audio)
                vid_emb = model.forward_video(video)
                
                # Calculate Losses
                # Align everything to Text (since description covers all)
                # Calculate Losses with Masking
                # We only compute loss for samples where the modality exists.
                # However, contrastive loss expects a batch. 
                # Strategy: We can't easily do batch-wise contrastive loss if the batch size varies per modality 
                # (i.e. if we filter). 
                # Simpler approach: Compute loss for ALL, but multiply the individual sample losses by the mask.
                # But `contrastive_loss` returns a scalar (mean over batch). We need to modify it or 
                # handle it here.
                
                # Let's start by getting embeddings for valid data only? 
                # No, to keep batch alignment (text[i] vs image[i]), we must keep the batch structure.
                # If image[i] is empty (zeros), its embedding will be garbage, but we should zero out its contribution to loss.
                
                loss_total = 0
                count_modalities = 0

                # Helper to compute masked loss
                def masked_contrastive(emb1, emb2, mask_valid):
                    """
                    emb1: Text (always valid)
                    emb2: Other Modality
                    mask_valid: (B,) 1 if valid, 0 if not
                    """
                    if mask_valid.sum() == 0:
                        return torch.tensor(0.0, device=device, requires_grad=True)
                        
                    # 1. Compute full contrastive matrix (B, B)
                    # 2. But we only care about the rows/cols where mask is 1.
                    # Actually, if we just slice the valid indices, we get a smaller batch (B', B')
                    # This is valid for InfoNCE: we just compare valid pairs against other valid pairs.
                    
                    valid_idx = torch.nonzero(mask_valid).squeeze(1)
                    if len(valid_idx) == 0:
                         return torch.tensor(0.0, device=device, requires_grad=True)
                         
                    emb1_sub = emb1[valid_idx]
                    emb2_sub = emb2[valid_idx]
                    
                    return contrastive_loss(emb1_sub, emb2_sub, TEMP)

                loss_ti = masked_contrastive(text_emb, img_emb, mask_image)
                loss_ta = masked_contrastive(text_emb, aud_emb, mask_audio)
                loss_tv = masked_contrastive(text_emb, vid_emb, mask_video)
                
                # Weighted average based on valid losses
                # We add them up. If a loss is 0 (no data), it doesn't contribute.
                # We should average by the number of *active* tasks, but here we can just sum or average 
                # non-zero ones if we want to balance gradients.
                # Simple sum is often fine, or average by 3 if we want to keep scale.
                # Better: Average by number of valid modality types present in the *dataset*?
                # Let's just sum them and let the optimizer handle it, or average non-zero terms.
                
                valid_losses = []
                if mask_image.sum() > 0: valid_losses.append(loss_ti)
                if mask_audio.sum() > 0: valid_losses.append(loss_ta)
                if mask_video.sum() > 0: valid_losses.append(loss_tv)
                
                if len(valid_losses) > 0:
                    loss = sum(valid_losses) / len(valid_losses)
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

                loss = loss / ACCUM_STEPS

            # Backward
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            total_loss += loss.item() * ACCUM_STEPS
            
            loop.set_postfix(loss=total_loss / (i+1))

        # Save checkpoint
        torch.save(model.state_dict(), "checkpoints/last_model.pt")
        print(f"Epoch {epoch+1} Complete. Loss: {total_loss / len(dataloader):.4f}")

    print("Training Complete. Model saved to checkpoints/last_model.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--metadata", type=str, default="data/user_dataset_metadata.json", help="Path to metadata JSON")
    args = parser.parse_args()
    
    # Check if user metadata exists, else fallback to toy
    if args.metadata == "data/user_dataset_metadata.json" and not os.path.exists(args.metadata):
        print(f"User metadata not found at {args.metadata}, falling back to toy dataset.")
        args.metadata = "data/toy_dataset_metadata.json"

    train(epochs=args.epochs, batch_size=args.batch_size, metadata_file=args.metadata)
