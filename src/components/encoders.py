import torch
import torch.nn as nn
import clip
from transformers import AutoModel, AutoTokenizer
import whisper
from typing import List
import torch.nn.functional as F

class TextEncoder(nn.Module):
    """
    Zero-Shot Alignment:
    Instead of using DistilBERT, we reuse the exact same CLIP model used for images/video.
    This guarantees mathematically identical latent spaces without needing projection heads.
    """
    def __init__(self, clip_model, device="cpu"):
        super().__init__()
        self.device = device
        self.clip_model = clip_model

    def forward(self, texts: List[str]):
        # Tokens restricted to max 77 by CLIP design
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
        return text_features.float()

class ImageEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        # Load CLIP model
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # FREEZE BACKBONE
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, images):
        # images expected to be preprocessed tensor batch
        # shape: (B, 3, 224, 224)
        with torch.no_grad():
            image_features = self.model.encode_image(images)
        return image_features.float()

class AudioEncoder(nn.Module):
    def __init__(self, model_size="tiny", device="cpu"):
        super().__init__()
        self.device = device
        self.model = whisper.load_model(model_size, device=self.device)
        
        # FREEZE BACKBONE
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, audio_mels):
        with torch.no_grad():
            audio_features = self.model.encoder(audio_mels)
        
        # Whisper encoder output: (B, n_ctx, n_state)
        # We'll use global average pooling for a single vector
        pooled = audio_features.mean(dim=1)
        
        # VRAM preservation: model_size="tiny" outputs 384 dims. 
        # We pad to 512 dims to math VectorIndex natively without crashing FAISS.
        if pooled.shape[-1] < 512:
            pad_size = 512 - pooled.shape[-1]
            pooled = F.pad(pooled, (0, pad_size))
            
        return pooled

class VideoEncoder(nn.Module):
    """
    Memory Optimization:
    Instead of loading a massive ViViT model (~1GB+), we reuse the CLIP Image Encoder!
    Strategy: Extract N frames -> CLIP Encode each -> Average Pooling.
    This saves massive VRAM since we already have CLIP loaded.
    """
    def __init__(self, clip_model, device="cpu"):
        super().__init__()
        self.device = device
        self.clip_model = clip_model # Share the instance!

    def forward(self, video_frames):
        # video_frames: (B, Num_Frames, 3, 224, 224)
        # Flatten to (B * Num_Frames, 3, 224, 224)
        B, F, C, H, W = video_frames.shape
        flat_frames = video_frames.view(-1, C, H, W)
        
        with torch.no_grad():
            frame_features = self.clip_model.encode_image(flat_frames)
        
        # Reshape back to (B, F, Embed_Dim) and Average
        frame_features = frame_features.view(B, F, -1)
        video_embedding = frame_features.mean(dim=1)
        
        return video_embedding.float()
