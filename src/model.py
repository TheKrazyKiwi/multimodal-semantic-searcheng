import torch
import torch.nn as nn
import torch.nn.functional as F
from .components.encoders import TextEncoder, ImageEncoder, AudioEncoder, VideoEncoder

class MultimodalSearchEngine(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        
        # --- ENCODERS (Frozen, Zero-Shot) ---
        print("Loading Image/Video Encoder (CLIP)...")
        # We share the underlying CLIP model for Text, Image and Video to save VRAM and maintain perfect alignment.
        self.image_encoder = ImageEncoder(device=device)
        clip_model = self.image_encoder.model
        
        print("Loading Text Encoder (CLIP)...")
        self.text_encoder = TextEncoder(clip_model=clip_model, device=device)
        self.video_encoder = VideoEncoder(clip_model=clip_model, device=device)
        
        print("Loading Audio Encoder (Whisper)...")
        self.audio_encoder = AudioEncoder(device=device)

        # Note: Zero-Shot architecture skips custom ProjectionHead translation layers completely.
        self.to(device)

    def forward_text(self, texts):
        features = self.text_encoder(texts)
        return F.normalize(features, p=2, dim=1)

    def forward_image(self, images):
        features = self.image_encoder(images)
        return F.normalize(features, p=2, dim=1)

    def forward_video(self, video_frames):
        features = self.video_encoder(video_frames)
        return F.normalize(features, p=2, dim=1)

    def forward_audio(self, audio_mels):
        features = self.audio_encoder(audio_mels)
        return F.normalize(features, p=2, dim=1)
