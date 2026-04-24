
from src.data.dataset import MultimodalDataset
import clip
import torch
import json
import os

# Dummy metadata
meta_path = "temp_video_test.json"
video_path = "/home/kiwi/multi-semantic-searcheng/data/video/videos/video0.mp4"

if not os.path.exists(video_path):
    print(f"ERROR: Video path does not exist: {video_path}")
    exit(1)

with open(meta_path, 'w') as f:
    json.dump([{
        "id": "video_test",
        "text": "test video",
        "image_path": "",
        "audio_path": "",
        "video_path": video_path,
        "original_filename": "video0.mp4"
    }], f)

print("Loading CLIP...")
_, preprocess = clip.load("ViT-B/32", device="cpu")

print("Initializing Dataset...")
ds = MultimodalDataset(meta_path, image_transform=preprocess)

print("Loading Item 0...")
item = ds[0]

print(f"Video Mask: {item['mask']['video']}")
print(f"Video Shape: {item['video'].shape}")
print(f"Is Video Zero? {torch.all(item['video'] == 0)}")

# Clean up
if os.path.exists(meta_path):
    os.remove(meta_path)
