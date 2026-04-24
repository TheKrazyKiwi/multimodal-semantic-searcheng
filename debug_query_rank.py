
import sys
import torch
import clip
import faiss
import pickle
import numpy as np

# Load Index
print("Loading index...", flush=True)
index = faiss.read_index("index.faiss")
with open("index_meta.pkl", "rb") as f:
    meta = pickle.load(f)

# Load Model
print("Loading CLIP...", flush=True)
device = "cpu"
model, _ = clip.load("ViT-B/32", device=device)

# Query that SHOULD match video0 ("white man driving black car on road .")
query = "white man driving black car on road"
print(f"Query: '{query}'", flush=True)

with torch.no_grad():
    text_inputs = clip.tokenize([query]).to(device)
    q_vec = model.encode_text(text_inputs).float().numpy()
    faiss.normalize_L2(q_vec)

# Search DEEP
k_search = 10000
print(f"Searching top {k_search}...", flush=True)
D, I = index.search(q_vec, k_search)

found_video = False
for rank, idx in enumerate(I[0]):
    if idx == -1: break
    item = meta[idx]
    if item['type'] == 'video':
        print(f"FOUND VIDEO at Rank {rank}: {item['content']}")
        print(f"Score: {D[0][rank]}")
        found_video = True
        break

if not found_video:
    print("Video NOT found in top 10000.")
