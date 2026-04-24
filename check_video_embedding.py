
import faiss
import pickle
import numpy as np

try:
    print("Loading index...", flush=True)
    full_index = faiss.read_index("index.faiss")
    
    print("Loading metadata...", flush=True)
    with open("index_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    print(f"Total entries: {len(meta)}", flush=True)
    d = full_index.d
    
    # Find first video
    v_idx = -1
    for i, m in enumerate(meta):
        if m['type'] == 'video':
            v_idx = i
            print(f"Found video at index {i}: {m}")
            break
            
    if v_idx != -1:
        # Reconstruct vector
        vec = full_index.reconstruct(v_idx)
        print(f"Video Vector Norm: {np.linalg.norm(vec)}")
        print(f"Video Vector Max: {np.max(vec)}")
        print(f"Video Vector Min: {np.min(vec)}")
        print(f"Is Zero? {np.all(vec == 0)}")
    else:
        print("No video found.")

except Exception as e:
    print(f"Error: {e}")
