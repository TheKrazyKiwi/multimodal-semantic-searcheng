
import faiss
import pickle
import numpy as np

# Load Index & Metadata
try:
    print("Loading index...", flush=True)
    full_index = faiss.read_index("index.faiss")
    
    print("Loading metadata...", flush=True)
    with open("index_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    print(f"Total entries: {len(meta)}", flush=True)
    d = full_index.d  # Dimension of vectors (e.g., 512 for CLIP)
    
    # Identify indices for each modality
    video_indices = [i for i, m in enumerate(meta) if m['type'] == 'video']
    audio_indices = [i for i, m in enumerate(meta) if m['type'] == 'audio']
    image_indices = [i for i, m in enumerate(meta) if m['type'] == 'image']

    print(f"Found {len(video_indices)} Videos, {len(audio_indices)} Audio, {len(image_indices)} Images.", flush=True)

    # Function to create sub-index
    def create_sub_index(indices, name):
        if not indices:
            print(f"Skipping {name} index creation (no items).", flush=True)
            return None, []
        
        print(f"Building {name} index with {len(indices)} items...", flush=True)
        sub_index = faiss.IndexFlatIP(d)
        
        # Reconstruct vectors from full index
        vectors = np.zeros((len(indices), d), dtype='float32')
        for i, idx in enumerate(indices):
            # Try reconstruct (works for IndexFlatIP/L2)
            try:
                vec = full_index.reconstruct(idx)
                vectors[i] = vec
            except Exception as e:
                print(f"Error reconstructing vector {idx}: {e}", flush=True)
                return None, []
        
        sub_index.add(vectors)
        return sub_index, [meta[i] for i in indices]

    # Test Creation
    video_idx, video_meta = create_sub_index(video_indices, "Video")
    audio_idx, audio_meta = create_sub_index(audio_indices, "Audio")
    
    if video_idx:
        print(f"Video Index created with {video_idx.ntotal} vectors.", flush=True)
        # Dummy query to test search
        q = np.random.rand(1, d).astype('float32')
        faiss.normalize_L2(q)
        D, I = video_idx.search(q, k=min(5, len(video_meta)))
        print(f"Video Search Result IDs: {I}", flush=True)

except Exception as e:
    print(f"Critical Error: {e}", flush=True)
