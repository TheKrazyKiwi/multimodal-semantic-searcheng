
import faiss
import pickle
import numpy as np
import os
import sys

def main():
    print("Loading full index...", flush=True)
    try:
        index = faiss.read_index("index.faiss")
        with open("index_meta.pkl", "rb") as f:
            meta = pickle.load(f)
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    d = index.d
    ntotal = index.ntotal
    print(f"Full Index loaded: {ntotal} vectors, dim={d}", flush=True)

    # Prepare containers
    indices = {
        'image': [],
        'audio': [],
        'video': [],
        'text': []
    }
    
    # Classify
    for i, m in enumerate(meta):
        t = m.get('type', 'unknown')
        if t in indices:
            indices[t].append(i)
    
    # Create sub-indices
    for key in ['image', 'audio', 'video']:
        idxs = indices[key]
        if not idxs:
            print(f"Skipping {key}: No items found.", flush=True)
            continue
            
        print(f"Creating {key} index with {len(idxs)} items...", flush=True)
        sub_index = faiss.IndexFlatIP(d)
        sub_meta = []
        
        # Reconstruct vectors
        vectors = np.zeros((len(idxs), d), dtype='float32')
        for i, original_idx in enumerate(idxs):
            try:
                vec = index.reconstruct(original_idx)
                vectors[i] = vec
                sub_meta.append(meta[original_idx])
            except RuntimeError as e:
                print(f"Error reconstructing {original_idx}: {e}", flush=True)
                # Fallback: maybe just skip or use zeros? 
                # If reconstruct fails, index is corrupt or unsupported.
                # IndexFlatIP supports reconstruct.
                return

        sub_index.add(vectors)
        
        # Save
        index_path = f"index_{key}.faiss"
        meta_path = f"index_{key}_meta.pkl"
        
        faiss.write_index(sub_index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(sub_meta, f)
            
        print(f"Saved {index_path} and {meta_path}", flush=True)

if __name__ == "__main__":
    main()
