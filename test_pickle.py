
import pickle
import time
import os

print("Starting pickle load test...")
start = time.time()
try:
    with open("description_map.pkl", "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} items in {time.time() - start:.4f}s")
    print(f"Sample item: {list(data.items())[0]}")
except Exception as e:
    print(f"Failed to load pickle: {e}")
