
import json
import time
import os

start = time.time()
meta_file = "data/user_dataset_metadata.json"

if os.path.exists(meta_file):
    print(f"Loading {meta_file}...")
    with open(meta_file, 'r') as f:
        data = json.load(f)
    print(f"JSON load took {time.time() - start:.2f}s")
    
    start_process = time.time()
    path_to_text = {}
    count = 0
    for item in data:
        text = item.get('text', '')
        if not text: continue
        
        for key in ['image_path', 'video_path', 'audio_path']:
            path = item.get(key, '')
            if path and path not in path_to_text:
                path_to_text[path] = text
        count += 1
        
    print(f"Processing {count} items took {time.time() - start_process:.2f}s")
    print(f"Total paths mapped: {len(path_to_text)}")
else:
    print("Metadata file not found.")
