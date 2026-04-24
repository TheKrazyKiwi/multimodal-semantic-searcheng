
import json
import os
import time

def main():
    start = time.time()
    meta_file = "data/user_dataset_metadata.json"
    output_file = "description_map.json"
    
    if not os.path.exists(meta_file):
        print(f"Metadata file {meta_file} not found.")
        return

    print(f"Loading {meta_file}...")
    try:
        with open(meta_file, 'r') as f:
            data = json.load(f)
        print(f"JSON load took {time.time() - start:.2f}s")
        
        path_to_text = {}
        count = 0
        for item in data:
            text = item.get('text', '')
            if not text: continue
            
            for key in ['image_path', 'video_path', 'audio_path']:
                path = item.get(key, '')
                if path:
                     if path not in path_to_text:
                        path_to_text[path] = text
            count += 1
            
        print(f"Processed {count} items into {len(path_to_text)} mappings.")
        
        with open(output_file, "w") as f:
            json.dump(path_to_text, f)
            
        print(f"Saved description map to {output_file}. Total time: {time.time() - start:.2f}s")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
