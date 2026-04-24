
import pickle
import collections

try:
    with open("index_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    print(f"Total entries: {len(meta)}")
    video_count = sum(1 for m in meta if m.get('type') == 'video')
    print(f"Video entries: {video_count}")
    
    if video_count > 0:
        print("First video entry:")
        for m in meta:
            if m.get('type') == 'video':
                print(m)
                break
    else:
        print("NO VIDEO ENTRIES FOUND.")

except Exception as e:
    print(f"Error reading index: {e}")
