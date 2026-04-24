
import pickle
import sys

print("Loading index metadata...", flush=True)
try:
    with open("index_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    print(f"Total entries: {len(meta)}", flush=True)
    
    video_entries = [m for m in meta if m.get('type') == 'video']
    print(f"Total video entries: {len(video_entries)}", flush=True)

    if video_entries:
        print(f"First video entry: {video_entries[0]}", flush=True)
    else:
        print("No video entries found.", flush=True)

except Exception as e:
    print(f"Error: {e}", flush=True)
