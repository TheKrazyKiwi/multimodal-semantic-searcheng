
import pickle
import collections

with open("index_meta.pkl", "rb") as f:
    meta = pickle.load(f)

print(f"Total entries: {len(meta)}")
types = [m.get('type', 'unknown') for m in meta]
print("Type distribution:", collections.Counter(types))

# Print first few non-text items to verify paths
for m in meta:
    if m.get('type') != 'text':
        print(f"Sample {m['type']}: {m}")
        break
