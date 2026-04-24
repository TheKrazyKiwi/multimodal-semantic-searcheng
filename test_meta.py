import pickle

with open('index_video_meta.pkl', 'rb') as f:
    meta = pickle.load(f)
    print("Total items:", len(meta))
    paths = set(m['content'] for m in meta)
    print("Unique videos:", len(paths))
    for p in paths:
        print(" -", p)
