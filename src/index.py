import faiss
import numpy as np
import pickle
import os

class VectorIndex:
    def __init__(self, dimension=512, index_file="index.faiss", metadata_file="index_meta.pkl"):
        self.dimension = dimension
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.metadata = [] # List of dicts corresponding to vectors
        
        # Initialize Index
        # Using FlatL2 for exact search on small data. 
        # For large data, use IndexIVFFlat.
        self.index = faiss.IndexFlatL2(dimension)
        
    def add(self, vectors, meta_list):
        """
        vectors: (N, D) numpy array
        meta_list: List of N metadata dicts
        """
        if len(vectors) != len(meta_list):
            raise ValueError("Vectors and metadata list must have same length")
            
        # FAISS expects float32
        vectors = vectors.astype(np.float32)
        self.index.add(vectors)
        self.metadata.extend(meta_list)
        
    def search(self, query_vector, k=5):
        """
        query_vector: (1, D) numpy array
        """
        query_vector = query_vector.astype(np.float32)
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                res = self.metadata[idx].copy()
                res['score'] = float(dist)
                results.append(res)
                
        return results

    def save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"Index saved to {self.index_file}")

    def load(self):
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"Index loaded from {self.index_file}. Size: {self.index.ntotal}")
        else:
            print("Index file not found. Starting fresh.")
