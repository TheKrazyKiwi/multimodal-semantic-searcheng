
import streamlit as st
import torch
import os
from PIL import Image
import numpy as np
import faiss
import pickle
import clip
from src.model import MultimodalSearchEngine
from src.index import VectorIndex
from src.data.dataset import MultimodalDataset
from torch.utils.data import DataLoader

@st.cache_resource
def load_resources():
    st.text("Loading Model (Encoders + Projection Heads)...")
    device = "cpu"
    # Load Custom Model
    model = MultimodalSearchEngine(device=device)
    if os.path.exists("checkpoints/last_model.pt"):
        st.text(f"Loading weights from checkpoints/last_model.pt")
        model.load_state_dict(torch.load("checkpoints/last_model.pt", map_location=device))
    model.eval()
    
    # Load Split Indices
    indices = {}
    metadata = {}
    
    for modality in ['image', 'video', 'audio']:
        index_path = f"index_{modality}.faiss"
        meta_path = f"index_{modality}_meta.pkl"
        
        if os.path.exists(index_path) and os.path.exists(meta_path):
            st.text(f"Loading {modality.capitalize()} Index...")
            try:
                indices[modality] = faiss.read_index(index_path)
                with open(meta_path, "rb") as f:
                    metadata[modality] = pickle.load(f)
            except Exception as e:
                st.error(f"Failed to load {modality} index: {e}")
                indices[modality] = None
        else:
            # Fallback (optional, but good for robustness)
            st.warning(f"Missing separate index for {modality}. Run split_index.py first.")
            indices[modality] = None

    return model, indices, metadata

def main():
    st.set_page_config(layout="wide", page_title="Multimodal Search")
    st.title("Multimodal Search Engine 🔍")
    
    # Load resources
    try:
        model, indices, metadata = load_resources()
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return

    # Info
    st.info("Searching across separate indices for improved visibility of all media types.")

    # Main Search UI
    tab1, tab2 = st.tabs(["Text Search", "Image Search"])
    
    with tab1:
        query_text = st.text_input("Enter search query...")
        if query_text:
            if st.button("Search Text"):
                # Use custom model forward pass (includes projection)
                with torch.no_grad():
                    q_vec = model.forward_text([query_text]).cpu().numpy()
                faiss.normalize_L2(q_vec)
                
                # Search Each Index
                results = {'image': [], 'video': [], 'audio': []}
                seen_paths = set()
                
                for modality in ['image', 'video', 'audio']:
                    idx = indices.get(modality)
                    meta = metadata.get(modality)
                    
                    if idx and meta:
                        k = 12 if modality == 'image' else 5
                        D, I = idx.search(q_vec, k)
                        
                        for rank, i in enumerate(I[0]):
                            if i == -1: continue
                            item = meta[i].copy()
                            path = item['content']
                            
                            # Deduplication Logic
                            if path in seen_paths: continue
                            seen_paths.add(path)
                            
                            item['score'] = float(D[0][rank])
                            results[modality].append(item)

                # Display Results
                st.subheader("Results")
                
                # Videos
                if results['video']:
                    st.markdown(f"### 🎬 Videos ({len(results['video'])} found)")
                    for res in results['video']: 
                        with st.container():
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                if os.path.exists(res['content']):
                                    st.video(res['content'])
                                else:
                                    st.error(f"File not found: {res['content']}")
                            with col2:
                                st.write(f"**Score:** {res['score']:.4f}")
                                st.caption(f"Path: {os.path.basename(res['content'])}")
                        st.divider()

                # Audio
                if results['audio']:
                    st.markdown(f"### 🎵 Audio ({len(results['audio'])} found)")
                    for res in results['audio']:
                        with st.container():
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                if os.path.exists(res['content']):
                                    st.audio(res['content'])
                                else:
                                    st.error(f"File not found: {res['content']}")
                            with col2:
                                st.write(f"**Score:** {res['score']:.4f}")
                                st.caption(f"Path: {os.path.basename(res['content'])}")
                        st.divider()

                # Images
                if results['image']:
                    st.markdown(f"### 🖼️ Images ({len(results['image'])} found)")
                    cols = st.columns(3)
                    for idx, res in enumerate(results['image']):
                        with cols[idx % 3]:
                            if os.path.exists(res['content']):
                                st.image(res['content'], use_container_width=True)
                                st.caption(f"{res['score']:.4f}")
                            else:
                                st.error("Missing file")
                    st.divider()
    
    with tab2:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='Uploaded Image', width=300)
            
            if st.button("Search Image"):
                # Use custom model forward pass (includes preprocessing and projection)
                transform = model.image_encoder.preprocess
                img_tensor = transform(image).unsqueeze(0).to(model.device)
                
                with torch.no_grad():
                    q_vec = model.forward_image(img_tensor).cpu().numpy()
                faiss.normalize_L2(q_vec)
                
                # Search Each Index
                results = {'image': [], 'video': [], 'audio': []}
                seen_paths = set()
                
                for modality in ['image', 'video', 'audio']:
                    idx = indices.get(modality)
                    meta = metadata.get(modality)
                    
                    if idx and meta:
                        k = 12 if modality == 'image' else 5
                        D, I = idx.search(q_vec, k)
                        
                        for rank, i in enumerate(I[0]):
                            if i == -1: continue
                            item = meta[i].copy()
                            path = item['content']
                            
                            # Deduplication Logic
                            if path in seen_paths: continue
                            seen_paths.add(path)
                            
                            item['score'] = float(D[0][rank])
                            results[modality].append(item)

                # Display Results (Copy of above logic)
                st.subheader("Results")
                
                # Videos
                if results['video']:
                    st.markdown(f"### 🎬 Videos ({len(results['video'])} found)")
                    for res in results['video']: 
                        with st.container():
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                if os.path.exists(res['content']):
                                    st.video(res['content'])
                                else:
                                    st.error(f"File not found: {res['content']}")
                            with col2:
                                st.write(f"**Score:** {res['score']:.4f}")
                                st.caption(f"Path: {os.path.basename(res['content'])}")
                        st.divider()

                # Audio
                if results['audio']:
                    st.markdown(f"### 🎵 Audio ({len(results['audio'])} found)")
                    for res in results['audio']:
                        with st.container():
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                if os.path.exists(res['content']):
                                    st.audio(res['content'])
                                else:
                                    st.error(f"File not found: {res['content']}")
                            with col2:
                                st.write(f"**Score:** {res['score']:.4f}")
                                st.caption(f"Path: {os.path.basename(res['content'])}")
                        st.divider()

                # Images
                if results['image']:
                    st.markdown(f"### 🖼️ Images ({len(results['image'])} found)")
                    cols = st.columns(3)
                    for idx, res in enumerate(results['image']):
                        with cols[idx % 3]:
                            if os.path.exists(res['content']):
                                st.image(res['content'], use_container_width=True)
                                st.caption(f"{res['score']:.4f}")
                            else:
                                st.error("Missing file")
                    st.divider()

if __name__ == "__main__":
    main()
