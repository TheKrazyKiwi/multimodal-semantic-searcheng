import os
import torch
import faiss
import pickle
from PIL import Image
import io
import clip
import tempfile
import numpy as np
import whisper
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.model import MultimodalSearchEngine

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("WARNING: OpenCV not found. Video search will return empty results.")

app = FastAPI(title="Multimodal Search Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Resources
model = None
indices = {}
metadata = {}

@app.on_event("startup")
async def load_resources():
    global model, indices, metadata
    print("Loading resources (this might take a few moments)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultimodalSearchEngine(device=device)
    model.eval()

    # Load separate indices
    for modality in ['image', 'video', 'audio']:
        index_path = f"index_{modality}.faiss"
        meta_path = f"index_{modality}_meta.pkl"
        
        if os.path.exists(index_path) and os.path.exists(meta_path):
            try:
                indices[modality] = faiss.read_index(index_path)
                with open(meta_path, "rb") as f:
                    metadata[modality] = pickle.load(f)
            except Exception as e:
                print(f"Failed to load {modality} index: {e}")
                indices[modality] = None
        else:
            print(f"Missing separate index for {modality}.")
            indices[modality] = None
            
    print("Resources loaded successfully.")

# Mount media serving (for images and videos directly via URL)
os.makedirs("data", exist_ok=True)
app.mount("/media", StaticFiles(directory="data"), name="media")

# Frontend mounting moved to bottom


def _search_all_indices(q_vec):
    results = {'image': [], 'video': [], 'audio': []}
    seen_paths = set()
    for modality in ['image', 'video', 'audio']:
        idx = indices.get(modality)
        meta = metadata.get(modality)
        if idx and meta:
            # Search deeply to allow proper deduplication of paths
            k_search = min(100, idx.ntotal)
            if k_search == 0: continue
            
            D, I = idx.search(q_vec, k_search)
            
            # Limits based on UI layout
            limit = 18 if modality == 'image' else 6
            count = 0
            
            for rank, i in enumerate(I[0]):
                if count >= limit: break
                if i == -1: continue
                item = meta[i].copy()
                path = item['content']
                
                # Deduplication Logic
                if path in seen_paths: continue
                seen_paths.add(path)
                
                item['score'] = float(D[0][rank])
                
                # Normalize path for media serving
                if "data/" in path:
                    item['url'] = "/media/" + path.split("data/", 1)[-1]
                else:
                    item['url'] = path
                
                results[modality].append(item)
                count += 1
    return results

@app.post("/api/search/text")
async def search_text(query: str = Body(..., embed=True)):
    try:
        with torch.no_grad():
            q_vec = model.forward_text([query]).cpu().numpy()
        faiss.normalize_L2(q_vec)
        results = _search_all_indices(q_vec)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/api/search/image")
async def search_image(file: UploadFile = File(...)):
    try:
        _, preprocess = clip.load("ViT-B/32", device="cpu") 
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(model.device)
        
        with torch.no_grad():
            q_vec = model.forward_image(image_tensor).cpu().numpy()
        faiss.normalize_L2(q_vec)
        results = _search_all_indices(q_vec)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
@app.post("/api/search/video")
async def search_video(file: UploadFile = File(...)):
    if not HAS_OPENCV:
        return JSONResponse(status_code=500, content={"message": "OpenCV not installed"})
    try:
        _, preprocess = clip.load("ViT-B/32", device="cpu")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            indices = np.linspace(0, total_frames-1, 8, dtype=int)
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frame = preprocess(frame)
                    frames.append(frame)
        cap.release()
        os.remove(tmp_path)
        
        if len(frames) == 0:
            return JSONResponse(status_code=400, content={"message": "Could not read video frames"})
            
        video_tensor = torch.stack(frames).unsqueeze(0).to(model.device)
        with torch.no_grad():
            q_vec = model.forward_video(video_tensor).cpu().numpy()
        faiss.normalize_L2(q_vec)
        results = _search_all_indices(q_vec)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/api/search/audio")
async def search_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
            
        audio = whisper.load_audio(tmp_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).unsqueeze(0).to(model.device)
        
        os.remove(tmp_path)
        
        with torch.no_grad():
            q_vec = model.forward_audio(mel).cpu().numpy()
        faiss.normalize_L2(q_vec)
        results = _search_all_indices(q_vec)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

# Mount frontend at the very end to avoid 405 routing conflicts
import os
os.makedirs("frontend", exist_ok=True)
if not os.path.exists("frontend"):
    print("Warning: frontend directory missing.")
else:
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
