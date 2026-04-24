# Multimodal Semantic Search Engine

A modern, "Zero-Shot" Multimodal Search Engine that allows you to search across **Text, Images, Video, and Audio** simultaneously. Powered by OpenAI's CLIP, Whisper, and FAISS, this engine mathematically aligns different media types into a unified semantic space without requiring custom training.

## Features
- **True Multimodal Search:** Query your database using text, upload an image, drop in a video, or use an audio clip.
- **Zero-Shot Architecture:** Uses pre-trained CLIP and Whisper models. No fine-tuning or custom projection layers required.
- **Cross-Modal Deduplication:** Advanced FAISS retrieval logic ensures that multiple frames from the same video don't crowd your search results.
- **Luxury UI:** A beautifully crafted, responsive, and highly interactive frontend.

---

## 🏗 System Requirements

To run this application effectively, your system should meet the following requirements:

- **OS:** Linux or macOS (Windows supported via WSL2)
- **Python:** 3.10 or higher
- **GPU:** NVIDIA GPU with at least 6GB VRAM highly recommended for fast indexing (CUDA enabled). *CPU-only execution is supported but will be significantly slower.*
- **System Dependencies:**
  - `ffmpeg` (Required by Whisper for audio processing)
  - `libgl1-mesa-glx` (Required by OpenCV for video processing on Linux)

### Installing FFmpeg (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg libsm6 libxext6 -y
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/multi-semantic-searcheng.git
cd multi-semantic-searcheng
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🗄️ Data Preparation (Crucial Step)

**Note:** To keep the repository lightweight, no datasets or vector indices are included. You must provide your own media and build the indices before running the server.

### Step 1: Prepare your Metadata
Create a JSON file (e.g., `data/user_dataset_metadata.json`) formatted like this:
```json
[
  {
    "id": "item_001",
    "text": "A description or caption",
    "image_path": "data/images/photo1.jpg",
    "video_path": "data/videos/clip1.mp4",
    "audio_path": "data/audio/sound1.wav"
  }
]
```
*Note: If an item doesn't have a video or audio file, just leave the string empty.*

### Step 2: Build the Monolithic Index
This script will process all your media through CLIP and Whisper to generate the semantic vectors.
```bash
python src/build_index.py --metadata data/user_dataset_metadata.json --batch_size 32
```
*This will create a massive `index.faiss` file and `index_meta.pkl`.*

### Step 3: Split the Indices
For the UI to work properly and serve separate modality tabs, we split the monolithic index into specific ones (`index_image.faiss`, `index_video.faiss`, `index_audio.faiss`).
```bash
python src/split_index.py
```

---

## 🖥 Running the Server

Once your indices are built, you can start the FastAPI server.

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open your web browser and navigate to: **`http://localhost:8000`**

### Running in the Background
If you want to keep the server running after closing your terminal:
```bash
nohup venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

---

## 🧠 Technical Architecture

1. **The Brain (`src/model.py`)**: A unified architecture utilizing **CLIP (ViT-B/32)** for text, image, and video frames, and **Whisper (tiny)** for audio. All outputs are padded/normalized to a 512-dimensional vector space.
2. **The Database (`faiss`)**: Facebook AI Similarity Search is used to quickly find the nearest neighbors in the 512-dim space.
3. **The Backend (`server.py`)**: A FastAPI controller that handles media uploads, temporary file processing (via OpenCV), and querying the FAISS indices.
4. **The Frontend (`frontend/`)**: Pure HTML/CSS/JS. No heavy web frameworks. It utilizes modern CSS Grid/Flexbox and smooth transitions.
