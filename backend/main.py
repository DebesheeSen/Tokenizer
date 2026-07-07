import sys
import os
import pickle
import threading
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Resolve backend imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pre_tokenizer import PreTokenizer

app = FastAPI(title="Tokenize Backend", description="FastAPI Backend for BPE Tokenizer Visualizer")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Determine if running on Vercel
IS_VERCEL = os.environ.get("VERCEL") == "1" or "VERCEL" in os.environ

ORIGINAL_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))

if IS_VERCEL:
    # On Vercel, the filesystem is read-only except for /tmp.
    # We use /tmp/models for model read/writes.
    MODELS_DIR = "/tmp/models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Copy pre-packaged models from original models folder to /tmp/models if they don't exist yet
    if os.path.exists(ORIGINAL_MODELS_DIR):
        for file_name in os.listdir(ORIGINAL_MODELS_DIR):
            if file_name.endswith(".pkl"):
                src = os.path.join(ORIGINAL_MODELS_DIR, file_name)
                dst = os.path.join(MODELS_DIR, file_name)
                if not os.path.exists(dst):
                    try:
                        import shutil
                        shutil.copy2(src, dst)
                    except Exception as e:
                        print(f"Failed to copy {file_name} to /tmp/models: {e}")
else:
    MODELS_DIR = ORIGINAL_MODELS_DIR
    os.makedirs(MODELS_DIR, exist_ok=True)

os.makedirs(DATASETS_DIR, exist_ok=True)

class TokenizerManager:
    def __init__(self):
        self._cache = {}

    def get_tokenizer(self, model_name: str) -> PreTokenizer:
        # Clean name to prevent directory traversal
        model_name = os.path.basename(model_name)
        model_path = os.path.join(MODELS_DIR, model_name)
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found.")
            
        # Get modification time to reload if file has changed
        mtime = os.path.getmtime(model_path)
        
        if model_name not in self._cache or self._cache[model_name]["mtime"] < mtime:
            try:
                tokenizer = PreTokenizer()
                tokenizer.load(model_path)
                self._cache[model_name] = {
                    "tokenizer": tokenizer,
                    "mtime": mtime
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error loading model {model_name}: {str(e)}")
                
        return self._cache[model_name]["tokenizer"]

manager = TokenizerManager()

# Global training state
training_status = {
    "status": "idle",  # "idle", "training", "completed", "error"
    "progress": 0,
    "total": 0,
    "message": "",
    "error": ""
}
training_lock = threading.Lock()

class TokenizeRequest(BaseModel):
    text: str
    model: str

class TokenInfo(BaseModel):
    id: int
    text: str
    bytes: List[int]

class TokenizeResponse(BaseModel):
    tokens: List[TokenInfo]
    token_count: int
    char_count: int
    bytes_count: int
    compression_ratio: float
    efficiency: float

class TrainRequest(BaseModel):
    dataset: str
    vocab_size: int
    model_name: str

@app.get("/api/models")
def list_models():
    """List all available tokenizer models."""
    try:
        files = os.listdir(MODELS_DIR)
        pkl_files = [f for f in files if f.endswith(".pkl")]
        
        models_list = []
        for f in pkl_files:
            path = os.path.join(MODELS_DIR, f)
            size = os.path.getsize(path)
            mtime = os.path.getmtime(path)
            models_list.append({
                "name": f,
                "size_bytes": size,
                "modified_at": mtime
            })
            
        # Sort by modification time (newest first)
        models_list.sort(key=lambda x: x["modified_at"], reverse=True)
        return models_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload a new .pkl model file."""
    if not file.filename.endswith(".pkl"):
        raise HTTPException(status_code=400, detail="Only .pkl files are supported.")
        
    filename = os.path.basename(file.filename)
    dest_path = os.path.join(MODELS_DIR, filename)
    
    try:
        content = await file.read()
        
        # Verify it's a valid pickle and contains merges/vocab before saving
        data = pickle.loads(content)
        if not isinstance(data, dict) or "merges" not in data or "vocab" not in data:
            raise HTTPException(status_code=400, detail="Invalid model structure. Must be a pickle dictionary containing 'merges' and 'vocab'.")
            
        with open(dest_path, "wb") as f:
            f.write(content)
            
        return {"filename": filename, "status": "success"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=f"Failed to parse pickle: {str(e)}")

@app.post("/api/tokenize", response_model=TokenizeResponse)
def tokenize_text(request: TokenizeRequest):
    """Tokenize the input text using a specified model."""
    if not request.text:
        return TokenizeResponse(
            tokens=[],
            token_count=0,
            char_count=0,
            bytes_count=0,
            compression_ratio=1.0,
            efficiency=0.0
        )
        
    tokenizer = manager.get_tokenizer(request.model)
    
    try:
        # Encode the text to get token IDs
        token_ids = tokenizer.encode(request.text)
        
        tokens_info = []
        for idx in token_ids:
            if idx in tokenizer.vocab:
                token_bytes = tokenizer.vocab[idx]
            else:
                # Fallback for unknown IDs (should not happen with standard BPE)
                token_bytes = bytes([idx]) if idx < 256 else b"?"
                
            token_text = token_bytes.decode("utf-8", errors="replace")
            tokens_info.append(TokenInfo(
                id=idx,
                text=token_text,
                bytes=list(token_bytes)
            ))
            
        # Stats calculations
        char_count = len(request.text)
        bytes_count = len(request.text.encode("utf-8"))
        token_count = len(token_ids)
        
        compression_ratio = char_count / max(1, token_count)
        # Space savings/efficiency: 1 - (tokens / bytes)
        efficiency = 1.0 - (token_count / max(1, bytes_count))
        
        return TokenizeResponse(
            tokens=tokens_info,
            token_count=token_count,
            char_count=char_count,
            bytes_count=bytes_count,
            compression_ratio=round(compression_ratio, 2),
            efficiency=round(efficiency, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tokenization error: {str(e)}")

@app.get("/api/datasets")
def list_datasets():
    """List available training dataset text files."""
    try:
        files = os.listdir(DATASETS_DIR)
        txt_files = [f for f in files if f.endswith(".txt")]
        
        datasets = []
        for f in txt_files:
            path = os.path.join(DATASETS_DIR, f)
            size = os.path.getsize(path)
            datasets.append({
                "name": f,
                "size_bytes": size
            })
        return datasets
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_training(dataset_name: str, vocab_size: int, model_name: str):
    """Background training worker thread."""
    global training_status
    
    dataset_path = os.path.join(DATASETS_DIR, dataset_name)
    if not os.path.exists(dataset_path):
        with training_lock:
            training_status["status"] = "error"
            training_status["message"] = f"Dataset {dataset_name} not found"
        return

    try:
        with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # Custom tokenizer class that reports progress back
        class ProgressPreTokenizer(PreTokenizer):
            def train(self, text_data, target_vocab_size):
                global training_status
                self.vocab = {i: bytes([i]) for i in range(256)}
                self.history = []
                num_merges = target_vocab_size - 256
                
                with training_lock:
                    training_status["total"] = num_merges
                    training_status["progress"] = 0
                    training_status["message"] = "Pre-tokenizing text..."
                
                chunks = self._pre_tokenize(text_data)
                ids = [list(chunk.encode("utf-8")) for chunk in chunks]

                for i in range(num_merges):
                    # Check if training was aborted or reset
                    with training_lock:
                        if training_status["status"] != "training":
                            break
                        training_status["progress"] = i
                        training_status["message"] = f"Merge {i + 1} of {num_merges}..."

                    pair_counter = {}
                    for chunk in ids:
                        stats = self.get_counts(chunk)
                        for pair, count in stats.items():
                            pair_counter[pair] = pair_counter.get(pair, 0) + count

                    if not pair_counter:
                        break

                    pair = max(pair_counter, key=pair_counter.get)
                    idx = 256 + i
                    ids = [self.merge(chunk, pair, idx) for chunk in ids]
                    self.merges[pair] = idx
                    self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

                    self.history.append({
                        "pair": pair,
                        "frequency": pair_counter[pair],
                        "id": idx,
                    })

        tokenizer = ProgressPreTokenizer()
        tokenizer.train(text, vocab_size)

        # Make sure target path has extension
        if not model_name.endswith(".pkl"):
            model_name += ".pkl"
            
        dest_path = os.path.join(MODELS_DIR, model_name)
        tokenizer.save(dest_path)

        with training_lock:
            training_status["status"] = "completed"
            training_status["progress"] = vocab_size - 256
            training_status["message"] = f"Successfully trained model: {model_name}"
            
    except Exception as e:
        with training_lock:
            training_status["status"] = "error"
            training_status["message"] = f"Training failed: {str(e)}"
            training_status["error"] = str(e)

@app.post("/api/train")
def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """Trigger BPE Tokenizer training in a background task (or synchronously on Vercel)."""
    global training_status
    
    with training_lock:
        if training_status["status"] == "training":
            raise HTTPException(status_code=400, detail="A training process is already running.")
            
        training_status = {
            "status": "training",
            "progress": 0,
            "total": 0,
            "message": "Starting training task...",
            "error": ""
        }
        
    if IS_VERCEL:
        # Run synchronously for Vercel's serverless environment to prevent task freezing
        run_training(request.dataset, request.vocab_size, request.model_name)
        with training_lock:
            if training_status["status"] == "error":
                raise HTTPException(status_code=500, detail=training_status["message"])
            return {"status": "completed", "message": f"Successfully trained model: {request.model_name}"}
    else:
        background_tasks.add_task(
            run_training, 
            request.dataset, 
            request.vocab_size, 
            request.model_name
        )
        return {"status": "started", "message": "Training started in background."}

@app.get("/api/train/status")
def get_train_status():
    """Get the current training progress status."""
    with training_lock:
        return training_status

@app.post("/api/train/reset")
def reset_train_status():
    """Reset training status to idle."""
    global training_status
    with training_lock:
        training_status = {
            "status": "idle",
            "progress": 0,
            "total": 0,
            "message": "",
            "error": ""
        }
        return {"status": "reset"}
