# Interactive BPE Token Visualizer 🔍🎨

An interactive, premium web application to visualize **Byte Pair Encoding (BPE)** tokenization in real-time. Built with a **FastAPI** backend and a **Vite + React (Tailwind CSS v4)** frontend.

---

## ✨ Features

- **Split-Box Token chips**: Visualizes subword boundaries. The character representation (with spaces replaced by clean middle dots `·` for readability) and the token integer ID are displayed in distinct, separate sub-boxes.
- **Byte Breakdown on Hover**: Hovering over any token chip opens a tooltip displaying the exact decimal UTF-8 byte sequence, token ID, and character count.
- **Model Selection & Configuration**: Dynamically lists and switches between BPE models (e.g. `monte_cristo.pkl`, `oliver_twist.pkl`) in the `/models` directory.
- **Custom Model Uploader**: Allows uploading custom `.pkl` tokenizer files directly from the UI.
- **Interactive BPE Training**: Train a brand new BPE tokenizer directly from the UI! Select a training dataset (from the `/datasets` directory), configure the target vocabulary size, name your model, and follow the BPE merge progression with a real-time progress bar.

---

## 🛠️ Project Structure

```
TOKENIZER/
├── backend/
│   ├── main.py              # FastAPI server entry point (routing, BPE caching, training)
│   ├── bpe_tokenizer.py     # Base BPE Tokenizer class
│   ├── pre_tokenizer.py     # Pre-tokenization & multi-process trainer subclasses
│   └── train.py             # CLI training script
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main visualizer workspace component
│   │   ├── index.css        # Tailwind config & layout tokens
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js       # Vite bundler configuration with API Proxy setup
├── datasets/                # Text files used for training tokenizers
├── models/                  # Pickled BPE models (.pkl files)
├── requirements.txt         # Python backend dependencies
└── README.md
```

---

## 🚀 How to Run the Project

Follow these steps to set up and run the frontend and backend servers.

### 1. Backend Setup (FastAPI)

1. Navigate to the root directory.
2. Create and activate a Python virtual environment:
   ```bash
   # Create virtual environment
   python3 -m venv .venv

   # Activate on macOS/Linux
   source .venv/bin/env/activate  # Or source .venv/bin/activate

   # Activate on Windows
   .venv\Scripts\activate
   ```
3. Install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI development server:
   ```bash
   uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   The backend API documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

### 2. Frontend Setup (React + Vite)

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install npm dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   The application will run locally, typically at [http://localhost:5173](http://localhost:5173) or [http://localhost:5174](http://localhost:5174).

---

### 3. Vercel Deployment

This project is pre-configured for a zero-config deployment on Vercel:

1. **Vercel Config**: At the root, `api/index.py` acts as the FastAPI entrypoint and `vercel.json` rewrites all requests matching `/api/*` to it.
2. **Import Project**: Link your GitHub repository in the Vercel Dashboard.
3. **Configure Project Settings**:
   - **Framework Preset**: Other (or Vite)
   - **Root Directory**: `.` (the project root, default)
   - **Build Command**: `npm --prefix frontend install && npm --prefix frontend run build`
   - **Output Directory**: `frontend/dist`
4. Click **Deploy**. Vercel will build the frontend assets, set up the serverless Python functions for `/api`, and host them under the same domain.

---

## 💡 How BPE Tokenization Works in this App

1. **Pre-tokenization**: The input text is split into word-level chunks using regex (matching GPT-4 pattern).
2. **Byte Conversion**: The chunks are broken down into raw UTF-8 byte representations (characters are integers 0-255).
3. **Merges**: The tokenizer applies learned merges sequentially. The most frequent byte pairs are replaced by higher-index token IDs (256 onwards).
4. **Compression Stats**: The visualizer computes:
   - **Tokens count**: Total number of subword tokens.
   - **Space Savings (Efficiency)**: Percent reduction in size compared to raw UTF-8 bytes (`1.0 - (Token Count / Byte Count)`).
   - **Compression Ratio**: Average number of characters represented per token (`Character Count / Token Count`).

---

## 👤 Credits

Developed with ❤️ by **[Debeshee Sen](https://debeshee-sen-portfolio.vercel.app/)**
