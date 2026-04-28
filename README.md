# Local AI Knowledge Base

Ask natural-language questions about any MySQL database.
Trains once, answers offline — no cloud APIs, no data leaves your machine.

## Stack

| Layer | Tech |
|---|---|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | FAISS (cosine similarity) |
| LLM | Ollama (llama3.2 / mistral / phi3) |
| API | FastAPI + Uvicorn |
| Frontend | Vanilla JS widget (zero dependencies) |

## Quick Start

### 1. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> Activate the venv (`source .venv/bin/activate`) in every new terminal before running `train.py` or `app.py`.

### 2. Install & start Ollama

```bash
# Install: https://ollama.com
ollama pull llama3.2   # ~2 GB download, one-time
ollama serve           # keep running in a terminal
```

### 3. Load sample data (optional)

```bash
mysql -u root -p < data/sample_data.sql
```

### 4. Train

```bash
python train.py \
  --db_host=localhost \
  --db_user=root \
  --db_pass=YOUR_PASSWORD \
  --db_name=knowledge_demo
```

Training output is saved to:
- `data/raw/`        — raw JSON schema
- `data/processed/`  — natural-language documents
- `data/vectors/`    — FAISS index + chunk list

**The database is never accessed again after training.**

To retrain: add `--retrain`

### 5. Start the API server

```bash
python app.py
# Server runs at http://localhost:8000
```

### 6. Use the widget

Open `frontend/index.html` in a browser, or embed the widget anywhere:

```html
<script src="http://localhost:8000/static/chat-widget.js"></script>
<script>
  ChatWidget.init({
    apiUrl:       "http://localhost:8000/ask",
    title:        "AI Assistant",
    primaryColor: "#4F46E5",
  });
</script>
```

### 7. Admin UI

Open `frontend/admin.html` to:
- Check server status
- Browse knowledge chunks
- Test Q&A live

---

## API

### `POST /ask`

```json
// Request
{ "question": "How many users are there?", "top_k": 5 }

// Response
{
  "answer": "There are 5 users in the users table.",
  "sources_count": 3,
  "latency_ms": 1240.5
}
```

### `GET /health`

```json
{ "status": "ok", "index_loaded": true, "chunks_count": 48 }
```

### `GET /chunks?limit=20&offset=0`

Browse raw indexed chunks (debug).

Interactive docs at `http://localhost:8000/docs`

---

## Widget Options

```js
ChatWidget.init({
  apiUrl:       "http://localhost:8000/ask",  // backend URL
  title:        "AI Assistant",               // header text
  placeholder:  "Ask a question…",            // input placeholder
  primaryColor: "#4F46E5",                    // any CSS colour
  position:     "bottom-right",              // or "bottom-left"
  top_k:        5,                           // context chunks per query
});
```

---

## Project Structure

```
├── train.py            CLI: extract → embed → save
├── app.py              FastAPI server entry point
├── requirements.txt
├── .env.example
│
├── db/
│   └── extractor.py    MySQL schema + sample row extraction
│
├── embeddings/
│   └── vector_store.py FAISS build / save / load / search
│
├── llm/
│   └── model.py        Ollama HTTP wrapper
│
├── api/
│   └── routes.py       /ask  /health  /chunks
│
├── data/
│   ├── raw/            Raw JSON from DB (timestamped)
│   ├── processed/      NL documents + readable .txt
│   ├── vectors/        faiss.index + chunks.json
│   └── sample_data.sql Example MySQL schema + data
│
└── frontend/
    ├── chat-widget.js  Embeddable floating widget
    ├── index.html      Demo / docs page
    └── admin.html      Admin dashboard
```

---

## Environment Variables

Copy `.env.example` to `.env`:

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=info
```

---

## Changing the LLM Model

Any model available in Ollama works:

```bash
ollama pull mistral
OLLAMA_MODEL=mistral python app.py
```

Recommended lightweight models:
- `llama3.2` (3B) — best quality/speed balance
- `phi3` (3.8B) — Microsoft, fast
- `mistral` (7B) — strong reasoning, needs ~5GB RAM
