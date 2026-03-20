# Embedded Search

A semantic file search engine powered by Google's Gemini embedding model. Index your local files and search them using natural language queries through a modern web UI.

![PWA](https://img.shields.io/badge/PWA-installable-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## Features

- **Semantic search** — find files by meaning, not just keywords
- **Dual indexing** — index file paths/names (fast) or file contents (deep)
- **Broad file support** — text, code, Office docs (DOCX/XLSX/PPTX), images, video, audio, PDFs
- **Smart keyword expansion** — abbreviations like `HW02` or `EX3` are expanded for better matching
- **Background scanning** — index files while you search, with progress tracking and ETA
- **Installable PWA** — works as a standalone app on desktop and mobile

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your-google-gemini-api-key
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com/).

### 3. Run the app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser. Use the web UI to start a scan and begin searching.

## Indexing

### Via the web UI (recommended)

Set a root folder and start a scan directly from the interface.

### Via CLI

```bash
# Index file names and paths (fast — 1 API call per 100 files)
python index.py /path/to/folder

# Index file contents (slower — reads and chunks each file)
python index_content.py /path/to/folder

# View index stats
python index.py --stats

# Clear the index
python index.py --clear
```

## Configuration

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Google Gemini API key |
| `SEARCH_ROOT` | `~` (home) | Root folder to scan |
| `CHROMA_DB_PATH` | `./chroma_db` | Vector database location |

### File handling

Configured in [config.py](config.py):

- **Text chunk size**: 8,000 chars with 500 char overlap
- **Max file sizes**: 1 MB (text), 20 MB (images), 50 MB (audio/PDF), 100 MB (video)
- **Skipped folders**: `.git`, `node_modules`, `__pycache__`, `.venv`, `build`, `dist`, and more
- **50+ text/code extensions** supported out of the box

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Embeddings | Google Gemini (`gemini-embedding-2-preview`, 3072 dimensions) |
| Vector DB | ChromaDB (cosine similarity, local persistent storage) |
| Frontend | Vanilla HTML/CSS/JS, PWA with Service Worker |
| Doc parsing | python-docx, openpyxl, python-pptx, Pillow |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/search` | Semantic search query |
| `GET` | `/api/stats` | Index statistics |
| `GET` | `/api/preview/<path>` | File content preview |
| `POST` | `/api/scan/start` | Start background scan |
| `GET` | `/api/scan/status` | Scan progress |
| `POST` | `/api/scan/cancel` | Cancel running scan |
| `GET` | `/api/scan/preview` | Estimate scan time |
| `POST` | `/api/scan/set-root` | Change scan root folder |

## Project Structure

```
gemini-search/
├── app.py              # Flask server and API routes
├── config.py           # Models, paths, file type settings
├── embedder.py         # Gemini embedding with keyword expansion
├── store.py            # ChromaDB vector store wrapper
├── scanner.py          # File system scanner
├── utils.py            # Shared utilities
├── index.py            # CLI — index file paths
├── index_content.py    # CLI — index file contents
├── requirements.txt    # Python dependencies
├── static/             # Web UI (HTML, CSS, JS, PWA assets)
└── chroma_db/          # Vector database (auto-created)
```
