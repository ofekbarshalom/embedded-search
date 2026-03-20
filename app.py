"""
Flask web UI for Embedded Search.
"""
import os
import sys
import threading
import time
import math

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, send_from_directory
from google import genai

import config
from store import VectorStore
from scanner import scan_folder, scan_summary, FileCategory
from embedder import GeminiEmbedder, EmbeddingResult
from utils import read_file_content, chunk_text

app = Flask(__name__, static_folder="static")

# Lazy initialization
_store = None
_client = None

# Scan state — shared between the background thread and API
_scan_state = {
    "running": False,
    "type": None,           # "names" or "content"
    "progress": 0,          # files processed
    "total": 0,             # total files to process
    "chunks_done": 0,       # chunks embedded (content mode)
    "errors": 0,
    "skipped": 0,
    "current_file": "",
    "start_time": 0,
    "avg_time_per_item": 0, # rolling average seconds per API call
    "done": False,
    "error_message": "",
    "cancelled": False,
}


def get_store():
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def get_client():
    global _client
    if _client is None:
        api_key = config.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        _client = genai.Client(api_key=api_key)
    return _client


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/sw.js")
def service_worker():
    return send_from_directory("static", "sw.js", mimetype="application/javascript")



@app.route("/api/scan/set-root", methods=["POST"])
def set_scan_root():
    """Change the scan root folder."""
    data = request.json or {}
    new_root = data.get("root", "").strip()
    if not new_root:
        return jsonify({"error": "root is required"}), 400
    if not os.path.isdir(new_root):
        return jsonify({"error": f"'{new_root}' is not a valid directory"}), 400
    config.ROOT_FOLDER = new_root
    return jsonify({"root": config.ROOT_FOLDER})


@app.route("/api/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "").strip()
    n_results = data.get("n_results", 15)
    category = data.get("category", None)

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        client = get_client()
        result = client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=query,
        )
        query_embedding = list(result.embeddings[0].values)

        store = get_store()
        results = store.search(
            query_embedding=query_embedding,
            n_results=n_results,
            category_filter=category if category else None,
        )

        # Deduplicate by file path, keeping best score
        seen_files = {}
        for r in results:
            fp = r["metadata"].get("relative_path", "")
            if fp not in seen_files or r["score"] > seen_files[fp]["score"]:
                seen_files[fp] = r

        deduped = sorted(seen_files.values(), key=lambda x: x["score"], reverse=True)

        return jsonify({"results": deduped, "query": query})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def stats():
    try:
        store = get_store()
        return jsonify(store.get_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/preview/<path:filepath>")
def preview(filepath):
    """Return a text preview of a file."""
    full_path = os.path.join(config.ROOT_FOLDER, filepath)
    ext = os.path.splitext(filepath)[1].lower()
    if ext in config.TEXT_EXTENSIONS or ext in config.OFFICE_EXTENSIONS:
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(5000)
            return jsonify({"type": "text", "content": content})
        except FileNotFoundError:
            return jsonify({"error": "File not found"}), 404
        except Exception:
            return jsonify({"type": "text", "content": "[Could not read file]"})
    elif ext in config.IMAGE_EXTENSIONS:
        return jsonify({"type": "image", "path": full_path})
    else:
        try:
            size = os.path.getsize(full_path)
        except FileNotFoundError:
            return jsonify({"error": "File not found"}), 404
        return jsonify({"type": "other", "content": f"[{ext} file — {size} bytes]"})



# --- Scan endpoints ---

@app.route("/api/scan/preview")
def scan_preview():
    """Count files and estimate scan time without starting a scan."""
    try:
        root = config.ROOT_FOLDER
        summary = scan_summary(root)
        total_files = 0
        readable_files = 0
        categories = {}
        for cat, info in summary.items():
            total_files += info["count"]
            if cat in ("text", "office"):
                readable_files += info["count"]
            categories[cat] = info["count"]

        # Estimate: ~1.2s per API call
        # Names: 1 API call per batch of 100
        names_api_calls = math.ceil(total_files / 100)
        names_est_seconds = names_api_calls * 1.2

        # Content: ~1 API call per chunk, most files are 1 chunk
        # Rough estimate: 1.3 chunks per file on average
        content_est_chunks = int(readable_files * 1.3)
        content_est_seconds = content_est_chunks * 1.2

        return jsonify({
            "root": root,
            "total_files": total_files,
            "readable_files": readable_files,
            "categories": categories,
            "estimates": {
                "names": {"files": total_files, "seconds": round(names_est_seconds)},
                "content": {"files": readable_files, "chunks": content_est_chunks,
                            "seconds": round(content_est_seconds)},
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _embed_and_store_batch(batch, embedder, store):
    """Embed a batch of files and store results. Updates _scan_state."""
    t0 = time.time()
    try:
        results = embedder.embed_files_batch(batch)
        if results:
            store.add_batch(results)
        else:
            _scan_state["errors"] += len(batch)
    except Exception:
        _scan_state["errors"] += len(batch)

    elapsed = time.time() - t0
    n = _scan_state["progress"] + len(batch)
    old_avg = _scan_state["avg_time_per_item"]
    _scan_state["avg_time_per_item"] = (old_avg * _scan_state["progress"] + elapsed) / n if n else 0
    _scan_state["progress"] += len(batch)


def _run_name_scan(root, api_key):
    """Background thread: index file names/paths."""
    global _scan_state
    try:
        embedder = GeminiEmbedder(api_key=api_key)
        store = VectorStore()
        BATCH_SIZE = 100

        files = list(scan_folder(root))
        _scan_state["total"] = len(files)

        batch = []
        for scanned in files:
            if _scan_state["cancelled"]:
                break
            batch.append(scanned)
            _scan_state["current_file"] = scanned.relative_path

            if len(batch) >= BATCH_SIZE:
                _embed_and_store_batch(batch, embedder, store)
                batch = []

        if batch:
            _embed_and_store_batch(batch, embedder, store)

        _scan_state["done"] = True
    except Exception as e:
        _scan_state["error_message"] = str(e)
        _scan_state["done"] = True
    finally:
        _scan_state["running"] = False


def _run_content_scan(root, api_key):
    """Background thread: index file contents."""
    global _scan_state
    READABLE = {FileCategory.TEXT, FileCategory.OFFICE}
    try:
        embedder = GeminiEmbedder(api_key=api_key)
        store = VectorStore()

        files = [f for f in scan_folder(root) if f.category in READABLE]
        _scan_state["total"] = len(files)

        for scanned in files:
            if _scan_state["cancelled"]:
                break
            _scan_state["current_file"] = scanned.relative_path

            content = read_file_content(scanned.path)
            if not content or not content.strip():
                _scan_state["skipped"] += 1
                _scan_state["progress"] += 1
                continue

            folder = os.path.dirname(scanned.relative_path)
            filename = os.path.basename(scanned.path)
            chunks = chunk_text(content)

            texts_to_embed = []
            metas = []
            for ci, chunk in enumerate(chunks):
                header = f"File: {filename} | Folder: {folder}\n\n"
                texts_to_embed.append(header + chunk)
                metas.append({
                    "file_path": scanned.path,
                    "relative_path": scanned.relative_path,
                    "category": scanned.category.value,
                    "extension": scanned.extension,
                    "size": scanned.size,
                    "folder": folder,
                    "filename": filename,
                    "chunk_index": ci,
                    "total_chunks": len(chunks),
                    "embed_type": "content",
                })

            try:
                t0 = time.time()
                embeddings = embedder.embed_texts_batch(texts_to_embed)
                elapsed = time.time() - t0

                # Update rolling average (per chunk)
                done = _scan_state["chunks_done"]
                old_avg = _scan_state["avg_time_per_item"]
                n_chunks = len(embeddings)
                avg_per_chunk = elapsed / n_chunks if n_chunks else 0
                _scan_state["avg_time_per_item"] = (old_avg * done + avg_per_chunk * n_chunks) / (done + n_chunks) if (done + n_chunks) else 0
                _scan_state["chunks_done"] += n_chunks

                file_results = [
                    EmbeddingResult(
                        file_path=scanned.path,
                        relative_path=scanned.relative_path,
                        category=scanned.category.value,
                        chunk_index=ci,
                        chunk_text=texts_to_embed[ci],
                        embedding=embeddings[ci],
                        metadata=metas[ci],
                    )
                    for ci in range(n_chunks)
                ]
                store.add_batch(file_results)
            except Exception:
                _scan_state["errors"] += len(chunks)

            _scan_state["progress"] += 1

        _scan_state["done"] = True
    except Exception as e:
        _scan_state["error_message"] = str(e)
        _scan_state["done"] = True
    finally:
        _scan_state["running"] = False


@app.route("/api/scan/start", methods=["POST"])
def scan_start():
    """Start a scan in the background. type = 'names' or 'content'."""
    global _scan_state
    if _scan_state["running"]:
        return jsonify({"error": "A scan is already running"}), 409

    data = request.json or {}
    scan_type = data.get("type", "names")
    if scan_type not in ("names", "content"):
        return jsonify({"error": "type must be 'names' or 'content'"}), 400

    api_key = config.GEMINI_API_KEY
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY not set"}), 500

    root = config.ROOT_FOLDER

    _scan_state = {
        "running": True,
        "type": scan_type,
        "progress": 0,
        "total": 0,
        "chunks_done": 0,
        "errors": 0,
        "skipped": 0,
        "current_file": "Scanning files...",
        "start_time": time.time(),
        "avg_time_per_item": 1.2,
        "done": False,
        "error_message": "",
        "cancelled": False,
    }

    target = _run_name_scan if scan_type == "names" else _run_content_scan
    thread = threading.Thread(target=target, args=(root, api_key), daemon=True)
    thread.start()

    return jsonify({"status": "started", "type": scan_type})


@app.route("/api/scan/status")
def scan_status():
    """Return current scan progress."""
    s = _scan_state
    elapsed = time.time() - s["start_time"] if s["start_time"] else 0

    # ETA calculation
    remaining = 0
    if s["running"] and s["total"] > 0 and s["progress"] > 0:
        if s["type"] == "content":
            # For content, ETA is based on avg time per chunk * estimated remaining chunks
            files_left = s["total"] - s["progress"]
            # Use actual chunks/file ratio so far
            if s["progress"] - s["skipped"] > 0:
                chunks_per_file = s["chunks_done"] / (s["progress"] - s["skipped"])
            else:
                chunks_per_file = 1.3
            remaining_chunks = files_left * chunks_per_file
            remaining = remaining_chunks * s["avg_time_per_item"]
        else:
            # For names, ETA based on avg time per file
            files_left = s["total"] - s["progress"]
            remaining = files_left * s["avg_time_per_item"]

    return jsonify({
        "running": s["running"],
        "done": s["done"],
        "type": s["type"],
        "progress": s["progress"],
        "total": s["total"],
        "chunks_done": s["chunks_done"],
        "errors": s["errors"],
        "skipped": s["skipped"],
        "current_file": s["current_file"],
        "elapsed": round(elapsed),
        "eta_seconds": round(max(0, remaining)),
        "error_message": s["error_message"],
        "cancelled": s.get("cancelled", False),
    })


@app.route("/api/scan/cancel", methods=["POST"])
def scan_cancel():
    """Cancel a running scan."""
    global _scan_state
    if not _scan_state["running"]:
        return jsonify({"error": "No scan is running"}), 400
    _scan_state["cancelled"] = True
    return jsonify({"status": "cancelling"})


if __name__ == "__main__":
    print(f"Starting Embedded Search UI...")
    print(f"Index has {get_store().count()} embeddings")
    print(f"Open http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
