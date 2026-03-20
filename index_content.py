#!/usr/bin/env python3
"""
Content indexing script — reads file contents and embeds them
into the ChromaDB vector store using Gemini Embedding 2.

Unlike index.py (which embeds file names/paths), this script reads
the actual text content of files, chunks it, and embeds each chunk.

Usage:
    export GEMINI_API_KEY="your-key-here"
    python index_content.py                      # Index your home folder
    python index_content.py /path/to/folder      # Index a specific folder
    python index_content.py --stats              # Show content index stats
    python index_content.py --clear              # Clear only content embeddings
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import config
from scanner import scan_folder, scan_summary, FileCategory
from embedder import GeminiEmbedder, EmbeddingResult
from store import VectorStore
from utils import read_file_content, chunk_text, format_size


# File categories whose content we can read as text
READABLE_CATEGORIES = {FileCategory.TEXT, FileCategory.OFFICE}


def show_scan_preview(root: str):
    """Show what will be indexed before starting."""
    print(f"\nScanning: {root}")
    print("=" * 60)
    summary = scan_summary(root)
    total_files = 0
    readable_files = 0
    for cat in sorted(summary.keys()):
        info = summary[cat]
        total_files += info["count"]
        is_readable = cat in ("text", "office")
        if is_readable:
            readable_files += info["count"]
        marker = " *" if is_readable else ""
        exts = ", ".join(info["extensions"][:10])
        if len(info["extensions"]) > 10:
            exts += f", ... (+{len(info['extensions']) - 10} more)"
        print(f"  {cat:8s}  {info['count']:5d} files  ({format_size(info['total_size']):>10s})  [{exts}]{marker}")
    print("-" * 60)
    print(f"  {'TOTAL':8s}  {total_files:5d} files  (readable: {readable_files})")
    print(f"  * = will be content-indexed\n")
    return readable_files


def run_indexing(root: str, api_key: str = None):
    """Main content indexing loop."""
    embedder = GeminiEmbedder(api_key=api_key)
    store = VectorStore()

    total_readable = show_scan_preview(root)
    if total_readable == 0:
        print("No readable files found to index.")
        return

    print(f"Starting content indexing of {total_readable} readable files...")
    print(f"Existing index has {store.count()} embeddings.\n")

    indexed_files = 0
    indexed_chunks = 0
    errors = 0
    skipped = 0
    start_time = time.time()

    for scanned in scan_folder(root):
        if scanned.category not in READABLE_CATEGORIES:
            continue

        content = read_file_content(scanned.path)
        if not content or not content.strip():
            skipped += 1
            continue

        indexed_files += 1
        folder = os.path.dirname(scanned.relative_path)
        filename = os.path.basename(scanned.path)
        pct = (indexed_files / total_readable * 100) if total_readable > 0 else 0

        chunks = chunk_text(content)
        print(f"  [{indexed_files}/{total_readable}] ({pct:.0f}%) {scanned.relative_path} ({len(chunks)} chunk{'s' if len(chunks) != 1 else ''})...", end="", flush=True)

        file_results = []
        file_error = False
        for ci, chunk in enumerate(chunks):
            # Prepend file context to each chunk for better retrieval
            header = f"File: {filename} | Folder: {folder}\n\n"
            text_to_embed = header + chunk

            meta = {
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
            }

            try:
                embedding = embedder.embed_text(text_to_embed)
                file_results.append(EmbeddingResult(
                    file_path=scanned.path,
                    relative_path=scanned.relative_path,
                    category=scanned.category.value,
                    chunk_index=ci,
                    chunk_text=text_to_embed,
                    embedding=embedding,
                    metadata=meta,
                ))
            except Exception as e:
                errors += 1
                file_error = True
                print(f" ERROR on chunk {ci}: {e}")
                break

        if file_results:
            store.add_batch(file_results)
            indexed_chunks += len(file_results)

        if not file_error:
            print(f" ok")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Content indexing complete!")
    print(f"  Files processed: {indexed_files}")
    print(f"  Chunks embedded: {indexed_chunks}")
    print(f"  Errors:          {errors}")
    print(f"  Skipped (empty): {skipped}")
    print(f"  Total embeddings: {store.count()}")
    print(f"  Time:            {elapsed:.1f}s")
    print(f"\nRun 'python app.py' to start the search UI.")


def main():
    parser = argparse.ArgumentParser(description="Content Indexer — embeds file contents")
    parser.add_argument("folder", nargs="?", default=None,
                        help="Folder to index (default: home folder or SEARCH_ROOT env)")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    parser.add_argument("--clear", action="store_true",
                        help="Clear content embeddings (keeps path embeddings from index.py)")
    parser.add_argument("--api-key", default=None, help="Gemini API key (or set GEMINI_API_KEY env)")
    args = parser.parse_args()

    if args.stats:
        store = VectorStore()
        stats = store.get_stats()
        print(f"\nIndex Statistics")
        print("=" * 40)
        print(f"  Total files:      {stats.get('total_files', 0)}")
        print(f"  Total embeddings: {stats.get('total_chunks', 0)}")
        for cat, info in stats.get("categories", {}).items():
            print(f"  {cat:8s}  {info['files']} files, {info['chunks']} chunks")
        return

    if args.clear:
        confirm = input("Clear content embeddings? Path embeddings will be kept. (y/N): ")
        if confirm.lower() == "y":
            store = VectorStore()
            # Delete only content embeddings by looking for content_chunk IDs
            all_data = store.collection.get(include=["metadatas"])
            content_ids = [
                id for id, meta in zip(all_data["ids"], all_data["metadatas"])
                if meta.get("embed_type") == "content" or "::content_chunk" in id
            ]
            if content_ids:
                # ChromaDB delete has a batch limit
                for i in range(0, len(content_ids), 5000):
                    store.collection.delete(ids=content_ids[i:i + 5000])
                print(f"Cleared {len(content_ids)} content embeddings.")
            else:
                print("No content embeddings found.")
        return

    # Determine root folder
    root = args.folder or config.ROOT_FOLDER
    if not os.path.isdir(root):
        print(f"Error: '{root}' is not a valid directory.")
        sys.exit(1)

    # Check API key
    api_key = args.api_key or config.GEMINI_API_KEY
    if not api_key:
        print("Error: No API key provided.")
        print("Set GEMINI_API_KEY environment variable or use --api-key flag.")
        sys.exit(1)

    run_indexing(root, api_key=api_key)


if __name__ == "__main__":
    main()
