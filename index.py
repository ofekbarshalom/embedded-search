#!/usr/bin/env python3
"""
Main indexing script — scans your folder and embeds all files
into the ChromaDB vector store using Gemini Embedding 2.

Usage:
    export GEMINI_API_KEY="your-key-here"
    python index.py                          # Index your home folder
    python index.py /path/to/folder          # Index a specific folder
    python index.py --stats                  # Show index stats
    python index.py --clear                  # Clear the entire index
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import config
from scanner import scan_folder, scan_summary, FileCategory
from embedder import GeminiEmbedder
from store import VectorStore
from utils import format_size


def show_scan_preview(root: str):
    """Show what will be indexed before starting."""
    print(f"\nScanning: {root}")
    print("=" * 60)
    summary = scan_summary(root)
    total_files = 0
    total_size = 0
    for cat in sorted(summary.keys()):
        info = summary[cat]
        total_files += info["count"]
        total_size += info["total_size"]
        exts = ", ".join(info["extensions"][:10])
        if len(info["extensions"]) > 10:
            exts += f", ... (+{len(info['extensions']) - 10} more)"
        print(f"  {cat:8s}  {info['count']:5d} files  ({format_size(info['total_size']):>10s})  [{exts}]")
    print("-" * 60)
    print(f"  {'TOTAL':8s}  {total_files:5d} files  ({format_size(total_size):>10s})")
    print()
    return total_files


def run_indexing(root: str, api_key: str = None):
    """Main indexing loop."""
    embedder = GeminiEmbedder(api_key=api_key)
    store = VectorStore()

    total = show_scan_preview(root)
    if total == 0:
        print("No files found to index.")
        return

    print(f"Starting indexing of {total} files...")
    print(f"Existing index has {store.count()} embeddings.\n")

    BATCH_SIZE = 100  # files per API call

    indexed = 0
    errors = 0
    skipped = 0
    start_time = time.time()

    batch = []
    for scanned in scan_folder(root):
        batch.append(scanned)
        if len(batch) >= BATCH_SIZE:
            indexed += len(batch)
            pct = (indexed / total * 100) if total > 0 else 0
            elapsed = time.time() - start_time
            print(f"  [{indexed}/{total}] ({pct:.0f}%) Embedding batch of {len(batch)} files...", end="", flush=True)

            try:
                results = embedder.embed_files_batch(batch)
                if results:
                    store.add_batch(results)
                    print(f"  -> {len(results)} embeddings")
                else:
                    errors += len(batch)
                    print(f"  -> batch failed")
            except Exception as e:
                errors += len(batch)
                print(f"  -> ERROR: {e}")
            batch = []

    # Process remaining files
    if batch:
        indexed += len(batch)
        pct = (indexed / total * 100) if total > 0 else 0
        print(f"  [{indexed}/{total}] ({pct:.0f}%) Embedding batch of {len(batch)} files...", end="", flush=True)
        try:
            results = embedder.embed_files_batch(batch)
            if results:
                store.add_batch(results)
                print(f"  -> {len(results)} embeddings")
            else:
                errors += len(batch)
                print(f"  -> batch failed")
        except Exception as e:
            errors += len(batch)
            print(f"  -> ERROR: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Indexing complete!")
    print(f"  Files processed: {indexed}")
    print(f"  Errors:          {errors}")
    print(f"  Skipped:         {skipped}")
    print(f"  Total embeddings: {store.count()}")
    print(f"  Time:            {elapsed:.1f}s")
    print(f"\nRun 'python app.py' to start the search UI.")


def main():
    parser = argparse.ArgumentParser(description="Gemini Multimodal File Indexer")
    parser.add_argument("folder", nargs="?", default=None,
                        help="Folder to index (default: home folder or SEARCH_ROOT env)")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    parser.add_argument("--clear", action="store_true", help="Clear the entire index")
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
        confirm = input("Are you sure you want to clear the entire index? (y/N): ")
        if confirm.lower() == "y":
            store = VectorStore()
            store.clear()
            print("Index cleared.")
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
