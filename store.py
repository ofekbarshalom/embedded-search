"""
ChromaDB vector store — handles indexing and semantic search.
"""
import chromadb

import config
from embedder import EmbeddingResult


class VectorStore:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.CHROMA_DB_PATH
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _make_id(result: EmbeddingResult) -> str:
        embed_type = result.metadata.get("embed_type", "path")
        if embed_type == "content":
            return f"{result.relative_path}::content_chunk{result.chunk_index}"
        return f"{result.relative_path}::chunk{result.chunk_index}"

    def add_batch(self, results: list[EmbeddingResult]):
        """Add a batch of embedding results."""
        if not results:
            return
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        for r in results:
            doc_id = self._make_id(r)
            ids.append(doc_id)
            embeddings.append(r.embedding)
            documents.append(r.chunk_text or f"[{r.category}] {r.relative_path}")
            metadatas.append({
                "file_path": r.file_path,
                "relative_path": r.relative_path,
                "category": r.category,
                "chunk_index": r.chunk_index,
                "extension": r.metadata.get("extension", ""),
                "folder": r.metadata.get("folder", ""),
                "filename": r.metadata.get("filename", ""),
                "size": r.metadata.get("size", 0),
            })
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(self, query_embedding: list[float], n_results: int = 10,
               category_filter: str = None) -> list[dict]:
        """Search the store with a query embedding."""
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                search_results.append({
                    "id": doc_id,
                    "document": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 1.0,
                    "score": 1.0 - (results["distances"][0][i] if results["distances"] else 1.0),
                })

        return search_results

    def count(self) -> int:
        """Return the number of items in the collection."""
        return self.collection.count()

    def get_stats(self) -> dict:
        """Get stats about what's indexed."""
        count = self.count()
        if count == 0:
            return {"total": 0, "categories": {}}

        # Sample to get category distribution
        sample = self.collection.get(
            limit=min(count, 10000),
            include=["metadatas"],
        )
        categories = {}
        files_seen = set()
        for meta in sample["metadatas"]:
            cat = meta.get("category", "unknown")
            rel = meta.get("relative_path", "")
            if cat not in categories:
                categories[cat] = {"chunks": 0, "files": set()}
            categories[cat]["chunks"] += 1
            categories[cat]["files"].add(rel)
            files_seen.add(rel)

        # Convert sets to counts
        for cat in categories:
            categories[cat] = {
                "chunks": categories[cat]["chunks"],
                "files": len(categories[cat]["files"]),
            }

        return {
            "total_chunks": count,
            "total_files": len(files_seen),
            "categories": categories,
        }

    def clear(self):
        """Delete all data from the collection."""
        self.client.delete_collection(config.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
