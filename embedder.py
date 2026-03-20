"""
Embedding engine — embeds file paths and names via Gemini Embedding API.
"""
import os
import re
import time
from dataclasses import dataclass

from google import genai

import config
from scanner import ScannedFile


@dataclass
class EmbeddingResult:
    file_path: str
    relative_path: str
    category: str
    chunk_index: int
    chunk_text: str
    embedding: list[float]
    metadata: dict


class GeminiEmbedder:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required. Set it as an environment variable or pass it directly.")
        self.client = genai.Client(api_key=self.api_key)
        self.model = config.EMBEDDING_MODEL
        self._request_count = 0
        self._last_request_time = 0

    def _rate_limit(self):
        """Simple rate limiter — max ~100 requests/min to stay safe."""
        self._request_count += 1
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < 1.2:  # ~50 req/min for free tier
            time.sleep(1.2 - elapsed)
        self._last_request_time = time.time()

    def _call_with_retry(self, contents, max_retries=5):
        """Call embed_content with exponential backoff on 429 errors.

        contents can be a single string or a list of strings.
        Returns a single embedding list or a list of embedding lists.
        """
        is_batch = isinstance(contents, list)
        for attempt in range(max_retries):
            self._rate_limit()
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=contents,
                )
                if is_batch:
                    return [list(e.values) for e in result.embeddings]
                return list(result.embeddings[0].values)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80 seconds
                    print(f"\n  Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})...", end="", flush=True)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"Still rate limited after {max_retries} retries")

    def embed_text(self, text: str) -> list[float]:
        """Embed a text string."""
        return self._call_with_retry(text)

    def embed_texts_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call."""
        return self._call_with_retry(texts)

    # Prefix abbreviations: matched at the start of a word, followed by digits
    _PREFIX_EXPANSIONS = {
        "ex": "exercise",
        "hw": "homework",
        "ass": "assignment",
        "assign": "assignment",
        "lec": "lecture",
        "lect": "lecture",
        "lab": "lab",
        "proj": "project",
        "sol": "solution",
        "ans": "answer",
        "ch": "chapter",
        "chap": "chapter",
        "sec": "section",
        "tut": "tutorial",
        "mid": "midterm",
        "ps": "problem set",
        "q": "quiz",
        "rec": "recitation",
        "sum": "summary",
        "rev": "review",
        "prac": "practice",
        "prep": "preparation",
        "ref": "reference",
        "fig": "figure",
        "pt": "part",
        "wk": "week",
    }

    # Whole-word expansions for folder/path names
    _WORD_EXPANSIONS = {
        "calc": "calculus",
        "alg": "algebra",
        "algo": "algorithms",
        "ds": "data structures",
        "os": "operating systems",
        "db": "database",
        "ml": "machine learning",
        "dl": "deep learning",
        "ai": "artificial intelligence",
        "cv": "computer vision",
        "nlp": "natural language processing",
        "oop": "object oriented programming",
        "cs": "computer science",
        "ee": "electrical engineering",
        "math": "mathematics",
        "stats": "statistics",
        "stat": "statistics",
        "phys": "physics",
        "chem": "chemistry",
        "bio": "biology",
        "econ": "economics",
        "prog": "programming",
        "intro": "introduction",
        "adv": "advanced",
        "fund": "fundamentals",
        "sem": "semester",
        "img": "image",
        "pic": "picture",
        "vid": "video",
        "doc": "document",
        "docs": "documents",
        "pres": "presentation",
        "hw": "homework",
        "ex": "exercise",
    }

    def _expand_keywords(self, filename: str, folder: str) -> str:
        """Generate expanded keywords from filename and folder for better search."""
        keywords = []
        name_no_ext = os.path.splitext(filename)[0]

        # Expand prefix+number patterns (e.g. EX3 -> exercise 3, HW02 -> homework 2)
        for prefix, expansion in self._PREFIX_EXPANSIONS.items():
            pattern = re.compile(rf'^{re.escape(prefix)}[_\-]?(\d+)$', re.IGNORECASE)
            match = pattern.match(name_no_ext)
            if match:
                num = match.group(1).lstrip("0") or "0"
                keywords.append(f"{expansion} {num}")
                break

        # Expand abbreviations found in folder path
        folder_words = re.split(r'[\\/\s_\-]+', folder.lower())
        for word in folder_words:
            if word in self._WORD_EXPANSIONS:
                expanded = self._WORD_EXPANSIONS[word]
                if expanded not in keywords:
                    keywords.append(expanded)

        # Add readable version of CamelCase or underscore-separated filename
        readable = re.sub(r'([a-z])([A-Z])', r'\1 \2', name_no_ext)
        readable = re.sub(r'[_\-]+', ' ', readable).strip()
        if readable.lower() != name_no_ext.lower():
            keywords.append(readable)

        return ", ".join(keywords)

    def _prepare_file(self, scanned: ScannedFile) -> tuple[str, dict]:
        """Prepare the text and metadata for a single file (no API call)."""
        metadata = {
            "file_path": scanned.path,
            "relative_path": scanned.relative_path,
            "category": scanned.category.value,
            "extension": scanned.extension,
            "size": scanned.size,
            "folder": os.path.dirname(scanned.relative_path),
            "filename": os.path.basename(scanned.path),
        }

        folder = os.path.dirname(scanned.relative_path)
        filename = os.path.basename(scanned.path)
        keywords = self._expand_keywords(filename, folder)
        path_text = f"File: {filename} | Folder: {folder} | Path: {scanned.relative_path}"
        if keywords:
            path_text += f" | Keywords: {keywords}"

        return path_text, metadata

    def embed_file(self, scanned: ScannedFile) -> list[EmbeddingResult]:
        """
        Embed a single file. Embeds only the file path, folder, and filename
        as text (no file content is read).
        """
        path_text, metadata = self._prepare_file(scanned)

        try:
            embedding = self.embed_text(path_text)
        except Exception as e:
            print(f"  ERROR embedding {scanned.relative_path}: {e}")
            return []

        return [EmbeddingResult(
            file_path=scanned.path,
            relative_path=scanned.relative_path,
            category=scanned.category.value,
            chunk_index=0,
            chunk_text=path_text,
            embedding=embedding,
            metadata=metadata,
        )]

    def embed_files_batch(self, files: list[ScannedFile]) -> list[EmbeddingResult]:
        """
        Embed multiple files in a single API call.
        Returns all successful EmbeddingResults.
        """
        if not files:
            return []

        prepared = [self._prepare_file(f) for f in files]
        texts = [p[0] for p in prepared]

        try:
            embeddings = self.embed_texts_batch(texts)
        except Exception as e:
            print(f"\n  ERROR embedding batch: {e}")
            return []

        results = []
        for i, scanned in enumerate(files):
            path_text, metadata = prepared[i]
            results.append(EmbeddingResult(
                file_path=scanned.path,
                relative_path=scanned.relative_path,
                category=scanned.category.value,
                chunk_index=0,
                chunk_text=path_text,
                embedding=embeddings[i],
                metadata=metadata,
            ))
        return results
