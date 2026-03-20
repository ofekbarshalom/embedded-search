"""
Configuration for Gemini Multimodal Search
"""
import os

from dotenv import load_dotenv
load_dotenv()

# --- API ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# --- Model ---
EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIMENSIONS = 3072

# --- ChromaDB ---
# On Windows, the DB is stored alongside the project. When running in a Linux VM
# or container, override with CHROMA_DB_PATH env var if the mounted folder doesn't
# support SQLite locking (e.g. network/FUSE mounts).
_default_db = os.path.join(os.path.dirname(__file__), "chroma_db")
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", _default_db)
COLLECTION_NAME = "file_embeddings"

# --- File scanning ---
ROOT_FOLDER = os.environ.get("SEARCH_ROOT", os.path.expanduser("~"))

# Max file sizes (bytes) before skipping
MAX_TEXT_FILE_SIZE = 1_000_000       # 1 MB
MAX_IMAGE_FILE_SIZE = 20_000_000     # 20 MB
MAX_VIDEO_FILE_SIZE = 100_000_000    # 100 MB
MAX_AUDIO_FILE_SIZE = 50_000_000     # 50 MB
MAX_PDF_FILE_SIZE = 50_000_000       # 50 MB

# Text chunk size (characters) — ~4 chars per token, target ~2000 tokens
TEXT_CHUNK_SIZE = 8000
TEXT_CHUNK_OVERLAP = 500

# --- Supported extensions mapped to categories ---
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"}
PDF_EXTENSIONS = {".pdf"}

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".cpp", ".hpp", ".c", ".h", ".cs", ".go", ".rs", ".rb",
    ".php", ".swift", ".kt", ".kts", ".scala", ".r", ".sql", ".sh",
    ".bash", ".zsh", ".bat", ".ps1", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".conf", ".xml", ".html", ".htm", ".css", ".scss", ".sass",
    ".less", ".json", ".csv", ".tsv", ".tex", ".log", ".env", ".gitignore",
    ".dockerfile", ".makefile", ".cmake", ".gradle", ".sbt",
    ".ipynb", ".mjs", ".mts", ".editorconfig",
}

# Extensions that need conversion before embedding
OFFICE_EXTENSIONS = {".docx", ".xlsx", ".pptx"}

# Folders to skip entirely
SKIP_FOLDERS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".idea", ".vscode", ".vs", "bin", "obj", "build", "dist",
    "target", ".gradle", ".next", "Library", "Temp", "Logs",
    ".antigravity", "AppData", "Local Settings", "Application Data",
    "Recent", "SendTo", "Start Menu", "Templates", "Cookies",
    "NetHood", "PrintHood", "Links", "Searches", "Contacts",
    "Favorites", "CrossDevice", "My Documents",
    "chroma_db", "embedded-search",
}

# File extensions to skip entirely (binary/system)
SKIP_EXTENSIONS = {
    ".exe", ".dll", ".so", ".dylib", ".bin", ".obj", ".o", ".a", ".lib",
    ".pdb", ".pyc", ".pyo", ".class", ".dex",
    ".vmdk", ".vmx", ".vmem", ".vmss", ".vmsd", ".vmxf", ".nvram",
    ".iso", ".img", ".vhd", ".vhdx",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".whl", ".jar", ".war", ".ear",
    ".lnk", ".url", ".desktop",
    ".dat", ".db", ".sqlite", ".mdb",
    ".lock", ".pack", ".idx",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".min.js", ".min.css",
    ".map",
    ".sample",
}

# MIME types for Gemini API
MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".pdf": "application/pdf",
}
