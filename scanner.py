"""
File scanner — walks the folder tree, classifies files by type,
and yields them for embedding.
"""
import os
from dataclasses import dataclass
from enum import Enum
from typing import Generator

import config


class FileCategory(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    PDF = "pdf"
    OFFICE = "office"
    SKIP = "skip"


@dataclass
class ScannedFile:
    path: str
    relative_path: str
    category: FileCategory
    extension: str
    size: int


def classify_extension(ext: str) -> FileCategory:
    """Classify a file extension into a category."""
    ext = ext.lower()
    if ext in config.SKIP_EXTENSIONS:
        return FileCategory.SKIP
    if ext in config.IMAGE_EXTENSIONS:
        return FileCategory.IMAGE
    if ext in config.VIDEO_EXTENSIONS:
        return FileCategory.VIDEO
    if ext in config.AUDIO_EXTENSIONS:
        return FileCategory.AUDIO
    if ext in config.PDF_EXTENSIONS:
        return FileCategory.PDF
    if ext in config.OFFICE_EXTENSIONS:
        return FileCategory.OFFICE
    if ext in config.TEXT_EXTENSIONS:
        return FileCategory.TEXT
    return FileCategory.SKIP


def should_skip_folder(folder_name: str) -> bool:
    """Check if a folder should be skipped."""
    return folder_name in config.SKIP_FOLDERS or folder_name.startswith(".")


def is_within_size_limit(scanned: ScannedFile) -> bool:
    """Check if a file is within the size limit for its category."""
    limits = {
        FileCategory.TEXT: config.MAX_TEXT_FILE_SIZE,
        FileCategory.IMAGE: config.MAX_IMAGE_FILE_SIZE,
        FileCategory.VIDEO: config.MAX_VIDEO_FILE_SIZE,
        FileCategory.AUDIO: config.MAX_AUDIO_FILE_SIZE,
        FileCategory.PDF: config.MAX_PDF_FILE_SIZE,
        FileCategory.OFFICE: config.MAX_TEXT_FILE_SIZE,
    }
    return scanned.size <= limits.get(scanned.category, 0)


def scan_folder(root: str) -> Generator[ScannedFile, None, None]:
    """Walk the folder tree and yield classified files."""
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out folders we should skip (modifying in-place prevents descent)
        dirnames[:] = [
            d for d in dirnames
            if not should_skip_folder(d)
        ]

        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            relative = os.path.relpath(filepath, root)
            _, ext = os.path.splitext(filename)

            if not ext:
                continue

            category = classify_extension(ext)
            if category == FileCategory.SKIP:
                continue

            try:
                size = os.path.getsize(filepath)
            except OSError:
                continue

            scanned = ScannedFile(
                path=filepath,
                relative_path=relative,
                category=category,
                extension=ext.lower(),
                size=size,
            )

            if is_within_size_limit(scanned):
                yield scanned


def scan_summary(root: str) -> dict:
    """Return a summary of files found by category."""
    summary = {}
    for f in scan_folder(root):
        cat = f.category.value
        if cat not in summary:
            summary[cat] = {"count": 0, "total_size": 0, "extensions": set()}
        summary[cat]["count"] += 1
        summary[cat]["total_size"] += f.size
        summary[cat]["extensions"].add(f.extension)
    # Convert sets to sorted lists for JSON serialization
    for cat in summary:
        summary[cat]["extensions"] = sorted(summary[cat]["extensions"])
    return summary
