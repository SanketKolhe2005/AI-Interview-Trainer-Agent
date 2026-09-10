"""
rag/loader.py
=============
Loads all knowledge-base documents (JSON and TXT) into LangChain Document objects
with rich metadata that the retriever can filter on later.

Document metadata schema
------------------------
Every Document carries:
  - source      : relative file path (str)
  - file_type   : "json" | "txt"
  - category    : broad knowledge category (e.g. "technical_questions", "hr_questions")
  - role        : job role slug or "general" for role-agnostic documents
  - sub_type    : for JSON records, the question type or competency; empty for TXT
"""

import json
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _role_slug_from_filename(filename: str) -> str:
    """Convert a filename stem to a human-readable role slug.

    Examples
    --------
    "software_engineer" -> "Software Engineer"
    "data_scientist"    -> "Data Scientist"
    """
    return filename.replace("_", " ").title()


def _doc_from_json_entry(entry: dict, source: str, category: str, role: str) -> Document:
    """Build a single LangChain Document from one JSON entry dict.

    The page_content is assembled from all string values in the entry so that
    every field is searchable via embedding.
    """
    # Build searchable text from all string fields in the entry
    parts: List[str] = []
    for key, value in entry.items():
        if isinstance(value, str) and value.strip():
            parts.append(f"{key.replace('_', ' ').title()}: {value.strip()}")

    page_content = "\n".join(parts)

    metadata = {
        "source": source,
        "file_type": "json",
        "category": category,
        "role": role,
        "sub_type": entry.get("category", entry.get("competency", entry.get("topic", ""))),
        "difficulty": entry.get("difficulty", ""),
    }
    return Document(page_content=page_content, metadata=metadata)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_json_documents(directory: Path) -> List[Document]:
    """Load all *.json files under *directory* into LangChain Documents.

    Each JSON file must be a JSON array of objects. Every array element
    becomes its own Document so retrieval is at the question/entry level.

    Parameters
    ----------
    directory : Path
        Directory to search (non-recursive).

    Returns
    -------
    list of Document
    """
    documents: List[Document] = []
    json_files = sorted(directory.glob("*.json"))

    if not json_files:
        logger.warning("No JSON files found in %s", directory)
        return documents

    for filepath in json_files:
        role_raw = filepath.stem  # e.g. "software_engineer"
        role = _role_slug_from_filename(role_raw) if role_raw not in (
            "hr_questions", "behavioral_questions"
        ) else "general"

        # Derive category from the parent directory name
        category = filepath.parent.name  # e.g. "technical_questions"

        try:
            raw = json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load %s: %s", filepath, exc)
            continue

        if not isinstance(raw, list):
            logger.warning("Expected a JSON array in %s — skipping", filepath)
            continue

        source = str(filepath.relative_to(settings.KNOWLEDGE_BASE_DIR.parent))
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            doc = _doc_from_json_entry(entry, source, category, role)
            documents.append(doc)

        logger.debug("Loaded %d entries from %s", len(raw), filepath.name)

    return documents


def load_text_documents(directory: Path) -> List[Document]:
    """Load all *.txt files under *directory* as single LangChain Documents.

    Each file is loaded as one Document (chunking happens downstream in
    rag/chunker.py).

    Parameters
    ----------
    directory : Path
        Directory to search (non-recursive).

    Returns
    -------
    list of Document
    """
    documents: List[Document] = []
    txt_files = sorted(directory.glob("*.txt"))

    if not txt_files:
        logger.warning("No TXT files found in %s", directory)
        return documents

    for filepath in txt_files:
        role_raw = filepath.stem
        role = _role_slug_from_filename(role_raw) if directory.name == "role_descriptions" \
            else "general"
        category = directory.name

        try:
            content = filepath.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Failed to read %s: %s", filepath, exc)
            continue

        if not content.strip():
            logger.warning("Empty file — skipping: %s", filepath)
            continue

        source = str(filepath.relative_to(settings.KNOWLEDGE_BASE_DIR.parent))
        doc = Document(
            page_content=content,
            metadata={
                "source": source,
                "file_type": "txt",
                "category": category,
                "role": role,
                "sub_type": "",
                "difficulty": "",
            },
        )
        documents.append(doc)
        logger.debug("Loaded %s (%d chars)", filepath.name, len(content))

    return documents


def load_all_documents() -> List[Document]:
    """Load the entire knowledge base into a flat list of Documents.

    Traverses all sub-directories of ``settings.KNOWLEDGE_BASE_DIR`` and
    delegates to :func:`load_json_documents` or :func:`load_text_documents`
    based on the file types present.

    Returns
    -------
    list of Document
        All documents from every knowledge-base sub-directory.
    """
    kb = settings.KNOWLEDGE_BASE_DIR
    all_docs: List[Document] = []

    subdirs = [
        (kb / "technical_questions", "json"),
        (kb / "hr_questions", "json"),
        (kb / "behavioral_questions", "json"),
        (kb / "role_descriptions", "txt"),
        (kb / "industry_guidelines", "txt"),
    ]

    for dirpath, ftype in subdirs:
        if not dirpath.exists():
            logger.warning("Knowledge-base directory not found: %s", dirpath)
            continue
        if ftype == "json":
            docs = load_json_documents(dirpath)
        else:
            docs = load_text_documents(dirpath)
        logger.info("  %s: loaded %d documents", dirpath.name, len(docs))
        all_docs.extend(docs)

    logger.info("Total documents loaded: %d", len(all_docs))
    return all_docs
