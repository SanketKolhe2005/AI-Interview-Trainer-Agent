"""
rag/vector_store.py
===================
Manages the ChromaDB persistent vector store.

Responsibilities
----------------
- ``build_store(chunks)``  : Embeds a list of Document chunks and persists them
  to the directory configured by ``settings.CHROMA_PERSIST_DIR``.
- ``load_store()``         : Opens the existing persisted ChromaDB collection
  and returns it ready for similarity search.

ChromaDB collection
-------------------
Collection name : ``settings.CHROMA_COLLECTION_NAME``
Persistence dir : ``settings.CHROMA_PERSIST_DIR``  (created automatically)

The collection stores:
  - Embedding vectors (384-dimensional, all-MiniLM-L6-v2)
  - Original chunk text (page_content)
  - Full metadata dict (source, role, category, difficulty, …)
"""

import logging
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import settings
from rag.embedder import get_embedder

logger = logging.getLogger(__name__)


def _ensure_persist_dir() -> str:
    """Create the ChromaDB persistence directory if it does not exist.

    Returns the directory path as a string (ChromaDB expects str, not Path).
    """
    persist_dir = Path(settings.CHROMA_PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return str(persist_dir)


def _delete_existing_collection(persist_dir: str) -> None:
    """Delete the existing ChromaDB collection if it exists.

    This ensures a full rebuild produces exactly *len(chunks)* vectors,
    not the old count plus the new count.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=persist_dir)
        existing = [c.name for c in client.list_collections()]
        if settings.CHROMA_COLLECTION_NAME in existing:
            client.delete_collection(settings.CHROMA_COLLECTION_NAME)
            logger.info(
                "Deleted existing collection '%s' before rebuild.",
                settings.CHROMA_COLLECTION_NAME,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not delete existing collection: %s", exc)


def build_store(chunks: List[Document]) -> Chroma:
    """Embed *chunks* and persist them into a ChromaDB collection.

    If a collection already exists at ``CHROMA_PERSIST_DIR`` it will be
    **replaced** — call this only from the ingest pipeline.

    Parameters
    ----------
    chunks : list of Document
        Pre-chunked documents from :mod:`rag.chunker`.

    Returns
    -------
    Chroma
        The newly built and persisted vector store instance.

    Raises
    ------
    ValueError
        If *chunks* is empty.
    """
    if not chunks:
        raise ValueError(
            "build_store received an empty chunk list. "
            "Run the loader and chunker before calling build_store()."
        )

    persist_dir = _ensure_persist_dir()
    embedder = get_embedder()

    # Delete any existing collection so rebuild produces a clean store
    _delete_existing_collection(persist_dir)

    logger.info(
        "Building ChromaDB collection '%s' with %d chunks -> %s",
        settings.CHROMA_COLLECTION_NAME,
        len(chunks),
        persist_dir,
    )

    # Chroma.from_documents embeds all chunks and writes to disk in one call.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        persist_directory=persist_dir,
    )

    logger.info(
        "Vector store built and persisted. Collection: '%s', Chunks: %d",
        settings.CHROMA_COLLECTION_NAME,
        len(chunks),
    )
    return vector_store


def load_store() -> Chroma:
    """Load an existing persisted ChromaDB collection from disk.

    Returns
    -------
    Chroma
        A ready-to-query vector store backed by the persisted collection.

    Raises
    ------
    FileNotFoundError
        If the persistence directory does not exist (ingest has not been run).
    RuntimeError
        If the collection appears to be empty.
    """
    persist_dir = Path(settings.CHROMA_PERSIST_DIR)
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"ChromaDB persistence directory not found: {persist_dir}. "
            "Run `python -m rag.ingest` to build the vector store first."
        )

    embedder = get_embedder()

    vector_store = Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embedder,
        persist_directory=str(persist_dir),
    )

    # Sanity check — verify the collection is not empty
    count = vector_store._collection.count()  # noqa: SLF001
    if count == 0:
        raise RuntimeError(
            f"ChromaDB collection '{settings.CHROMA_COLLECTION_NAME}' is empty. "
            "Re-run `python -m rag.ingest` to rebuild the vector store."
        )

    logger.info(
        "Loaded ChromaDB collection '%s' with %d vectors from %s",
        settings.CHROMA_COLLECTION_NAME,
        count,
        persist_dir,
    )
    return vector_store
