"""
rag/ingest.py
=============
One-time (or on-demand) ingestion pipeline that builds the ChromaDB vector
store from the knowledge base documents.

Pipeline
--------
  load_all_documents()
       ↓
  chunk_documents()
       ↓
  build_store()   [embeds + persists to CHROMA_PERSIST_DIR]

Usage
-----
Run from the project root:

    python -m rag.ingest

Options
-------
  --rebuild   : Force a full rebuild even if the vector store already exists.
                Without this flag the script exits early if the store is present.
  --verbose   : Enable DEBUG-level logging.

Exit codes
----------
  0  : Success
  1  : Failure (details logged to stderr)
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as __main__
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import settings  # noqa: E402  (after sys.path patch)
from rag.loader import load_all_documents  # noqa: E402
from rag.chunker import chunk_documents  # noqa: E402
from rag.vector_store import build_store  # noqa: E402


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _store_exists() -> bool:
    """Return True if a non-empty ChromaDB persistence directory already exists."""
    persist_dir = Path(settings.CHROMA_PERSIST_DIR)
    if not persist_dir.exists():
        return False
    # ChromaDB creates at minimum a chroma.sqlite3 file on first ingest
    return any(persist_dir.iterdir())


def run_ingest(rebuild: bool = False) -> None:
    """Execute the full load → chunk → embed → persist pipeline.

    Parameters
    ----------
    rebuild : bool
        When ``True``, always rebuild the store even if it already exists.
        When ``False`` (default), skip if the store is already present.
    """
    logger = logging.getLogger(__name__)

    if _store_exists() and not rebuild:
        logger.info(
            "Vector store already exists at '%s'. "
            "Pass --rebuild to force a rebuild.",
            settings.CHROMA_PERSIST_DIR,
        )
        return

    if _store_exists() and rebuild:
        logger.info("--rebuild flag set — rebuilding vector store.")

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading knowledge base documents")
    logger.info("=" * 60)
    documents = load_all_documents()

    if not documents:
        logger.error("No documents loaded — aborting. Check knowledge_base/ directory.")
        sys.exit(1)

    logger.info("Loaded %d documents total.", len(documents))

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(
        "STEP 2: Chunking documents (chunk_size=%d, overlap=%d)",
        settings.CHUNK_SIZE,
        settings.CHUNK_OVERLAP,
    )
    logger.info("=" * 60)
    chunks = chunk_documents(documents)
    logger.info("Created %d chunks from %d documents.", len(chunks), len(documents))

    # ── Step 3: Embed & Persist ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Embedding and persisting to ChromaDB")
    logger.info("  Model   : %s", settings.EMBEDDING_MODEL)
    logger.info("  Store   : %s", settings.CHROMA_PERSIST_DIR)
    logger.info("  Collection: %s", settings.CHROMA_COLLECTION_NAME)
    logger.info("=" * 60)

    build_store(chunks)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("INGEST COMPLETE")
    logger.info("  Documents : %d", len(documents))
    logger.info("  Chunks    : %d", len(chunks))
    logger.info("  Location  : %s", Path(settings.CHROMA_PERSIST_DIR).resolve())
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the ChromaDB vector store from the knowledge base."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a full rebuild even if the vector store already exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    _configure_logging(verbose=args.verbose)

    try:
        run_ingest(rebuild=args.rebuild)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).error("Ingest failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
