"""
rag/retriever.py
================
Semantic similarity retrieval over the ChromaDB knowledge base.

Primary entry point: :func:`retrieve`
--------------------------------------
Given a natural-language *query* (e.g. "technical questions for a senior
software engineer on system design"), the retriever:

1. Embeds the query with the same all-MiniLM-L6-v2 model used during ingest.
2. Performs a cosine-similarity search in ChromaDB.
3. Optionally filters results to a specific *role* using ChromaDB metadata
   filtering so that, for example, a Data Scientist profile does not receive
   DevOps questions.
4. Returns the top-*k* most relevant Document chunks.

Metadata filtering
------------------
The ``role`` metadata field is stored on every chunk during ingest.
The filter uses ChromaDB's ``$in`` operator to match:
  - The requested role (e.g. "Software Engineer")
  - "general"  — role-agnostic docs (HR questions, behavioral questions,
    interview guide) that are always relevant regardless of role.
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document

from config.settings import settings
from rag.vector_store import load_store

logger = logging.getLogger(__name__)

# Module-level cache so the vector store is opened only once per process
_store_cache = None


def _get_store():
    """Return the cached vector store, loading it from disk if necessary."""
    global _store_cache  # noqa: PLW0603
    if _store_cache is None:
        _store_cache = load_store()
    return _store_cache


def retrieve(
    query: str,
    role_filter: Optional[str] = None,
    k: int = settings.TOP_K_RETRIEVAL,
) -> List[Document]:
    """Retrieve the top-*k* most relevant documents for *query*.

    Parameters
    ----------
    query : str
        The natural-language search query.  Typically constructed by the agent
        from the user's role, experience level, and desired question type.
    role_filter : str or None
        When provided, limits results to documents whose ``role`` metadata
        equals *role_filter* **or** ``"general"``.  Pass the role exactly as
        it appears in ``settings.SUPPORTED_ROLES``, e.g. ``"Software Engineer"``.
        When ``None``, all documents in the collection are searched.
    k : int
        Maximum number of documents to return.  Defaults to
        ``settings.TOP_K_RETRIEVAL``.

    Returns
    -------
    list of Document
        Retrieved chunks ordered by descending similarity score, each carrying
        its original metadata (source, role, category, difficulty, …).

    Raises
    ------
    RuntimeError
        Propagated from :func:`rag.vector_store.load_store` if the vector
        store has not been built yet.
    """
    if not query or not query.strip():
        logger.warning("retrieve() called with an empty query — returning []")
        return []

    store = _get_store()

    # Build metadata filter
    where_filter: Optional[dict] = None
    if role_filter:
        # Match documents for the requested role OR general/role-agnostic docs
        where_filter = {
            "role": {"$in": [role_filter, "general"]}
        }
        logger.debug("Applying role filter: %s + general", role_filter)

    try:
        if where_filter:
            results = store.similarity_search(
                query=query,
                k=k,
                filter=where_filter,
            )
        else:
            results = store.similarity_search(query=query, k=k)
    except Exception as exc:  # noqa: BLE001
        logger.error("ChromaDB similarity_search failed: %s", exc)
        raise

    logger.debug(
        "retrieve('%s'…, role=%s, k=%d) → %d results",
        query[:60],
        role_filter,
        k,
        len(results),
    )
    return results


def retrieve_with_scores(
    query: str,
    role_filter: Optional[str] = None,
    k: int = settings.TOP_K_RETRIEVAL,
) -> List[tuple[Document, float]]:
    """Like :func:`retrieve` but also returns the similarity score for each result.

    Returns
    -------
    list of (Document, float)
        Pairs of (document, cosine_similarity_score) ordered by descending score.
        Scores are in [0, 1] — higher is more similar.
    """
    if not query or not query.strip():
        return []

    store = _get_store()
    where_filter: Optional[dict] = None
    if role_filter:
        where_filter = {"role": {"$in": [role_filter, "general"]}}

    try:
        if where_filter:
            results = store.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=where_filter,
            )
        else:
            results = store.similarity_search_with_relevance_scores(query=query, k=k)
    except Exception as exc:  # noqa: BLE001
        logger.error("ChromaDB similarity_search_with_relevance_scores failed: %s", exc)
        raise

    return results


def invalidate_store_cache() -> None:
    """Force the next :func:`retrieve` call to reload the vector store from disk.

    Useful after rebuilding the vector store with ``python -m rag.ingest``.
    """
    global _store_cache  # noqa: PLW0603
    _store_cache = None
    logger.info("Vector store cache invalidated.")
