"""
rag/embedder.py
===============
Wraps the sentence-transformers model in a LangChain-compatible Embeddings
class so that it can be used directly with ChromaDB and other LangChain
vector-store integrations.

The model runs **entirely locally** — no external API calls, no IBM Cloud
quota consumed.  The first call downloads the model weights (~90 MB) from
HuggingFace and caches them in the default HuggingFace cache directory
(~/.cache/huggingface/).

Model: sentence-transformers/all-MiniLM-L6-v2
  - Embedding dimension : 384
  - Max token length    : 256 tokens
  - Speed               : ~14 000 sentences/sec on CPU
"""

import logging
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings

logger = logging.getLogger(__name__)

# Module-level singleton so the model is loaded only once per process
_embedder_instance: HuggingFaceEmbeddings | None = None


def get_embedder() -> HuggingFaceEmbeddings:
    """Return the singleton HuggingFaceEmbeddings instance.

    The model is lazy-loaded on first call and reused on all subsequent calls.
    Thread safety is not guaranteed — call this from the main thread before
    spawning workers if multithreading is needed.

    Returns
    -------
    HuggingFaceEmbeddings
        A LangChain-compatible embeddings object backed by the configured
        sentence-transformers model.
    """
    global _embedder_instance  # noqa: PLW0603

    if _embedder_instance is None:
        model_name = settings.EMBEDDING_MODEL
        logger.info("Loading embedding model: %s", model_name)
        _embedder_instance = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully.")

    return _embedder_instance


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of plain text strings.

    Convenience wrapper around :func:`get_embedder` for callers that just
    want raw vectors without LangChain boilerplate.

    Parameters
    ----------
    texts : list of str

    Returns
    -------
    list of list of float
        One embedding vector per input text.
    """
    if not texts:
        return []
    embedder = get_embedder()
    return embedder.embed_documents(texts)


def embed_query(query: str) -> List[float]:
    """Embed a single query string.

    Parameters
    ----------
    query : str

    Returns
    -------
    list of float
        The embedding vector for the query.
    """
    embedder = get_embedder()
    return embedder.embed_query(query)
