"""
rag/chunker.py
==============
Splits LangChain Documents into smaller chunks suitable for embedding and
retrieval.  Uses LangChain's RecursiveCharacterTextSplitter which tries to
split on paragraph, sentence, and word boundaries in that order, producing
semantically coherent chunks while keeping size predictable.

JSON entry Documents (which are already short — one question per Document) are
typically smaller than the chunk size and pass through unchanged.  Long TXT
files (role descriptions, interview guide) are split into overlapping chunks.
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> List[Document]:
    """Split documents into chunks using RecursiveCharacterTextSplitter.

    Document metadata is preserved on every chunk so that downstream filters
    (e.g. role-based retrieval) continue to work after splitting.

    Short documents (already smaller than *chunk_size*) are returned as-is.

    Parameters
    ----------
    documents : list of Document
        Documents produced by :mod:`rag.loader`.
    chunk_size : int
        Maximum character length of each chunk.  Defaults to
        ``settings.CHUNK_SIZE`` (500).
    chunk_overlap : int
        Number of characters shared between consecutive chunks.  Defaults to
        ``settings.CHUNK_OVERLAP`` (50).

    Returns
    -------
    list of Document
        Flat list of all chunks, each carrying the parent document's metadata.
    """
    if not documents:
        logger.warning("chunk_documents received an empty document list.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Split on paragraph breaks first, then sentences, then words
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks: List[Document] = []
    short_count = 0

    for doc in documents:
        content_len = len(doc.page_content)

        if content_len <= chunk_size:
            # Document already fits in a single chunk — pass through unchanged
            chunks.append(doc)
            short_count += 1
            continue

        split_docs = splitter.split_documents([doc])
        # Ensure metadata is faithfully propagated (split_documents should do
        # this, but be explicit to guard against future API changes)
        for chunk in split_docs:
            chunk.metadata.update(doc.metadata)
        chunks.extend(split_docs)

    logger.info(
        "Chunking complete: %d documents → %d chunks "
        "(%d passed through unchanged, chunk_size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        short_count,
        chunk_size,
        chunk_overlap,
    )
    return chunks
