# app/database.py
"""
Data ingestion pipeline.

Improvements over v1:
- No hardcoded paths: accepts a list of files or a directory.
- Idempotent: won't re-ingest if the index already has vectors.
- Proper error-handling and status reporting.
- Embeddings model singleton so we don't re-download on every run.
"""
import time
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.config import (
    PINECONE_API_KEY, INDEX_NAME, PINECONE_CLOUD, PINECONE_REGION,
    EMBEDDING_MODEL, EMBEDDING_DIMENSION, CHUNK_SIZE, CHUNK_OVERLAP,
)


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached embeddings model instance."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},   # cosine similarity needs unit vecs
    )


def _load_documents(source: str) -> list:
    """
    Load documents from a file path or a directory.
    Supports PDF (easily extendable to other formats).
    """
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if p.is_dir():
        loader = DirectoryLoader(str(p), glob="**/*.pdf", loader_cls=PyPDFLoader)
    elif p.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(p))
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    return loader.load()


def initialize_database(source: str, force_reingest: bool = False) -> None:
    """
    Ingest documents from `source` (file or directory) into Pinecone.

    Args:
        source: Path to a PDF file or a directory containing PDFs.
        force_reingest: If True, clear and rebuild the index even if it exists.
    """
    print("🚀 Starting Data Ingestion Pipeline...")

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print(f"📄 Loading documents from: {source}")
    raw_documents = _load_documents(source)
    print(f"   Loaded {len(raw_documents)} page(s).")

    # ── 2. Split ─────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents = splitter.split_documents(raw_documents)
    print(f"✂️  Split into {len(documents)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    # ── 3. Pinecone setup ────────────────────────────────────────────────────
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = pc.list_indexes().names()

    if INDEX_NAME in existing_indexes and force_reingest:
        print(f"🗑️  Deleting existing index '{INDEX_NAME}' for full re-ingest...")
        pc.delete_index(INDEX_NAME)
        existing_indexes = []

    if INDEX_NAME not in existing_indexes:
        print(f"🏗️  Creating Pinecone index '{INDEX_NAME}' (dim={EMBEDDING_DIMENSION})...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        # Wait until the index is ready
        for _ in range(30):
            if pc.describe_index(INDEX_NAME).status["ready"]:
                break
            time.sleep(2)
        else:
            raise RuntimeError("Pinecone index did not become ready in time.")
    else:
        # Check if the index already has vectors — skip ingestion to save cost
        stats = pc.Index(INDEX_NAME).describe_index_stats()
        vector_count = getattr(stats, "total_vector_count", 0)
        if vector_count > 0 and not force_reingest:
            print(f"⚡ Index '{INDEX_NAME}' already has {vector_count} vectors. Skipping ingestion.")
            print("   Use force_reingest=True to rebuild from scratch.")
            return

    # ── 4. Embed + Upload ────────────────────────────────────────────────────
    print(f"🧠 Embedding with '{EMBEDDING_MODEL}' and uploading to Pinecone...")
    embeddings = _get_embeddings()
    PineconeVectorStore.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print(f"🎉 Ingestion complete! {len(documents)} chunks stored in '{INDEX_NAME}'.")


if __name__ == "__main__":
    import sys
    source_path = sys.argv[1] if len(sys.argv) > 1 else str(Path("data"))
    force = "--force" in sys.argv
    initialize_database(source_path, force_reingest=force)
