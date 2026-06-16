# app/config.py
"""
Centralized configuration for the Agentic RAG system.
All tuneable parameters live here — never scattered across files.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# ── Pinecone ───────────────────────────────────────────────────────────────────
PINECONE_API_KEY: str = os.environ["PINECONE_API_KEY"]
INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "agentic-rag-hf")
PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
EMBEDDING_DIMENSION: int = 384          # all-MiniLM-L6-v2 output dim

# ── Embeddings ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ── LLM (Groq) ─────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]

# Use a faster/cheaper model for classification, a smarter one for generation
GRADER_MODEL: str = "llama-3.1-8b-instant"
GENERATOR_MODEL: str = "llama-3.3-70b-versatile"   # ← upgrade: smarter answers
REWRITER_MODEL: str = "llama-3.1-8b-instant"

GRADER_TEMPERATURE: float = 0.0
GENERATOR_TEMPERATURE: float = 0.3
REWRITER_TEMPERATURE: float = 0.0

# ── Retrieval ──────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = 4               # more chunks = richer context
CHUNK_SIZE: int = 600
CHUNK_OVERLAP: int = 80

# ── Agent loop ─────────────────────────────────────────────────────────────────
MAX_LOOP_COUNT: int = 3                # hard ceiling before forced generation
MIN_RELEVANT_DOCS: int = 2            # how many docs must pass grading