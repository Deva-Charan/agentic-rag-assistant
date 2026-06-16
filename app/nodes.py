# app/nodes.py
"""
LangGraph node functions.

Key improvements over v1:
- Embeddings model is a module-level singleton → loaded once, reused every call.
- grade_documents grades ALL retrieved docs (not just the first) and writes to
  `relevance_scores` + `relevant_documents` — not the overloaded `generation` field.
- generate() falls back gracefully when no relevant docs passed grading.
- Every node returns only the keys it owns (clean state delta semantics).
- Prompts are more structured and instruction-tuned for LLaMA-3.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_pinecone import PineconeVectorStore

from app.state import GraphState
from app.config import (
    EMBEDDING_MODEL, INDEX_NAME,
    GRADER_MODEL, GENERATOR_MODEL, REWRITER_MODEL,
    GRADER_TEMPERATURE, GENERATOR_TEMPERATURE, REWRITER_TEMPERATURE,
    RETRIEVAL_TOP_K, MIN_RELEVANT_DOCS,
    GROQ_API_KEY,
)

# ── Singletons (created once at import time) ───────────────────────────────────

@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _get_vectorstore() -> PineconeVectorStore:
    return PineconeVectorStore(index_name=INDEX_NAME, embedding=_get_embeddings())


def _llm(model: str, temperature: float) -> ChatGroq:
    return ChatGroq(model=model, temperature=temperature, api_key=GROQ_API_KEY)


# ── Node 1: Retrieve ───────────────────────────────────────────────────────────

def retrieve_data(state: GraphState) -> Dict[str, Any]:
    """Vector-search Pinecone for the top-K most relevant chunks."""
    print("\n🔍 [RETRIEVE] Searching knowledge base...")
    question = state["question"]

    vectorstore = _get_vectorstore()
    docs = vectorstore.similarity_search(question, k=RETRIEVAL_TOP_K)
    document_texts = [doc.page_content for doc in docs]

    print(f"   Retrieved {len(document_texts)} chunks.")
    return {"documents": document_texts}


# ── Node 2: Grade ──────────────────────────────────────────────────────────────

_GRADER_PROMPT = PromptTemplate(
    template="""<|system|>
You are a strict binary relevance classifier for a RAG pipeline.
Your only job: decide if the retrieved chunk contains information that could help answer the question.

Rules:
- Reply ONLY with valid JSON. No prose, no markdown fences.
- The JSON must have exactly one key: "relevance_score" with value "yes" or "no".
- "yes"  → the chunk is on-topic and semantically useful.
- "no"   → the chunk is off-topic, too generic, or clearly unrelated.

<|user|>
Question: {question}

Retrieved chunk:
\"\"\"
{document}
\"\"\"

<|assistant|>""",
    input_variables=["question", "document"],
)


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """Grade every retrieved document and partition them into relevant / irrelevant."""
    print("\n⚖️  [GRADE] Evaluating document relevance...")
    question = state["question"]
    documents = state["documents"]

    grader = _GRADER_PROMPT | _llm(GRADER_MODEL, GRADER_TEMPERATURE) | JsonOutputParser()

    relevance_scores: list[str] = []
    relevant_documents: list[str] = []

    for i, doc in enumerate(documents):
        try:
            result = grader.invoke({"document": doc, "question": question})
            score = result.get("relevance_score", "no").lower()
        except Exception as e:
            print(f"   ⚠️  Grader error on chunk {i}: {e}")
            score = "no"

        relevance_scores.append(score)
        if score == "yes":
            relevant_documents.append(doc)
        print(f"   Chunk {i + 1}/{len(documents)}: {score.upper()}")

    print(f"   → {len(relevant_documents)} / {len(documents)} chunks passed.")
    return {
        "relevance_scores": relevance_scores,
        "relevant_documents": relevant_documents,
    }


# ── Node 3: Generate ───────────────────────────────────────────────────────────

_GENERATOR_PROMPT = PromptTemplate(
    template="""<|system|>
You are an expert research assistant with deep knowledge of technical documentation.
Your answers are grounded strictly in the provided context.
Do NOT fabricate information; if the context is insufficient, say so honestly.

<|user|>
Context retrieved from the knowledge base:
\"\"\"
{context}
\"\"\"

Original user question: {original_question}
Refined search query used: {question}

Provide a thorough, well-structured answer. Use bullet points or numbered lists where appropriate.

<|assistant|>""",
    input_variables=["context", "question", "original_question"],
)


def generate(state: GraphState) -> Dict[str, Any]:
    """Synthesize a grounded answer from the relevant document chunks."""
    print("\n✍️  [GENERATE] Synthesizing final answer...")

    question = state["question"]
    original_question = state.get("original_question", question)
    relevant_docs = state.get("relevant_documents") or state.get("documents", [])

    if not relevant_docs:
        return {
            "generation": (
                "I was unable to find sufficiently relevant information in the knowledge base "
                "to answer your question confidently. Please try rephrasing or check that the "
                "relevant documents have been ingested."
            )
        }

    context_str = "\n\n---\n\n".join(relevant_docs)
    generator = _GENERATOR_PROMPT | _llm(GENERATOR_MODEL, GENERATOR_TEMPERATURE) | StrOutputParser()

    answer = generator.invoke({
        "context": context_str,
        "question": question,
        "original_question": original_question,
    })

    print("   Answer generated successfully.")
    return {"generation": answer}


# ── Node 4: Rewrite Query ──────────────────────────────────────────────────────

_REWRITER_PROMPT = PromptTemplate(
    template="""<|system|>
You are an expert at crafting search queries for dense-vector semantic retrieval systems.
The previous query did not surface relevant documents. Rewrite it to improve recall:
- Extract the core technical concepts and domain-specific terminology.
- Expand acronyms if you know them.
- Remove filler words that don't carry semantic weight.
- Keep the rewritten query concise (≤ 20 words).
Output ONLY the rewritten query — no explanation, no preamble.

<|user|>
Previous query: {question}

<|assistant|>""",
    input_variables=["question"],
)


def rewrite_query(state: GraphState) -> Dict[str, Any]:
    """Rewrite the search query to improve vector-retrieval recall."""
    print("\n🔄 [REWRITE] Optimizing search query...")

    question = state["question"]
    loop_count = state.get("loop_count", 0)
    query_history = state.get("query_history", [question])

    rewriter = _REWRITER_PROMPT | _llm(REWRITER_MODEL, REWRITER_TEMPERATURE) | StrOutputParser()
    optimized_query = rewriter.invoke({"question": question}).strip()

    print(f"   Old → '{question}'")
    print(f"   New → '{optimized_query}'")

    return {
        "question": optimized_query,
        "loop_count": loop_count + 1,
        "query_history": query_history + [optimized_query],
    }