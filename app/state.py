# app/state.py
"""
The single shared-memory contract for every node in the graph.

Key design rule: each field has ONE owner and ONE purpose.
We previously overloaded `generation` to carry grading verdicts,
which made routing logic fragile. That is fixed here.
"""
from typing import TypedDict, List, Optional


class GraphState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────────
    question: str                        # current (possibly rewritten) query
    original_question: str               # preserved for context / UI display

    # ── Retrieval ──────────────────────────────────────────────────────────────
    documents: List[str]                 # raw page-content strings from Pinecone

    # ── Grading ── (owned exclusively by grade_documents) ─────────────────────
    relevance_scores: List[str]          # one "yes"/"no" per document
    relevant_documents: List[str]        # documents that passed grading

    # ── Generation ── (owned exclusively by generate) ─────────────────────────
    generation: Optional[str]            # final answer string

    # ── Loop control ──────────────────────────────────────────────────────────
    loop_count: int
    query_history: List[str]             # track every query tried (debug / UI)