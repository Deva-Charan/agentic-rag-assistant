# app/graph.py
"""
LangGraph state machine definition.

Architecture improvements over v1:
- Router inspects `relevant_documents` (not the overloaded `generation` field),
  giving clean separation of concerns between grading and generation.
- Loop-count ceiling is enforced before any routing decision.
- Graph is compiled once at module level and imported by callers.
"""
from langgraph.graph import StateGraph, END

from app.state import GraphState
from app.nodes import retrieve_data, grade_documents, generate, rewrite_query
from app.config import MAX_LOOP_COUNT, MIN_RELEVANT_DOCS

# ── Build graph ────────────────────────────────────────────────────────────────
workflow = StateGraph(GraphState)

workflow.add_node("retrieve",  retrieve_data)
workflow.add_node("grade",     grade_documents)
workflow.add_node("generate",  generate)
workflow.add_node("rewrite",   rewrite_query)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")


def route_after_grading(state: GraphState) -> str:
    """
    Routing decision after document grading.

    Logic:
      1. If the loop ceiling is reached → force generation (avoid infinite loops).
      2. If MIN_RELEVANT_DOCS or more chunks passed grading → generate.
      3. Otherwise → rewrite the query and try again.
    """
    loop_count = state.get("loop_count", 0)
    relevant_docs = state.get("relevant_documents", [])

    if loop_count >= MAX_LOOP_COUNT:
        print(f"🚨 [ROUTER] Loop limit ({MAX_LOOP_COUNT}) reached. Forcing generation.")
        return "generate"

    if len(relevant_docs) >= MIN_RELEVANT_DOCS:
        print(f"✅ [ROUTER] {len(relevant_docs)} relevant chunk(s) found. Proceeding to generation.")
        return "generate"

    print(f"⚠️  [ROUTER] Only {len(relevant_docs)} relevant chunk(s) (need {MIN_RELEVANT_DOCS}). Rewriting query.")
    return "rewrite"


workflow.add_conditional_edges(
    "grade",
    route_after_grading,
    {
        "generate": "generate",
        "rewrite":  "rewrite",
    },
)

workflow.add_edge("rewrite", "retrieve")   # retry loop
workflow.add_edge("generate", END)

# ── Compile once ───────────────────────────────────────────────────────────────
agent = workflow.compile()