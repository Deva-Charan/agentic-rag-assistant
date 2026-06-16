# app/main.py

import sys
from dotenv import load_dotenv

load_dotenv()

from app.graph import agent  # noqa: E402  (import after load_dotenv)


DIVIDER = "=" * 60


def run(question: str) -> str:
    """
    Execute the agentic RAG pipeline and return the final answer.

    Args:
        question: Natural language question to answer.

    Returns:
        The generated answer string.
    """
    initial_state = {
        "question": question,
        "original_question": question,
        "documents": [],
        "relevance_scores": [],
        "relevant_documents": [],
        "generation": None,
        "loop_count": 0,
        "query_history": [question],
    }

    print(f"\n{DIVIDER}")
    print("🤖  Agentic RAG Pipeline — Starting")
    print(f"   Question: {question}")
    print(DIVIDER)

    final_state: dict = {}
    for event in agent.stream(initial_state):
        for node_name, delta in event.items():
            print(f"\n   ✓ Node completed: [{node_name}]  keys={list(delta.keys())}")
            final_state.update(delta)

    answer = final_state.get("generation") or "No answer was produced."

    print(f"\n{DIVIDER}")
    print("🏁  FINAL ANSWER")
    print(DIVIDER)
    print(answer)
    print(DIVIDER)

    if final_state.get("query_history") and len(final_state["query_history"]) > 1:
        print("\n📝  Query evolution:")
        for i, q in enumerate(final_state["query_history"], 1):
            print(f"   {i}. {q}")

    return answer


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the core stages of the NASA product life cycle?"
    run(q)