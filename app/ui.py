# app/ui.py
"""
Streamlit chat interface for the Agentic RAG system.

Features:
- PDF upload in sidebar → auto-ingests immediately on upload (no button needed).
- Per-document grading breakdown in an expander.
- Query-rewrite history so users can see the agent's reasoning.
- Graceful error display.
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from app.graph import agent
from app.database import initialize_database
from app.config import (
    GENERATOR_MODEL, GRADER_MODEL, EMBEDDING_MODEL,
    RETRIEVAL_TOP_K, MAX_LOOP_COUNT, MIN_RELEVANT_DOCS,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []
if "last_uploaded_names" not in st.session_state:
    st.session_state.last_uploaded_names = []

# ── Auto-ingest helper ─────────────────────────────────────────────────────────

def _auto_ingest(uploaded_files) -> None:
    """Ingest any newly uploaded files automatically (skip already-ingested ones)."""
    new_files = [f for f in uploaded_files if f.name not in st.session_state.ingested_files]

    if not new_files:
        return

    with st.sidebar:
        progress = st.progress(0, text="Auto-ingesting…")
        for i, uploaded_file in enumerate(new_files):
            file_name = uploaded_file.name
            progress.progress(
                int((i / len(new_files)) * 90),
                text=f"📥 Ingesting: {file_name}",
            )
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf", prefix=f"{file_name}_"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                initialize_database(tmp_path, force_reingest=False)
                os.unlink(tmp_path)
                st.session_state.ingested_files.append(file_name)

            except Exception as e:
                st.error(f"❌ Failed to ingest **{file_name}**: {e}")

        progress.progress(100, text="✅ Done!")
        st.success(f"✅ {len(new_files)} file(s) ingested automatically!")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ System Config")
    st.markdown("---")

    st.subheader("📄 Upload Documents")
    st.caption("Drop PDF(s) here — they'll be ingested automatically.")

    uploaded_files = st.file_uploader(
        label="Choose PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        _auto_ingest(uploaded_files)

    if st.session_state.ingested_files:
        st.markdown("**Ingested this session:**")
        for fname in st.session_state.ingested_files:
            st.markdown(f"- 📎 `{fname}`")

    st.markdown("---")

    st.caption("**Models**")
    st.code(
        f"Generator : {GENERATOR_MODEL}\nGrader    : {GRADER_MODEL}\nEmbeddings: {EMBEDDING_MODEL}",
        language="text",
    )
    st.caption("**Retrieval**")
    st.code(
        f"Top-K         : {RETRIEVAL_TOP_K}\nMin relevant  : {MIN_RELEVANT_DOCS}\nMax loops     : {MAX_LOOP_COUNT}",
        language="text",
    )
    st.markdown("---")
    st.caption(
        "Self-correcting Agentic RAG  \n"
        "LangGraph · LLaMA-3 · HuggingFace · Pinecone"
    )

# ── Main UI ────────────────────────────────────────────────────────────────────
st.title("🤖 Self-Correcting Agentic RAG")
st.caption("Retrieves → Grades → Rewrites if needed → Generates a grounded answer.")

if not st.session_state.ingested_files:
    st.info("👈 Upload a PDF in the sidebar — it will be ingested automatically.")

st.markdown("---")

# ── Replay chat history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.write(msg["content"])
            if msg.get("metadata"):
                meta = msg["metadata"]
                with st.expander("🔍 Agent trace", expanded=False):
                    if meta.get("query_history") and len(meta["query_history"]) > 1:
                        st.markdown("**Query rewrites:**")
                        for i, q in enumerate(meta["query_history"], 1):
                            st.markdown(f"{i}. `{q}`")
                    if meta.get("relevance_scores"):
                        st.markdown("**Per-chunk grades:**")
                        cols = st.columns(len(meta["relevance_scores"]))
                        for i, (col, score) in enumerate(zip(cols, meta["relevance_scores"])):
                            icon = "✅" if score == "yes" else "❌"
                            col.metric(label=f"Chunk {i+1}", value=f"{icon} {score.upper()}")
        else:
            st.write(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if user_query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        # Inline live trace — shown as the agent runs, then replaced by the answer
        trace_placeholder = st.empty()
        answer_placeholder = st.empty()

        trace_lines = []

        initial_state = {
            "question": user_query,
            "original_question": user_query,
            "documents": [],
            "relevance_scores": [],
            "relevant_documents": [],
            "generation": None,
            "loop_count": 0,
            "query_history": [user_query],
        }

        accumulated_state: dict = {}
        final_response = "I was unable to produce an answer. Please try rephrasing."

        try:
            for event in agent.stream(initial_state):
                for node_name, delta in event.items():
                    accumulated_state.update(delta)

                    if node_name == "retrieve":
                        n = len(delta.get("documents", []))
                        trace_lines.append(f"🔍 Found {n} candidate chunk(s).")

                    elif node_name == "grade":
                        scores = delta.get("relevance_scores", [])
                        passed = delta.get("relevant_documents", [])
                        icon_row = " ".join("✅" if s == "yes" else "❌" for s in scores)
                        trace_lines.append(f"⚖️ {icon_row} — {len(passed)}/{len(scores)} relevant.")

                    elif node_name == "rewrite":
                        new_q = delta.get("question", "")
                        loop = delta.get("loop_count", "?")
                        trace_lines.append(f"🔄 Rewrite #{loop}: `{new_q}`")

                    elif node_name == "generate":
                        final_response = delta.get("generation", final_response)
                        trace_lines.append("✍️ Generating answer…")

                    # Update live trace while running
                    trace_placeholder.markdown("\n\n".join(trace_lines))

            # Clear trace, show final answer inline
            trace_placeholder.empty()
            answer_placeholder.write(final_response)

        except Exception as e:
            trace_placeholder.empty()
            answer_placeholder.error(f"An error occurred: {e}")
            final_response = f"Error: {e}"

        metadata = {
            "query_history": accumulated_state.get("query_history", [user_query]),
            "relevance_scores": accumulated_state.get("relevance_scores", []),
        }

        # Agent trace expander (collapsed by default, always visible after answer)
        with st.expander("🔍 Agent trace", expanded=False):
            qh = metadata.get("query_history", [])
            if len(qh) > 1:
                st.markdown("**Query rewrites:**")
                for i, q in enumerate(qh, 1):
                    st.markdown(f"{i}. `{q}`")
            rs = metadata.get("relevance_scores", [])
            if rs:
                st.markdown("**Per-chunk grades:**")
                cols = st.columns(len(rs))
                for i, (col, score) in enumerate(zip(cols, rs)):
                    icon = "✅" if score == "yes" else "❌"
                    col.metric(label=f"Chunk {i+1}", value=f"{icon} {score.upper()}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": final_response,
            "metadata": metadata,
        })