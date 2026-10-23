"""Streamlit UI for the Finance RAG Assistant."""

import streamlit as st
from dotenv import load_dotenv

from logger import logger
from rag import FinanceRAG, answer_with_llm

load_dotenv(override=False)

st.set_page_config(
    page_title="Finance RAG Assistant",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Finance RAG Assistant")
st.caption(
    "Ask questions about your financial reports using document-grounded answers."
)

# Session state
if "finance_rag" not in st.session_state:
    st.session_state.finance_rag = FinanceRAG(use_reranker=False)

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "messages" not in st.session_state:
    st.session_state.messages = []

engine = st.session_state.finance_rag

# Sidebar
with st.sidebar:
    st.header("Document settings")

    uploaded_files = st.file_uploader(
        "Upload financial documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="PDF and TXT files are supported.",
    )

    # Keep disabled on low-memory hosting.
    use_reranker = st.checkbox(
        "Use cross-encoder reranking",
        value=False,
        help=(
            "Uses additional memory. On a 512 MB instance, "
            "keep this disabled."
        ),
    )

    top_k = st.slider(
        "Passages to retrieve",
        min_value=1,
        max_value=5,
        value=3,
    )

    answer_mode = st.radio(
        "Answer mode",
        options=["Groq LLM", "Extractive (no API)"],
        index=0,
        help=(
            "Groq generates an answer from retrieved passages. "
            "Extractive mode returns retrieved text without an LLM API call."
        ),
    )

    st.caption("Tip: Start with one small PDF while testing deployment.")

    if st.button("Index documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF or TXT file first.")
        else:
            max_file_size = 10 * 1024 * 1024
            oversized = [
                file.name for file in uploaded_files
                if file.size > max_file_size
            ]

            if oversized:
                st.error(
                    "These files exceed the 10 MB limit: "
                    + ", ".join(oversized)
                )
                logger.warning("Upload rejected due to size limit: %s", oversized)
            else:
                try:
                    # Reranking is disabled by default for low-memory hosting.
                    engine.use_reranker = bool(use_reranker)

                    with st.spinner("Reading and indexing documents..."):
                        passage_count = engine.ingest(uploaded_files)

                    st.session_state.indexed = True
                    st.session_state.indexed_files = [
                        file.name for file in uploaded_files
                    ]
                    st.session_state.messages = []

                    st.success(
                        f"Indexed {len(uploaded_files)} file(s) "
                        f"into {passage_count} passages."
                    )
                except Exception as exc:
                    logger.exception("Document indexing failed in Streamlit UI")
                    st.error(
                        f"Indexing failed ({type(exc).__name__}). "
                        "Check application logs for details."
                    )

    if st.session_state.indexed:
        st.success("Documents are indexed.")
        with st.expander("Indexed files"):
            for filename in st.session_state.indexed_files:
                st.write(f"📄 {filename}")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Privacy: Uploaded documents are processed by this app. "
        "With Groq selected, retrieved excerpts are sent to the Groq API."
    )

# Previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources", [])

        if message["role"] == "assistant" and sources:
            with st.expander("Sources and retrieved passages"):
                for i, source in enumerate(sources, start=1):
                    page = source.get("page")
                    page_text = f", page {page}" if page is not None else ""
                    st.markdown(
                        f"**{i}. {source.get('source', 'Unknown')}{page_text}**"
                    )
                    st.write(source.get("text", ""))
                    if i < len(sources):
                        st.divider()

# Chat
question = st.chat_input(
    "Ask about revenue, expenses, profit, ratios, or financial risks..."
)

if question:
    if not st.session_state.indexed or not engine.chunks:
        st.warning("Upload and index documents before asking a question.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                engine.use_reranker = bool(use_reranker)

                with st.spinner("Searching financial documents..."):
                    contexts = engine.retrieve(question, top_k=top_k)

                with st.spinner("Preparing your answer..."):
                    use_llm = answer_mode == "Groq LLM"
                    answer = answer_with_llm(
                        question,
                        contexts,
                        use_llm=use_llm,
                        fallback_on_error=True,
                    )

                st.markdown(answer)

                if contexts:
                    with st.expander("Sources and retrieved passages"):
                        for i, source in enumerate(contexts, start=1):
                            page = source.get("page")
                            page_text = f", page {page}" if page is not None else ""
                            st.markdown(
                                f"**{i}. {source.get('source', 'Unknown')}"
                                f"{page_text}**"
                            )
                            st.write(source.get("text", ""))
                            if i < len(contexts):
                                st.divider()
                else:
                    st.info("No relevant passages were retrieved.")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": contexts,
                    }
                )

            except Exception as exc:
                logger.exception("Chat request failed in Streamlit UI")
                error_message = (
                    f"Something went wrong ({type(exc).__name__}). "
                    "Please try again. Check application logs if the issue persists."
                )
                st.error(error_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )
