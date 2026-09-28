
import streamlit as st
from dotenv import load_dotenv

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

# Keep the engine and chat history between Streamlit reruns.
if "finance_rag" not in st.session_state:
    st.session_state.finance_rag = FinanceRAG(use_reranker=False)

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "messages" not in st.session_state:
    st.session_state.messages = []

engine = st.session_state.finance_rag


# Sidebar settings.
with st.sidebar:
    st.header("Document settings")

    uploaded_files = st.file_uploader(
        "Upload financial documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="PDF and TXT files are supported.",
    )

    use_reranker = st.checkbox(
        "Use cross-encoder reranking",
        value=engine.use_reranker,
        help=(
            "May improve relevance, but loads an additional model "
            "and uses more memory."
        ),
    )

    top_k = st.slider(
        "Passages to retrieve",
        min_value=1,
        max_value=10,
        value=4,
    )

    answer_mode = st.radio(
        "Answer mode",
        options=[
            "Groq LLM",
            "Extractive (no API)",
        ],
        index=0,
        help=(
            "Groq LLM generates a response from retrieved passages. "
            "Extractive mode displays the passages directly and makes "
            "no LLM API request."
        ),
    )

    if st.button("Index documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF or TXT file first.")
        else:
            try:
                # Apply the current reranker setting.
                engine.use_reranker = use_reranker

                with st.spinner("Reading, chunking, and indexing documents..."):
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
                st.error(
                    f"Indexing failed: {type(exc).__name__}: {exc}"
                )

    if st.session_state.indexed:
        st.success("Documents are indexed.")

        with st.expander("Indexed files"):
            for filename in st.session_state.get("indexed_files", []):
                st.write(f"📄 {filename}")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Display previous conversation.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources and retrieved passages"):
                for i, source in enumerate(message["sources"], start=1):
                    page = source.get("page")
                    page_text = f", page {page}" if page is not None else ""

                    st.markdown(
                        f"**{i}. {source.get('source', 'Unknown')}"
                        f"{page_text}**"
                    )

                    st.caption(
                        "Hybrid score: "
                        f"{source.get('score', 0.0):.3f} · "
                        "Semantic score: "
                        f"{source.get('semantic_score', 0.0):.3f} · "
                        "BM25 score: "
                        f"{source.get('keyword_score', 0.0):.3f}"
                    )

                    st.write(source.get("text", ""))
                    if i < len(message["sources"]):
                        st.divider()


# Chat input.
question = st.chat_input(
    "Ask about revenue, expenses, profit, ratios, or financial risks..."
)

if question:
    if not st.session_state.indexed or not engine.chunks:
        st.warning("Upload and index your documents before asking a question.")
    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your financial documents..."):
                    engine.use_reranker = use_reranker
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

                with st.expander("Sources and retrieved passages"):
                    if not contexts:
                        st.write("No passages were retrieved.")
                    else:
                        for i, source in enumerate(contexts, start=1):
                            page = source.get("page")
                            page_text = (
                                f", page {page}"
                                if page is not None
                                else ""
                            )

                            st.markdown(
                                f"**{i}. {source.get('source', 'Unknown')}"
                                f"{page_text}**"
                            )

                            st.caption(
                                f"Hybrid: {source.get('score', 0.0):.3f} · "
                                f"Semantic: "
                                f"{source.get('semantic_score', 0.0):.3f} · "
                                f"BM25: "
                                f"{source.get('keyword_score', 0.0):.3f}"
                            )

                            st.write(source.get("text", ""))
                            if i < len(contexts):
                                st.divider()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": contexts,
                    }
                )

            except Exception as exc:
                error_message = (
                    f"Something went wrong: {type(exc).__name__}: {exc}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )
