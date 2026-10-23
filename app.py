"""Streamlit UI for the Finance RAG Assistant with LangSmith tracing."""

import os

import streamlit as st
from dotenv import load_dotenv
from langsmith import Client, traceable

# ---------------------------------------------------------
# 1. Load configuration BEFORE importing the RAG module.
# ---------------------------------------------------------

load_dotenv(override=False)

LANGSMITH_KEYS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
)

for key in LANGSMITH_KEYS:
    try:
        value = st.secrets.get(key)
        if value is not None:
            os.environ[key] = str(value)
    except Exception:
        # Streamlit secrets may not be configured locally.
        pass

# Set defaults only; never replace an existing API key.
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault(
    "LANGSMITH_ENDPOINT",
    "https://api.smith.langchain.com",
)
os.environ.setdefault(
    "LANGSMITH_PROJECT",
    "finance-rag-assistant",
)

from logger import logger
from rag import FinanceRAG, answer_with_llm


# ---------------------------------------------------------
# 2. Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Finance RAG Assistant",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Finance RAG Assistant")
st.caption(
    "Ask questions about your financial reports using "
    "document-grounded answers."
)


# ---------------------------------------------------------
# 3. LangSmith diagnostics
# ---------------------------------------------------------

def langsmith_config_status():
    """Return configuration status without revealing secrets."""
    return {
        "tracing_enabled": (
            os.getenv("LANGSMITH_TRACING", "").lower()
            in {"true", "1", "yes"}
        ),
        "endpoint_configured": bool(
            os.getenv("LANGSMITH_ENDPOINT")
        ),
        "api_key_configured": bool(
            os.getenv("LANGSMITH_API_KEY")
        ),
        "project": os.getenv(
            "LANGSMITH_PROJECT",
            "finance-rag-assistant",
        ),
    }


@traceable(name="streamlit_langsmith_test")
def langsmith_test_run():
    """A minimal trace independent of the RAG workflow."""
    return {
        "status": "success",
        "message": "LangSmith test function executed.",
    }


# ---------------------------------------------------------
# 4. Session state
# ---------------------------------------------------------

if "finance_rag" not in st.session_state:
    st.session_state.finance_rag = FinanceRAG(
        use_reranker=False
    )

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "messages" not in st.session_state:
    st.session_state.messages = []

engine = st.session_state.finance_rag


# ---------------------------------------------------------
# 5. Sidebar
# ---------------------------------------------------------

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
            "Extractive mode returns retrieved text without an "
            "LLM API call."
        ),
    )

    st.caption(
        "Tip: Start with one small PDF while testing deployment."
    )

    # -----------------------------------------------------
    # LangSmith status and test
    # -----------------------------------------------------

    with st.expander("LangSmith tracing diagnostics"):
        status = langsmith_config_status()

        st.write(
            "Tracing enabled:",
            status["tracing_enabled"],
        )
        st.write(
            "Endpoint configured:",
            status["endpoint_configured"],
        )
        st.write(
            "API key configured:",
            status["api_key_configured"],
        )
        st.write("Project:", status["project"])

        if st.button(
            "Test LangSmith tracing",
            use_container_width=True,
        ):
            if not status["tracing_enabled"]:
                st.error("LangSmith tracing is not enabled.")
            elif not status["api_key_configured"]:
                st.error(
                    "LangSmith API key is missing. "
                    "Check Streamlit Cloud Secrets."
                )
            else:
                try:
                    result = langsmith_test_run()
                    st.success(
                        f"Test function executed: "
                        f"{result['status']}"
                    )

                    # This performs an API operation, unlike merely
                    # constructing Client().
                    client = Client()
                    if client.info is not None:
                        st.success(
                            "LangSmith API authentication succeeded. "
                            "Now check the project for the test trace."
                        )
                except Exception as exc:
                    logger.exception(
                        "LangSmith diagnostic failed"
                    )
                    st.error(
                        "LangSmith diagnostic failed: "
                        f"{type(exc).__name__}. "
                        "Check Streamlit app logs for details."
                    )

        st.caption(
            "A successful test function does not guarantee the "
            "trace was uploaded. Verify the test run in LangSmith."
        )

    # -----------------------------------------------------
    # Index documents
    # -----------------------------------------------------

    if st.button(
        "Index documents",
        type="primary",
        use_container_width=True,
    ):
        if not uploaded_files:
            st.warning(
                "Upload at least one PDF or TXT file first."
            )
        else:
            max_file_size = 10 * 1024 * 1024

            oversized = [
                file.name
                for file in uploaded_files
                if file.size > max_file_size
            ]

            if oversized:
                st.error(
                    "These files exceed the 10 MB limit: "
                    + ", ".join(oversized)
                )
                logger.warning(
                    "Upload rejected due to size limit: %s",
                    oversized,
                )
            else:
                try:
                    engine.use_reranker = bool(use_reranker)

                    with st.spinner(
                        "Reading and indexing documents..."
                    ):
                        # Explicit trace for custom Python code.
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
                    logger.exception(
                        "Document indexing failed in Streamlit UI"
                    )
                    st.error(
                        f"Indexing failed ({type(exc).__name__}). "
                        "Check application logs for details."
                    )

    if st.session_state.indexed:
        st.success("Documents are indexed.")

        with st.expander("Indexed files"):
            for filename in st.session_state.indexed_files:
                st.write(f"📄 {filename}")

    if st.button(
        "Clear chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Privacy: Uploaded documents are processed by this app. "
        "With Groq selected, retrieved excerpts are sent to Groq."
    )


# ---------------------------------------------------------
# 6. Display previous messages
# ---------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        sources = message.get("sources", [])

        if message["role"] == "assistant" and sources:
            with st.expander("Sources and retrieved passages"):
                for i, source in enumerate(sources, start=1):
                    page = source.get("page")
                    page_text = (
                        f", page {page}"
                        if page is not None
                        else ""
                    )

                    st.markdown(
                        f"**{i}. "
                        f"{source.get('source', 'Unknown')}"
                        f"{page_text}**"
                    )
                    st.write(source.get("text", ""))

                    if i < len(sources):
                        st.divider()


# ---------------------------------------------------------
# 7. Chat and RAG execution
# ---------------------------------------------------------

question = st.chat_input(
    "Ask about revenue, expenses, profit, ratios, or financial risks..."
)

if question:
    if not st.session_state.indexed or not engine.chunks:
        st.warning(
            "Upload and index documents before asking a question."
        )
    else:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                engine.use_reranker = bool(use_reranker)

                # Trace retrieval, including the query and retrieved
                # passages. Avoid tracing sensitive financial content
                # unless you are authorized to send it to LangSmith.
                with st.spinner("Searching financial documents..."):
                    contexts = traced_retrieve(
                        engine,
                        question,
                        top_k,
                    )

                use_llm = answer_mode == "Groq LLM"

                with st.spinner("Preparing your answer..."):
                    answer = traced_answer(
                        question,
                        contexts,
                        use_llm,
                    )

                st.markdown(answer)

                if contexts:
                    with st.expander(
                        "Sources and retrieved passages"
                    ):
                        for i, source in enumerate(
                            contexts, start=1
                        ):
                            page = source.get("page")
                            page_text = (
                                f", page {page}"
                                if page is not None
                                else ""
                            )

                            st.markdown(
                                f"**{i}. "
                                f"{source.get('source', 'Unknown')}"
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
                    "Please try again. Check application logs "
                    "if the issue persists."
                )
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )


# ---------------------------------------------------------
# 8. Trace wrappers for custom RAG methods
# ---------------------------------------------------------

@traceable(name="finance_rag_retrieval")
def traced_retrieve(engine, question, top_k):
    return engine.retrieve(question, top_k=top_k)


@traceable(name="finance_rag_answer")
def traced_answer(question, contexts, use_llm):
    return answer_with_llm(
        question,
        contexts,
        use_llm=use_llm,
        fallback_on_error=True,
    )
