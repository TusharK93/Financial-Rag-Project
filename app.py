import os
from pathlib import Path

import nltk
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------
# 1. Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Finance RAG Assistant",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------
# 2. Initialize NLTK before importing rag.py
# ---------------------------------------------------------


# ---------------------------------------------------------
# 2. Initialize NLTK before importing rag.py
# ---------------------------------------------------------

# Prefer a corpus bundled with the project.
PROJECT_NLTK_DIR = (
    Path(__file__).parent / "resources" / "nltk_data"
)

# Fallback to a writable temporary directory on Streamlit Cloud.
TEMP_NLTK_DIR = Path("/tmp/nltk_data")

if PROJECT_NLTK_DIR.exists():
    NLTK_DATA_DIR = str(PROJECT_NLTK_DIR)
else:
    NLTK_DATA_DIR = str(TEMP_NLTK_DIR)
    TEMP_NLTK_DIR.mkdir(parents=True, exist_ok=True)

os.environ["NLTK_DATA"] = NLTK_DATA_DIR

if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

try:
    nltk.data.find("corpora/stopwords")

except LookupError:
    try:
        downloaded = nltk.download(
            "stopwords",
            download_dir=NLTK_DATA_DIR,
            quiet=False,
            raise_on_error=True,
        )

        if not downloaded:
            raise RuntimeError(
                "NLTK stopwords download failed."
            )

        nltk.data.find("corpora/stopwords")

    except Exception as exc:
        raise RuntimeError(
            "NLTK stopwords are unavailable. "
            "If Streamlit Cloud cannot download them, "
            "bundle the corpus in resources/nltk_data/corpora/stopwords."
        ) from exc


# ---------------------------------------------------------
# 3. Load environment variables and Streamlit secrets
# ---------------------------------------------------------

load_dotenv(override=False)

LANGSMITH_DEFAULTS = {
    "LANGSMITH_TRACING": "true",
    "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
    "LANGSMITH_PROJECT": "finance-rag-assistant",
}

# Read configured Streamlit secrets without printing their values.
for key in (
    "LANGSMITH_TRACING",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
):
    try:
        value = st.secrets.get(key)

        if value is not None and str(value).strip():
            os.environ[key] = str(value).strip()

    except Exception:
        # Secrets may not be configured during local development.
        pass

for key, value in LANGSMITH_DEFAULTS.items():
    os.environ.setdefault(key, value)

# Import tracing only after environment configuration.
from langsmith import Client, traceable
from logger import logger
from rag import FinanceRAG, answer_with_llm

# ---------------------------------------------------------
# 4. LangSmith diagnostics
# ---------------------------------------------------------

def langsmith_config_status():
    """Return configuration status without exposing credentials."""

    return {
        "tracing_enabled": os.getenv(
            "LANGSMITH_TRACING", ""
        ).lower() in {"true", "1", "yes"},
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
    """Create a small test trace in LangSmith."""

    return {"status": "success"}


# ---------------------------------------------------------
# 5. Traced RAG wrappers
# Define these BEFORE any chat code calls them.
# Hide inputs and outputs to avoid sending document passages
# or financial answers as trace payloads by default.
# ---------------------------------------------------------

@traceable(
    name="finance_rag_retrieval",
    hide_inputs=True,
    hide_outputs=True,
)
def traced_retrieve(engine, question, top_k):
    """Retrieve relevant passages from the indexed documents."""

    return engine.retrieve(question, top_k=top_k)


@traceable(
    name="finance_rag_answer",
    hide_inputs=True,
    hide_outputs=True,
)
def traced_answer(question, contexts, use_llm):
    """Generate an answer from the retrieved passages."""

    return answer_with_llm(
        question,
        contexts,
        use_llm=use_llm,
        fallback_on_error=True,
    )


# ---------------------------------------------------------
# 6. Streamlit page
# ---------------------------------------------------------

st.title("📊 Finance RAG Assistant")

st.caption(
    "Ask questions about your financial reports using "
    "document-grounded answers."
)

# ---------------------------------------------------------
# 7. Session state
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
# 8. Sidebar
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
            "Uses additional memory. Keep disabled on "
            "small-memory deployments."
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
            "Extractive mode returns retrieved text without "
            "an LLM API call."
        ),
    )

    st.caption(
        "Tip: Start with one small PDF while testing deployment."
    )

    # -----------------------------------------------------
    # LangSmith diagnostics
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
                st.error(
                    "Tracing is disabled. Set "
                    "LANGSMITH_TRACING to true."
                )

            elif not status["api_key_configured"]:
                st.error(
                    "LangSmith API key is missing. "
                    "Check Streamlit Cloud Secrets."
                )

            elif not status["endpoint_configured"]:
                st.error(
                    "LangSmith endpoint is not configured."
                )

            else:
                try:
                    # Execute a function decorated with @traceable.
                    result = langsmith_test_run()

                    # Check that the LangSmith API is reachable
                    # and accepts the configured credentials.
                    client = Client()
                    next(
                        iter(client.list_projects(limit=1)),
                        None,
                    )

                    st.success(
                        "LangSmith API request succeeded and "
                        "the test function executed. Check the "
                        "selected project for its trace."
                    )

                except Exception as exc:
                    logger.exception(
                        "LangSmith diagnostic failed"
                    )

                    st.error(
                        "LangSmith test failed "
                        f"({type(exc).__name__}). "
                        "Check the app logs for details."
                    )

        st.caption(
            "API authentication and trace upload are different "
            "checks. Confirm that the test run appears in LangSmith."
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
                        passage_count = engine.ingest(
                            uploaded_files
                        )

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
        "Privacy: With Groq selected, retrieved excerpts are sent "
        "to Groq. The traced RAG wrappers hide their inputs and "
        "outputs by default."
    )

# ---------------------------------------------------------
# 9. Display previous chat messages
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
# 10. Chat and RAG execution
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

                with st.spinner(
                    "Searching financial documents..."
                ):
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
                            contexts,
                            start=1,
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
                    st.info(
                        "No relevant passages were retrieved."
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": contexts,
                    }
                )

            except Exception as exc:
                logger.exception(
                    "Chat request failed in Streamlit UI"
                )

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
