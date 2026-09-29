# 📊 Finance RAG Assistant

A Retrieval-Augmented Generation (RAG) application that lets users upload financial documents and ask questions about their contents through an interactive Streamlit chat interface.

The application combines **semantic search, BM25 keyword retrieval, hybrid ranking, and optional cross-encoder reranking** to retrieve relevant passages from financial reports. It can generate context-grounded answers using the Groq API or operate in extractive mode without making any LLM API calls.

## ✨ Features

- **Interactive chat interface:** Ask questions about uploaded financial documents using Streamlit.
- **PDF and TXT ingestion:** Extract document text while preserving source filenames and page metadata.
- **LlamaIndex integration:** Use `Document` and `SentenceSplitter` for structured document processing and chunking, with a built-in fallback text splitter.
- **Semantic search:** Generate dense embeddings using Sentence Transformers.
- **BM25 keyword search:** Retrieve passages using keyword relevance, with a fallback BM25 implementation.
- **Hybrid retrieval:** Combine semantic and keyword search rankings using a Reciprocal Rank Fusion (RRF) strategy.
- **Optional cross-encoder reranking:** Improve the ordering of retrieved passages using a reranker model.
- **Groq-powered answers:** Generate answers using a configurable Groq chat-completion model.
- **Extractive mode:** Return relevant source passages without calling an LLM API.
- **Source attribution:** Display retrieved passages with filenames and page numbers.
- **Configurable retrieval:** Adjust the number of retrieved passages and enable or disable reranking.
- **Fallback handling:** Fall back to retrieved passages when Groq is not configured or answer generation fails.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | User interface and chat |
| PyMuPDF | PDF text extraction |
| LlamaIndex Core | Document processing and sentence splitting |
| Sentence Transformers | Dense embeddings |
| NumPy | Vector operations and similarity scoring |
| BM25 | Keyword-based retrieval |
| CrossEncoder | Optional passage reranking |
| Groq API | LLM-based answer generation |
| python-dotenv | Local environment configuration |

## 🏗️ Architecture

```text
             Financial Documents
                  PDF / TXT
                     |
                     v
             Document Ingestion
                     |
                     v
             Text Extraction
          + Page and Source Metadata
                     |
                     v
               Text Chunking
                     |
          +----------+----------+
          |                     |
          v                     v
     Dense Embeddings       BM25 Index
          |                     |
          v                     v
   Semantic Retrieval     Keyword Retrieval
          |                     |
          +----------+----------+
                     |
                     v
             Hybrid Ranking
                     |
                     v
          Optional Cross-Encoder
                Reranking
                     |
                     v
           Relevant Source Passages
                     |
                     v
              Answer Generation
               /            \
              v              v
         Groq LLM       Extractive Mode
               \            /
                v          v
               Streamlit Chat
```

*Note: The diagram describes the intended retrieval pipeline. Use Reciprocal Rank Fusion (RRF) in the implementation if you want the architecture to match the code exactly.*

## 📁 Project Structure

```text
finance_rag_assistant/
│
├── app.py              # Streamlit user interface
├── rag.py              # Document ingestion and RAG pipeline
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Files excluded from Git
└── README.md           # Project documentation
```

## ⚙️ Requirements

- Python 3.10 or later, compatible with your installed dependencies
- pip
- Internet access for downloading embedding models on first use
- A Groq API key for LLM-based answer generation; optional for extractive mode

## 🚀 Installation on Windows

### 1. Clone the repository

```powershell
git clone https://github.com/TusharK93/Financial-Rag-Project.git
cd Financial-Rag-Project
```

### 2. Create a virtual environment

```powershell
py -m venv .venv
```

### 3. Activate the environment

For PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you can activate the environment using Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Configure environment variables

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Add your Groq API key to `.env`:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANK_MODEL=BAAI/bge-reranker-base
```

The selected Groq model must be available to your account. Change `GROQ_MODEL` if necessary.

**Security:** Never commit `.env` or expose your API key in source code, screenshots, logs, or a public repository.

### 6. Run the application

```powershell
streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that address in your browser.

## 💬 How to Use

1. Start the Streamlit application.
2. Upload one or more PDF or TXT financial documents.
3. Click the document indexing button in the sidebar.
4. Wait for text extraction, chunking, and embedding generation to finish.
5. Choose the number of passages to retrieve.
6. Optionally enable cross-encoder reranking.
7. Select Groq-powered answer generation or extractive mode.
8. Ask questions about the uploaded documents.
9. Review the answer and the source passages used to support it.

Example questions:

- What was the company's total revenue?
- How did operating expenses change during the reporting period?
- What risks are discussed in the annual report?
- Summarize the company's cash flow position.
- Which page mentions outstanding debt?
- Compare the financial figures reported in the retrieved passages.

Answers depend on the content of the documents you upload. The assistant should not be treated as a substitute for professional financial advice.

## 🔍 Retrieval Pipeline

### 1. Document ingestion

PDF text is extracted with PyMuPDF, while TXT files are decoded as UTF-8. The application preserves the source filename and page number for each chunk.

### 2. Text chunking

Documents are divided into smaller passages to improve retrieval relevance. LlamaIndex's `SentenceSplitter` is used when available; otherwise, a built-in overlapping text splitter is used.

### 3. Dense retrieval

Sentence Transformers converts document passages and user questions into vector embeddings. Similarity scores help identify semantically relevant passages, including passages that use different wording from the query.

### 4. BM25 keyword retrieval

BM25 ranks passages according to keyword matches and term frequency. A fallback implementation is provided if an external BM25 package is unavailable.

### 5. Hybrid ranking

Semantic and keyword retrieval can complement one another. Reciprocal Rank Fusion (RRF), when implemented in the retrieval code, combines ranked results using:

\[
\operatorname{RRF}(d)=
\sum_{r \in R}
\frac{1}{k+\operatorname{rank}_r(d)}
\]

Here, \(d\) is a document passage, \(R\) is the set of retrieval rankings, and \(k\) is a ranking constant.

### 6. Optional reranking

A cross-encoder evaluates the question and candidate passages together to refine their order. Reranking may improve relevance but requires additional model resources.

### 7. Answer generation

In Groq mode, the retrieved passages are provided as context to a configurable chat model. In extractive mode, the application returns retrieved passages directly without calling an LLM API.

## ☁️ Deployment on Streamlit Community Cloud

### 1. Push the project to GitHub

Ensure your repository includes:

- `app.py`
- `rag.py`
- `requirements.txt`
- `README.md`
- `.env.example`

Do not commit your real `.env` file.

### 2. Create the Streamlit app

Visit [Streamlit Community Cloud](https://share.streamlit.io/), sign in, and create a new app using your GitHub repository.

Select the appropriate branch and set the main file path to:

```text
app.py
```

### 3. Configure secrets

Open your app's settings and add the Groq API key under **Secrets**:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

Use the model identifier currently supported by your Groq account. The `python-dotenv` package does not automatically load Streamlit Cloud's Secrets into environment variables; if the application relies on `os.getenv()`, make sure the code reads Streamlit secrets or maps them into its configuration.

### 4. Deploy

Deploy the application and check the logs if startup fails. Start with reranking disabled to reduce memory consumption and model-download requirements.

Hosting limits, available memory, inference quotas, and API pricing may change over time.

## 🔐 Security Best Practices

- Keep API keys in local environment variables or your hosting provider's secret manager.
- Add `.env` and `.venv/` to `.gitignore`.
- Never commit credentials or private financial documents.
- Do not assume uploaded documents are confidential merely because they are processed by a RAG application.
- Review your hosting provider's data handling and retention policies before uploading sensitive documents.

Recommended `.gitignore` entries:

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
.streamlit/secrets.toml
```

## ⚠️ Limitations

- **Scanned PDFs:** Image-only documents require OCR, which is not included.
- **In-memory index:** Indexed documents are not persisted across application restarts and may need to be uploaded and indexed again.
- **Model downloads:** Embedding and reranker models may require substantial download time, disk space, and memory.
- **Free hosting constraints:** Large documents, reranking, and model initialization may exceed available resources.
- **Retrieval accuracy:** Incorrect, incomplete, or poorly extracted source text can reduce answer quality.
- **LLM dependency:** Groq-generated answers require a valid API key, network access, model availability, and sufficient account quota.
- **Extractive mode:** This mode returns retrieved passages rather than synthesizing a complete answer.
- **Financial reliability:** Verify important figures and decisions against the original financial statements.

## 🛣️ Future Improvements

- Add OCR support for scanned financial reports.
- Persist indexes to avoid reprocessing documents after restarts.
- Implement and evaluate Reciprocal Rank Fusion (RRF) against alternative hybrid-ranking methods.
- Add document-level comparisons and financial metric extraction.
- Add automated retrieval evaluation and answer-quality tests.
- Improve streaming responses and error reporting.
- Add authentication and access controls for private documents.

## 🤝 Contributing

Contributions, bug reports, and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make and test your changes.
4. Submit a pull request describing the changes.

## 📄 License

Add a `LICENSE` file to specify how this project may be used, modified, and distributed.

---

**Built with Python, Streamlit, Sentence Transformers, and Groq.**
