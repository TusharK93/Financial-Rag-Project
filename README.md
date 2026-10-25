# Finance RAG Assistant

A Retrieval-Augmented Generation (RAG) application for asking questions about financial documents. Upload PDF or TXT files, retrieve relevant passages, and generate answers grounded in the retrieved content.

## Features

- **Document ingestion:** Upload PDF and TXT files.
- **Semantic search:** Retrieve relevant document passages using embeddings.
- **Keyword search:** Use BM25 to find relevant terms and phrases.
- **Hybrid retrieval:** Combine semantic and keyword scores to improve retrieval.
- **Optional reranking:** Use a cross-encoder reranker to refine retrieved results.
- **Multiple LLM providers:** Configure Groq or Gemini, depending on the providers supported by your implementation.
- **LLM gateway:** Centralize model-provider selection when supported by the project configuration.
- **Guardrails:** Validate inputs, filter retrieved context, and check generated responses when enabled.
- **Source attribution:** Display retrieved sources to help users verify answers.
- **Extractive fallback:** Support a fallback response mode when configured.
- **Optional LangSmith tracing:** Trace supported LLM and application operations when tracing is correctly configured.

## Tech Stack

- Python
- Streamlit
- Sentence Transformers
- BM25 keyword retrieval
- Cross-encoder reranking
- Groq and/or Google Gemini
- LangSmith for optional tracing

The exact dependencies and enabled features depend on the installed packages and the implementation in your project files.

## Project Structure

```text
Financial-Rag-Project/
├── app.py
├── rag.py
├── config.py
├── llm.py
├── gateway.py
├── guardrails.py
├── logger.py
├── requirements.txt
├── README.md
├── .env
└── tests/
```

Your actual project may contain additional files or use a slightly different structure. Keep this section synchronized with the files present in your repository.

## Prerequisites

- Python 3.10 or a compatible version supported by your dependencies
- Git
- A Groq API key and/or Google Gemini API key, depending on the configured provider
- Optional LangSmith API key if tracing is enabled

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/TusharK93/Financial-Rag-Project.git
cd Financial-Rag-Project
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt:

```bat
venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installation fails, check that your Python version is compatible with the packages in `requirements.txt`.

### 4. Configure environment variables

Create a `.env` file in the project root.

Example:

```dotenv
# LLM provider: auto, groq, or gemini
LLM_PROVIDER=auto

# Groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b

# Google Gemini
GOOGLE_API_KEY=your_google_api_key
GEMINI_MODEL=your_supported_gemini_model

# Embeddings and reranking
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANK_MODEL=BAAI/bge-reranker-base

# Optional LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=finance-rag-assistant
```

Use model identifiers supported by the relevant provider and by your implementation. The example model names are configuration examples, not a guarantee that every model is available to every account or API.

**Important:** Never commit real API keys or passwords to GitHub. Add `.env` to `.gitignore`. If a real key has already been exposed or committed, revoke or rotate it.

## Run the Application

From the project root, activate your virtual environment and run:

```bash
streamlit run app.py
```

Streamlit will display a local URL, typically:

```text
http://localhost:8501
```

Open that URL in your browser.

## How It Works

1. **Upload documents:** The user uploads financial documents.
2. **Extract text:** The application reads text from supported files.
3. **Split text into chunks:** The extracted text is divided into smaller passages.
4. **Create embeddings:** The embedding model represents passages as vectors.
5. **Retrieve relevant passages:** Semantic retrieval and BM25 keyword retrieval identify candidate passages.
6. **Combine retrieval scores:** The application combines normalized semantic and keyword scores according to the implementation in `rag.py`.
7. **Rerank candidates:** If configured, a cross-encoder reranks the retrieved passages.
8. **Generate an answer:** The selected LLM uses the retrieved context to answer the question.
9. **Apply guardrails:** Configured checks validate the input, retrieved context, or generated answer.
10. **Show sources:** The application presents source information where available.

### Retrieval Scoring

If your current `rag.py` uses a weighted normalized score, the combination can be represented as:

\[
S(d)=0.65S_{\text{semantic}}(d)+0.35S_{\text{BM25}}(d)
\]

Here, both scores should be normalized or otherwise made comparable before combination. This formula describes the weighted scoring implementation; it is not Reciprocal Rank Fusion (RRF).

## LLM Provider Configuration

Set `LLM_PROVIDER` in `.env` according to the provider-selection logic implemented in your project.

- `groq`: Use Groq if supported by your `llm.py` or gateway.
- `gemini`: Use Google Gemini if supported by your implementation.
- `auto`: Let the application choose a provider only if automatic selection is implemented.

The variable names in `.env` must match the names actually read by your Python files. If your code uses a different name, update either the configuration or the code consistently.

## Optional LangSmith Tracing

To enable LangSmith tracing, configure the appropriate environment variables:

```dotenv
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=finance-rag-assistant
```

Tracing requires more than setting environment variables. Ensure that the relevant LLM or application calls are instrumented using supported LangSmith integrations or tracing decorators. Confirm that the API key is valid and that the process can access the LangSmith service.

If traces do not appear, check:

- The environment variables are loaded before the traced calls execute.
- The tracing flag is enabled.
- The correct API key and project name are being used.
- The code path actually invokes an instrumented operation.
- No exception or configuration issue is preventing trace submission.

## Deployment

You can deploy the application to a hosting platform that supports Python and Streamlit.

General deployment steps:

1. Push the project to a GitHub repository.
2. Create an application on your chosen hosting platform.
3. Connect the repository and select the appropriate branch.
4. Configure the start command, typically:

   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

5. Add API keys and other environment variables through the platform's secret or environment-variable settings.
6. Install dependencies from `requirements.txt`.
7. Deploy and inspect the application logs.

The start command and port configuration may need to be adjusted for your hosting platform. Do not store API keys in source code or public repository files.

## Troubleshooting

### Missing API key

Check that the required key is present in `.env` locally or in the deployment platform's environment settings. Confirm that the variable name matches the name read by your code.

### Model not found

Verify the model identifier, provider access, and compatibility with the installed SDK.

### No relevant documents retrieved

Check that text extraction succeeded, chunks were created, embeddings were generated, and the retrieval thresholds and scoring logic are appropriate.

### Tracing does not appear

Verify LangSmith configuration and confirm that the relevant calls are instrumented. Setting environment variables alone does not guarantee traces.

### Dependency installation fails

Check the Python version, package versions, and any platform-specific dependencies.

## Security Notes

- Keep `.env` out of version control.
- Never expose API keys in source code, screenshots, logs, or public issue reports.
- Rotate credentials that have been exposed.
- Validate uploaded files and enforce reasonable file-size limits.
- Avoid logging sensitive financial document content.
- Apply appropriate access controls before deploying the application for multiple users.

## Limitations

- Generated answers may be inaccurate or incomplete.
- Retrieval quality depends on text extraction, chunking, embeddings, and ranking.
- Scanned PDFs may require OCR support if the current implementation does not extract their text.
- Model availability, rate limits, and costs depend on the selected provider.
- Source attribution helps verification but does not guarantee that every answer is fully supported by its sources.

## License

Add the license and usage terms applicable to this repository.

## Author

**Tushar Kumawat**

GitHub: [TusharK93](https://github.com/TusharK93)
