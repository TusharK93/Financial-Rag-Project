<<<<<<< HEAD
Finance RAG Assistant — Groq + LlamaIndex
=======
Finance RAG Assistant
>>>>>>> 43278181b7b698ef1de1f17b31c7ce65d0b317c5
Features
- Streamlit chat interface
- PDF and TXT ingestion with page/source metadata
- LlamaIndex Document and SentenceSplitter integration
- BM25 keyword search and Sentence Transformer dense retrieval
- Reciprocal Rank Fusion (RRF) hybrid ranking
- Optional cross-encoder reranker
- Groq chat completions using a configurable model ID
- Extractive mode that makes no LLM API call
- Fallback BM25 implementation if rank-bm25 is unavailable
Install on Windows
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
The embedding model is downloaded on first indexing. This needs internet access.
Configure Groq locally
Copy .env.example to .env and set GROQ_API_KEY.
Do not commit .env or publish the key. Verify the selected model is available to your
Groq account; change GROQ_MODEL if necessary.
Deploy on Streamlit Community Cloud
1. Push app.py, rag.py, and requirements.txt to GitHub.
2. Create an app at https://share.streamlit.io/.
3. Set GROQ_API_KEY in app Settings → Secrets, for example:
   GROQ_API_KEY = "your-key-here"
4. Deploy with app.py as the entry point.
5. Initially disable reranking to reduce memory use.
Free hosting and inference quotas are limited and can change. Do not assume Groq API usage
is unlimited or guaranteed to be free. The local extractive mode does not call Groq.
Limitations
- Scanned/image-only PDFs need OCR, which is not included.
- The in-memory index is not persistent; documents may need re-indexing after restarts.
- Reranker and embedding model downloads can exceed free-host memory or startup limits.
<<<<<<< HEAD
- Use this as a document research assistant, not as personalized investment advice.
=======
- Use this as a document research assistant, not as personalized investment advice.
>>>>>>> 43278181b7b698ef1de1f17b31c7ce65d0b317c5
