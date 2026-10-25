"""LangSmith tracing for the Finance RAG app.

One ROOT run (finance_rag_chat) is created per chat question. It contains
clean inputs/outputs so LangSmith online evaluators (correctness,
hallucination, ...) can read them:

    input.question      -> the user's question
    output.answer       -> the model's answer
    output.context_text -> the retrieved passages the answer was based on

Retrieval and generation appear as child runs under this root run.
"""

from __future__ import annotations

import os

from rag import generate_answer, traceable

try:
    from langsmith.run_helpers import get_current_run_tree
except Exception:  # langsmith missing or very old
    get_current_run_tree = None

LLM_PROVIDERS = ("groq", "gemini")


def log_content() -> bool:
    """False hides question/answer/passages from traces (also disables evaluators)."""
    return os.getenv("LANGSMITH_LOG_CONTENT", "true").strip().lower() != "false"


def _hide(_data):
    return {"hidden": "LANGSMITH_LOG_CONTENT=false"}


def _context_text(contexts) -> str:
    blocks = []
    for i, item in enumerate(contexts or [], start=1):
        page = item.get("page")
        label = f", page {page}" if page is not None else ""
        blocks.append(f"[Source {i}: {item.get('source', 'Unknown')}{label}]\n{item.get('text', '')}")
    return "\n\n".join(blocks)


def finance_rag_chat(engine, question, top_k, provider, use_llm, use_guardrails, run_id=None):
    """Retrieve + answer inside one traced root run. Returns an AnswerResult."""
    holder = {}
    hide = None if log_content() else _hide

    @traceable(name="finance_rag_retrieval", run_type="retriever", hide_inputs=hide, hide_outputs=hide)
    def _retrieve(question, top_k):
        return engine.retrieve(question, top_k=top_k)

    @traceable(
        name="finance_rag_chat",
        run_type="chain",
        tags=["finance-rag"],
        hide_inputs=hide,
        hide_outputs=hide,
    )
    def _run(question, top_k, provider, use_llm, use_guardrails):
        contexts = _retrieve(question, top_k)
        result = generate_answer(
            question,
            contexts,
            provider=provider,
            use_llm=use_llm,
            fallback_on_error=True,
            use_guardrails=use_guardrails,
        )
        holder["result"] = result

        # Tag runs answered by a real LLM so evaluators can skip refusals / extractive output.
        if get_current_run_tree is not None and result.provider in LLM_PROVIDERS and not result.blocked:
            try:
                rt = get_current_run_tree()
                if rt is not None:
                    rt.add_tags(["llm_answer"])
                    rt.add_metadata({"provider": result.provider, "model": result.model})
            except Exception:
                pass

        return {
            "answer": result.answer,
            "context_text": _context_text(result.contexts),
            "sources": [
                {"source": c.get("source"), "page": c.get("page")} for c in result.contexts
            ],
            "provider": result.provider,
            "model": result.model,
            "blocked": result.blocked,
            "latency_ms": result.latency_ms,
        }

    extra = {"run_id": run_id} if run_id is not None else None
    _run(question, top_k, provider, use_llm, use_guardrails, langsmith_extra=extra)
    return holder["result"]
