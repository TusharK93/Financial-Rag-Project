"""OPTIONAL in-app evaluators (correctness + hallucination).

Use this only if you also want scores produced by the app itself.
The normal way is a LangSmith *online evaluator* created in the project's
Evaluators tab (see LANGSMITH_EVALUATORS.md). Do not enable both for the
same feedback keys, or you will get duplicate scores.

Enable with:  INAPP_EVALS=true
Scores are sent to the trace as feedback:
    correctness   1.0 = answer correct / complete for the question, 0.0 = wrong
    hallucination 1.0 = answer contains unsupported claims,        0.0 = fully supported
"""

from __future__ import annotations

import json
import os
import re
import threading
import time

import llm
from logger import logger

JUDGE_SYSTEM = (
    "You are a strict evaluator of answers from a financial-document assistant. "
    "Judge ONLY with the given context. Reply with one JSON object and nothing else."
)

JUDGE_TEMPLATE = """Question:
{question}

Retrieved context:
{context}

Assistant answer:
{answer}

Score the answer:
- correctness: 0.0 to 1.0. Is the answer correct and complete for the question, according to the context?
- hallucination: 0.0 or 1.0. 1.0 if the answer states facts or figures NOT supported by the context, else 0.0.

Return exactly: {{"correctness": <number>, "hallucination": <number>, "reason": "<one short sentence>"}}"""


def enabled() -> bool:
    return os.getenv("INAPP_EVALS", "false").strip().lower() == "true"


def _parse(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return {
            "correctness": max(0.0, min(1.0, float(data["correctness"]))),
            "hallucination": 1.0 if float(data["hallucination"]) >= 0.5 else 0.0,
            "reason": str(data.get("reason", ""))[:300],
        }
    except Exception:
        return None
from langsmith import Client


def _post_feedback(client, run_id, scores: dict) -> None:
    client = Client()
    project = client.read_project(project_name="finance-rag-assistant")
    session_id = project.id
    for key in ("correctness", "hallucination"):
        for attempt in range(4):  # the run may not be ingested yet; retry briefly
            try:
                client.create_feedback(
                    run_id=run_id,
                    session_id=session_id,
                    key=key,
                    score=scores[key],
                    comment=scores["reason"],
                    feedback_source_type="model",)
                break
            except Exception as exc:
                if attempt == 3:
                    logger.warning("In-app eval: feedback '%s' failed: %s", key, type(exc).__name__)
                else:
                    time.sleep(3)


def _work(run_id, question, answer, context):
    try:
        from langsmith import Client

        verdict = llm.generate(
            JUDGE_SYSTEM,
            JUDGE_TEMPLATE.format(question=question, context=context[:12000], answer=answer),
        )
        scores = _parse(verdict.text)
        if scores is None:
            logger.warning("In-app eval: judge returned unparsable output")
            return
        _post_feedback(Client(), run_id, scores)
    except Exception as exc:
        logger.warning("In-app eval failed: %s", type(exc).__name__)


def evaluate_async(run_id, question: str, result) -> None:
    """Score an AnswerResult in a background thread (never blocks or breaks the chat)."""
    if not enabled() or run_id is None:
        return
    if result.provider not in ("groq", "gemini") or result.blocked or not result.contexts:
        return
    context = "\n\n".join(str(c.get("text", "")) for c in result.contexts)
    threading.Thread(
        target=_work, args=(run_id, question, result.answer, context), daemon=True
    ).start()
