# Real-time evaluators on LangSmith traces

## 1. Env vars (.env locally, Secrets on Streamlit/Render)

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=finance-rag-assistant      # the project the evaluator is attached to
LANGSMITH_LOG_CONTENT=true                   # false = question/answer/passages not sent to LangSmith
INAPP_EVALS=false                            # true = also score inside the app (optional)
```

## 2. Create the online evaluators (in the LangSmith UI, once)

Tracing -> your project -> **Evaluators** tab -> **+ Evaluator** -> **LLM-as-a-Judge**.

Create two evaluators with these settings:

| Setting | Value |
|---|---|
| Filter | `eq(name, "finance_rag_chat")` and tag `llm_answer` (root runs only) |
| Sampling rate | 1.0 while testing |
| Backfill | optional (scores older traces) |
| Judge model | add its API key in Secrets & API Keys |

### Correctness (no reference answer needed)

Prompt:
```
You are checking an answer from a financial-document assistant.

Question: {{input.question}}
Retrieved context: {{output.context_text}}
Answer: {{output.answer}}

Is the answer correct and complete for the question, judged using the context?
Score 1 if correct and complete, 0 if wrong or missing key information.
```
Feedback key: `correctness`

### Hallucination

Prompt:
```
Retrieved context: {{output.context_text}}
Answer: {{output.answer}}

Does the answer contain any claim or figure that is NOT supported by the context?
Score 1 if it contains unsupported claims (hallucination), 0 if every claim is supported.
```
Feedback key: `hallucination`

## 3. Test

1. Restart the app, ask a question.
2. Open the project -> trace named `finance_rag_chat`. Its Input must show `question`, Output must show `answer` and `context_text`.
3. After about a minute the feedback `correctness` and `hallucination` appear on the trace. If not, open Evaluators -> your evaluator -> Logs.
