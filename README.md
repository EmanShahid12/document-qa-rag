# Document Q&A with Citations (local RAG + LangGraph agent)

A fully local question-answering system for PDF/TXT/MD documents. Every answer comes with numbered citations that map back to the real file, page and chunk, and the system says "not found" instead of guessing when the documents do not contain the answer.

Runs entirely on your machine: FAISS + BM25 retrieval, a cross-encoder reranker, and a local LLM through Ollama. No API keys needed.

## Features

- Hybrid retrieval: FAISS (semantic) + BM25 (keywords), then a cross-encoder reranker.
- Citations: answers carry [1], [2] markers mapped to file, page, chunk and snippet.
- Confidence gating: each answer is `grounded`, `low_confidence` or `not_found`.
- Validated output: Pydantic models for answers and quizzes.
- Summarization (map-reduce) and quiz generation (validated JSON with retries).
- LangGraph agent with SQLite conversation memory and tool-calling (search, summarize, quiz, list).
- Evaluation scripts: hand-written eval set with page-level retrieval metrics.

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

ollama pull llama3.2          # needs Ollama installed and running
mkdir data                    # put your PDFs here
python app.py index --data data
python app.py ask "What is pre-training?"
```

Useful environment variables: `OLLAMA_MODEL`, `OLLAMA_NUM_CTX` (default 8192), `HIGH_CONF`, `LOW_CONF`, `RAG_DEBUG=1`.

## Evaluation

Test document: a 315-page book on large language models. Eval set: 30 hand-written questions (22 answerable, 8 unanswerable: 3 out-of-domain and 5 in-domain topics that are absent from the book). Expected pages were verified against the PDF. Page numbers are PDF pages.

```powershell
python evaluate_v2.py eval_set_v2.json
```

First measured run (before the fixes listed below):

| Metric | Result |
|---|---|
| Correct source page in top-4 chunks | 19/22 |
| Correct source page ranked first | 16/22 |
| Cited source page is correct | 19/22 |
| Answers with a usable [n] citation | 20/22 |
| Unanswerable questions correctly refused | 7/8 |

This run exposed a bug: the model replied `NOT FOUND.` (with a full stop) and the old normalisation did not recognise it, so the answer was not marked `not_found`. It also showed 2 answers without a numeric citation. Both were fixed in `rag.py` (see Changelog). The raw results are saved in `results/` (`eval_results_v2_before.json` and `eval_results_v2_after.json`).
After the fixes (answerable questions only, 22 total):

| Status | Before | After |
|---|---|---|
| grounded | 19 | 22 |
| low_confidence | 3 | 0 |

`grounded` means the confidence gate passed; it does not by itself prove the answer is correct. Answers were checked manually (see Limitations).
## Limitations

- The eval set is small and hand-written; results are indicative, not a benchmark.
- Keyword matching is a crude correctness proxy; answers must be read manually.
- Retrieval confidence and correctness are different things: an irrelevant question can still get a high reranker score, so the LLM's NOT_FOUND reply acts as a second gate.
- The LLM (llama3.2, 3B) can still cite incorrectly or paraphrase imprecisely. Citations show where the answer came from; they do not prove the answer is entailed by the source.
- Quiz answer keys are LLM-generated and not independently verified.
- The agent's tool routing uses regex rules as a safety guard on top of LLM tool-calling; unusual phrasing may be routed wrongly.
- `allow_dangerous_deserialization=True` is used to load the FAISS index. Only load indexes you built yourself.

## Changelog

- Fixed `NOT FOUND.` / `Not found!` not being treated as `not_found`.
- If the model answers without a valid numeric citation, it is asked once more to cite properly.
- Larger Ollama context (`num_ctx`) to avoid silent prompt truncation.

## Project layout

```
rag.py                  core pipeline (retrieval, answering, summarize, quiz)
app.py                  command-line interface
tools.py                LangChain tools wrapping the pipeline
graph_agent_step3.py    LangGraph agent with SQLite memory (final agent version)
evaluate.py             original small evaluation
evaluate_v2.py          30-question evaluation with page-level metrics
evaluate_agent.py       offline + live agent checks
debug_find.py           explains why a specific question failed retrieval
eval_set.json           original evaluation questions
eval_set_v2.json        30-question evaluation set
results/                saved evaluation outputs (before and after fixes)
archive/                earlier agent versions (agent.py, graph_agent.py)
requirements.txt        dependencies
```
