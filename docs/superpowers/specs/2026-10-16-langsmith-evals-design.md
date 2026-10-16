# LangSmith tracing and offline evaluation for kotaemon

Date: 2026-10-16
Status: draft, awaiting review

## Goal

Measure and improve the answer and retrieval quality of kotaemon's RAG pipeline
using LangSmith as the backend. Users run a fixed dataset of questions with
reference answers through the pipeline from a CLI, get scored experiments in
LangSmith, and can drill from a bad score into per-stage traces.

## Agreed scope

| Question | Decision |
|---|---|
| Primary use | Offline evals (dataset experiments), not online scoring of live traffic |
| Run surface | CLI script only; no Gradio tab |
| Dataset | Questions plus reference answers; optional source file / page |
| Approach | Opt-in tracing shim plus eval CLI (approach A) |
| Judge LLM | kotaemon's default LLM (`llms.get_default()`) |

Non-goals: online evaluators on live chats, feedback (thumbs up/down) sync,
synthetic dataset generation, an Evaluation tab in the app, OpenTelemetry
instrumentation.

## Background (current state)

- No tracing or eval framework exists. "trulens" in the code is only an
  LLM relevance-scoring reranker (`kotaemon/indices/rankings/llm_trulens.py`).
- OpenAI and Azure chat models use the plain `openai` SDK, created in the
  client factories in `libs/kotaemon/kotaemon/llms/chats/openai.py`
  (`ChatOpenAI` ~L299-305, `AzureChatOpenAI` ~L460-466).
- The RAG flow is `FullQAPipeline.stream()` in
  `libs/ktem/ktem/reasoning/simple.py`: rewrite, retrieve, rerank, answer,
  citations.
- The app builds a pipeline in `ChatPage.create_pipeline`
  (`libs/ktem/ktem/pages/chat/__init__.py`) from settings,
  `index.get_retriever_pipelines(settings, user_id, selected)` and
  `reasoning_cls.get_pipeline(settings, state, retrievers)`.

## Design

### 1. Tracing shim

New file `libs/kotaemon/kotaemon/utils/tracing.py`:

- `traceable(**kwargs)`: delegates to `langsmith.traceable` when `langsmith` is
  importable, otherwise returns the function unchanged.
- `maybe_wrap_openai(client)`: returns `langsmith.wrappers.wrap_openai(client)`
  when available, otherwise the client unchanged.

LangSmith itself honours `LANGSMITH_TRACING`, so tracing is opt-in.

Seams:

- `maybe_wrap_openai` in the `ChatOpenAI` and `AzureChatOpenAI` client
  factories (sync and async), giving token, latency and prompt spans.
- `@traceable` on the retrieve and answer steps of `FullQAPipeline`, so a
  low score can be attributed to retrieval or generation.

Open point for the plan: `stream()` is a generator and `generate_relevant_scores`
runs in a separate thread. Verify span nesting and context propagation, and
adjust decorator placement if needed.

### 2. Eval package `libs/ktem/ktem/evals/`

- `dataset.py`: loads CSV or JSONL with columns `question`,
  `reference_answer`, and optional `source_file`, `page`. Upserts a LangSmith
  dataset by name so re-runs do not duplicate examples.
- `runner.py`: builds the pipeline without Gradio by repeating the
  `create_pipeline` steps (load app and index manager, default settings,
  `get_retriever_pipelines`, `reasoning_cls.get_pipeline`). The target function
  drains `stream()` and returns `{answer, contexts, sources}`.
- `evaluators.py`:
  - correctness against the reference answer (LLM judge)
  - groundedness in the retrieved chunks (LLM judge)
  - retrieval hit-rate (no LLM; needs `source_file`)
  - citation validity (no LLM)
- `run.py`: CLI, `python -m ktem.evals.run`, flags `--dataset`, `--index`,
  `--reasoning`, `--experiment-prefix`, `--max-concurrency`. Calls
  `langsmith.evaluate(...)`.

### 3. Configuration and dependencies

- `langsmith` is an optional extra in the relevant `pyproject.toml`.
- `.env.example` documents `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`,
  `LANGSMITH_PROJECT`.
- No behaviour change for users who do not set them.

### 4. Errors and testing

- Missing `langsmith` package or API key: the CLI exits with a clear message.
- A failure on one row is recorded against that row and does not abort the run.
- Retrieval returning no documents yields empty `contexts`; evaluators score
  that row as 0 or skip, rather than raising.
- Tests:
  - shim is a no-op without `langsmith`
  - dataset loader (CSV and JSONL, optional columns)
  - each evaluator, with a stubbed judge LLM
  - CLI smoke test with a mocked LangSmith client (no network)

## Risks

- Headless app bootstrap may depend on Gradio-side state (selectors, user id);
  the plan must confirm which index selector value means "all files".
- Judge-model bias: using the same model that generated the answer can inflate
  scores. A separate judge model is a possible follow-up, not in scope.
- Span nesting through generators and threads (see section 1).
