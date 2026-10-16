# LangSmith tracing and offline evaluation for kotaemon

Date: 2026-10-16
Status: v3, approved; corrected against the code while writing the plan

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
| Reasoning pipeline | `simple` (`FullQAPipeline`) only in v1 |
| Headless file selection | Move file-id resolution out of the UI class (option a) |

Non-goals: online evaluators on live chats, feedback (thumbs up/down) sync,
synthetic dataset generation, an Evaluation tab in the app, OpenTelemetry
instrumentation, a separate judge model, evals for `FullDecomposeQAPipeline`,
ReAct or ReWOO, upgrading LangChain / LangSmith.

## Background (verified current state)

### Dependencies

- `langsmith` 0.1.147 is already installed transitively: `langchain` 0.2.15
  requires `langchain-core<0.3`, and `langchain-core` 0.2.43 requires
  `langsmith>=0.1.112,<0.2`. Moving to `langsmith>=0.2` would mean a LangChain
  0.3 migration, so this design targets the 0.1.147 API. Everything it uses
  exists there: `traceable` (with `reduce_fn`, `process_inputs`,
  `process_outputs`, generator support), `wrappers.wrap_openai`, `evaluate()`
  with `(inputs, outputs, reference_outputs)` evaluators,
  `utils.ContextThreadPoolExecutor`, `Client.create_examples(ids=...)`,
  `Client.update_examples`, `Client.list_examples`, `Client.delete_example`.
  There is no upsert for examples in 0.1.147.
- `LANGSMITH_TRACING=true` is honoured (`langsmith/utils.py:95-111`, which also
  accepts `LANGCHAIN_TRACING_V2`).

### Models

- OpenAI-compatible chat models (`ChatOpenAI`, which also serves Ollama, Groq
  and Mistral through `base_url`, plus `AzureChatOpenAI`) use the plain
  `openai` SDK. A new client is built on every call in `prepare_client`
  (`libs/kotaemon/kotaemon/llms/chats/openai.py:285-305` and `:444-466`).
  `StructuredOutputChatOpenAI` inherits `ChatOpenAI.prepare_client`.
- LangChain-based chat models (`LCGeminiChat`, the fallback default in
  `flowsettings.py` when no OpenAI key is set; `LCAnthropicChat`,
  `LCCohereChat`, `LCOllamaChat`) are traced by `langchain-core` itself.
  `langchain_core/tracers/context.py:146-149` defers to
  `langsmith.utils.tracing_is_enabled()`, which is true when the env var is set
  **or** a parent `@traceable` span is active. They need no wrapper.
- `llms.get(...)` returns shared model instances (`ktem/llms/manager.py:96-155`).
  The app already runs chat requests concurrently on them (`concurrency_limit=20`,
  `ktem/pages/chat/__init__.py:456`).

### RAG flow (`FullQAPipeline.stream`, `libs/ktem/ktem/reasoning/simple.py:281-331`)

1. `retrieve(message, history)` returns `(docs, infos)`. `infos` are HTML
   `Document`s for the UI panel.
2. `evidence_pipeline(docs)` builds the context string that the answer LLM sees.
   It is deterministic and calls no LLM (`kotaemon/indices/qa/format_context.py:28-114`).
   It labels each chunk with `metadata["file_name"]` and `metadata["page_label"]`
   and trims to `max_context_length` tokens.
3. If the LLM relevance scorer is on (`use_llm_reranking`, default true unless
   `USE_LOW_LLM_REQUESTS`), a `threading.Thread` runs
   `LLMTrulensScoring` (`simple.py:297-306`). It does **not** drop documents: it
   returns a re-sorted copy and adds `metadata["llm_trulens_score"]`
   (`kotaemon/indices/rankings/llm_trulens.py:113-182`). The evidence is built
   before it starts, so it only affects the UI (ordering and the low-relevance
   warning).
4. `answering_pipeline.stream(...)` yields chat chunks and **returns** the
   answer `Document`. `stream()` itself returns that answer. Raw docs and the
   evidence string are local variables and are never yielded.
5. `show_citations_and_addons` yields HTML for citations, mindmap and plots.

### Citation modes (`reasoning.options.simple.highlight_citation`)

| Mode | Answer pipeline | `answer.text` | `answer.metadata["citation"]` |
|---|---|---|---|
| `highlight` (default) | `AnswerWithContextPipeline` (`kotaemon/indices/qa/citation_qa.py`) | raw LLM output, may contain `<think>…</think>` | `CitationPipeline` output with `.evidences` (list of quotes) from an LLM call in a separate thread. `None` if that call misses `CITATION_TIMEOUT`. |
| `inline` | `AnswerWithInlineCitation` (`citation_qa_inline.py`) | text after `FINAL ANSWER`, with raw `【n】` markers; may include a trailing `CITATION LIST` part; **empty** when there is no evidence or the model never emits `FINAL ANSWER` | list of `InlineEvidence(idx, start_phrase, end_phrase)` parsed from the output |
| `off` | `AnswerWithContextPipeline` with `enable_citation=False` | raw LLM output | `None` |

The UI matches citations against docs with `find_text` (highlight) and
`find_start_end_phrase` (inline), both in `kotaemon/indices/qa/utils.py`.

### Threads on the chat path

contextvars do not propagate into `threading.Thread` or a stdlib
`ThreadPoolExecutor` (Python 3.10), so model calls made in these threads
become orphan root traces, or go untraced:

| Site | Work done in the thread |
|---|---|
| `ktem/reasoning/simple.py:303` | relevance scorer (LLM) |
| `kotaemon/indices/rankings/llm_trulens.py:123` | per-doc scorer LLM calls (pool inside the thread above) |
| `kotaemon/indices/qa/citation_qa.py:224` | citation LLM call (highlight mode) |
| `kotaemon/indices/qa/citation_qa.py:228`, `citation_qa_inline.py:224` | mindmap LLM call (off by default) |

`kotaemon/indices/vectorindex.py:218-219` also starts threads, but they only
query the stores; the embedding call happens on the caller's thread (`:188`).
theflow keeps no state in contextvars, so copying context into these threads
does not change pipeline behaviour.

### Index selection and users

- The chat page passes each index a selector value `[mode, file_ids, user_id]`,
  where `mode` is one of `"all"`, `"select"` or `"disabled"`.
- File-id resolution lives on the UI class `FileSelector.get_selected_ids`
  (`ktem/index/file/ui.py:1800-1821`), and `get_retriever_pipelines` reaches it
  through `self._selector_ui` (`ktem/index/file/index.py:473`, also
  `graph/graph_index.py:28`, `graph/light_graph_index.py:64`,
  `graph/nano_graph_index.py:64`).
- `_selector_ui` stays `None` until the chat page builds its UI, and
  `index.selector` is only set by `ChatPage`. **Reproduced:** with `BaseApp()`
  built headless, `get_retriever_pipelines` raises
  `AttributeError: 'NoneType' object has no attribute 'get_selected_ids'`.
- With `user_id=None`, `get_selected_ids` returns `[]`.
  `DocumentRetrievalPipeline.run` then skips retrieval and returns `[]`
  (`ktem/index/file/pipelines.py:142-144`). The answer LLM is then prompted with
  the bare question (`citation_qa.py:201-204`). Nothing errors: the eval would
  silently measure the LLM without RAG.
- User ids are strings (`ktem/db/base_models.py:58`). `Source.user` is a
  string column. Private indices (the default) only see the uploader's files.
  With user management off, the app uses the id `"default"`.
- Each user's saved settings live in the `Settings` table
  (`ktem/pages/settings.py:379-386`) and override the app defaults at login.

### LangSmith `evaluate()` behaviour (0.1.147, `langsmith/evaluation/_runner.py`)

- Forces tracing on for the target (`tracing_context(enabled=True)`, `:1661`),
  so evals do not need `LANGSMITH_TRACING`.
- Catches and logs target exceptions per example (`:1674-1679`). That row's run
  carries the error and evaluators see missing outputs.
- `max_concurrency=None` means the executor default worker count, not 1 (`:1357`).
- Runs evaluators under the `"evaluators"` project, so judge calls are traced there.

## Design

### 1. Tracing shim

New module `libs/kotaemon/kotaemon/utils/tracing.py` (creates the
`kotaemon.utils` package):

- `traceable(**kwargs)`: delegates to `langsmith.traceable`. Falls back to the
  undecorated function if `langsmith` cannot be imported. This is defensive
  only, because in practice `langsmith` is always present.
- `maybe_wrap_openai(client)`: returns `langsmith.wrappers.wrap_openai(client)`,
  or the client unchanged if `langsmith` cannot be imported. The wrapper is a
  pass-through when tracing is disabled.
- `run_in_context(fn)`: returns a callable that runs `fn` inside
  `contextvars.copy_context()` taken at wrap time. It uses only the stdlib. Call
  it once per thread or per `submit`, because one `Context` cannot be entered by
  two threads at once.
- `reduce_answer(items)`: the `reduce_fn` for streamed pipelines. langsmith
  appends a generator's return value to the collected items
  (`run_helpers.py:651-652`), so the final answer `Document` (no `channel`) is
  the last item. Returns `{"answer": text}`, falling back to the accumulated
  `chat` channel text (a `None` content resets it, as in the UI) when the
  generator raised before returning.
- `export_langsmith_env(search_path=None)`: copies `LANGSMITH_*` /
  `LANGCHAIN_*` tracing variables from `.env` into `os.environ` when not
  already set. kotaemon reads `.env` through python-decouple, which never
  populates `os.environ`, and langsmith only reads `os.environ`. Called from
  `BaseApp.__init__` and at the start of the eval CLI.

Tracing stays opt-in. Nothing is traced in the app unless `LANGSMITH_TRACING`
is set (eval runs are always traced, see Background).

Seams:

| Seam | Change |
|---|---|
| `ChatOpenAI.prepare_client`, `AzureChatOpenAI.prepare_client` (sync and async) | wrap the returned client with `maybe_wrap_openai` |
| `FullQAPipeline.stream` | `@traceable(name="kotaemon.qa", run_type="chain")` root span, with `reduce_fn` keeping only the final answer text, so one chat or one eval row is one trace |
| `FullQAPipeline.retrieve` | `@traceable(run_type="retriever")`, with `process_outputs` logging each doc as `{page_content, metadata: {file_name, page_label, doc_id, score}}` and dropping the HTML `infos` |
| `AnswerWithContextPipeline.stream`, `AnswerWithInlineCitation.stream` | `@traceable(run_type="chain")`, with `reduce_fn` keeping only the final answer text |
| the five thread sites listed in Background | `target=run_in_context(fn)` / `executor.submit(run_in_context(fn))` |

Verified in `langsmith/run_helpers.py`:
- The generator wrapper preserves the generator's `return` value (`:645-663`),
  so `answer = yield from self.answering_pipeline.stream(...)` keeps working.
- Each `next()` runs inside the span's saved context (`:1538`), so nesting holds
  even if the consumer resumes the generator on another thread, as Gradio does.
- `self` and `cls` are dropped from logged inputs (`:1464`), so pipeline objects
  holding API keys are not serialised.

Known gap: token counts for streamed OpenAI calls need
`stream_options={"include_usage": true}`, which some OpenAI-compatible servers
reject. Not in scope.

### 2. Headless file selection (refactor, option a)

- Add `FileIndex.get_selected_ids(selected) -> list[str]` holding the logic now
  in `FileSelector.get_selected_ids` (`ui.py:1800-1821`): mode handling, the
  `user_id is None` guard, and the private-index filter. It has no Gradio
  dependency.
- `FileSelector.get_selected_ids` delegates to `self._index.get_selected_ids`.
- Replace `self._selector_ui.get_selected_ids(selected)` with
  `self.get_selected_ids(selected)` in `FileIndex.get_retriever_pipelines` and
  the three graph indices. `KnowledgeNetworkFileIndex` and the graph indices
  inherit from `FileIndex`, so they get the method.
- No behaviour change in the app. `ChatPage._has_selected_files` still calls the
  UI method, which now delegates.

Note: building the selector UI headless (`get_selector_component_ui()`) also
works on gradio 4.39. This refactor is preferred because it keeps Gradio
components out of a CLI and separates data logic from UI.

### 3. Eval package `libs/ktem/ktem/evals/`

`dataset.py`, `cleaning.py` and `evaluators.py` must not import `ktem.app` or
theflow settings, so they can be tested without bootstrapping the app.

#### `dataset.py`

- Loads CSV or JSONL with `question` and `reference_answer`, and optional
  `source_file` and `page`. Rejects rows with an empty question, and duplicate
  questions.
- Example mapping:
  - `inputs = {question}`
  - `outputs = {reference_answer, source_file?, page?}`. These go in outputs so
    evaluators receive them as `reference_outputs`.
- Example id: `uuid5(namespace, dataset_name + "\n" + question.strip())`.
- Sync with LangSmith (0.1.147 has no example upsert):
  1. create the dataset if missing
  2. `list_examples`
  3. `create_examples(ids=...)` for new ids
  4. `update_examples` for ids whose outputs changed
  5. `delete_example` for ids no longer in the file, only with `--prune`

#### `bootstrap.py`

1. Build `BaseApp()` headless. Verified: it constructs settings, reasonings and
   the index manager without building Gradio Blocks.
2. Resolve `--user` (username) to its `User.id` through the DB. Use `"default"`
   when user management is off. Exit with an error if the user is not found.
3. Settings: app defaults (`default_settings.flatten()`), overlaid key by key
   with that user's saved `Settings` row so newer default keys survive, then
   the CLI overrides:
   - `--citation` sets `reasoning.options.simple.highlight_citation`.
   - `index.options.{index.id}.use_llm_reranking` (a per-index setting) is set
     to false unless `--llm-scoring` is given. The scorer only reorders docs for
     display (see Background), so it changes no metric but costs one LLM call
     per retrieved doc.
4. Select `--index` (name or id; default: the first `FileIndex`). Only this
   index contributes retrievers. The selector value is `["all", [], user_id]`.
5. Fail fast if `index.get_selected_ids(["all", [], user_id])` is empty: "user X
   has no files in index Y".

#### `runner.py`

- `EvalQAPipeline(FullQAPipeline)` overrides `retrieve` to record the returned
  `docs` before handing them back. Build it with
  `EvalQAPipeline.get_pipeline(settings, state, retrievers)`. That classmethod
  instantiates `cls`, and the subclass keeps `get_info()["id"] == "simple"`, so
  the settings prefixes are unchanged. Capture goes into `self._eval_docs`:
  theflow's `__setattr__` passes names starting with `_` straight through
  (`theflow/base.py:1155-1157`).
- One pipeline per example, including fresh retrievers, as the app does per
  request. The shared LLM instances are shared in the app too.
- The target receives `inputs`:
  1. Drain `pipeline.stream(question, conv_id=f"eval-{uuid4()}", history=[])`
     and take the generator's return value as `answer`.
  2. `evidence = pipeline.evidence_pipeline(docs).content[1]`. This is
     deterministic and reproduces exactly what the answer LLM saw.
  3. Compute the citation facts using the UI's own matchers:
     - highlight: each quote in `answer.metadata["citation"].evidences` is
       matched if `find_text(quote, doc.text)` hits any retrieved doc.
     - inline: each distinct `【n】` marker in `answer.text` is valid if `n` is
       in the parsed citation list and that entry's start and end phrases match
       a retrieved doc via `find_start_end_phrase`.
     - `find_start_end_phrase` crashes on a `None` phrase (verified:
       `AttributeError`), and `answer_to_citations` can emit an entry with one
       phrase missing, which also crashes the app's own citation display. Fix
       it to skip a `None` phrase, as its loop already intends.
  4. Return plain JSON only:
     ```
     {
       "answer":     cleaned answer text,
       "raw_answer": answer.text,
       "contexts":   [{doc_id, file_name, page_label, type, text}, ...],  # retrieve() order
       "evidence":   str,
       "citations":  {mode, claimed, matched}
     }
     ```

#### `cleaning.py`

- Removes `<think>…</think>` blocks.
- In inline mode, also removes `【n】` / `【n, m】` markers and everything from
  `CITATION LIST` onwards.

#### `evaluators.py`

Evaluators are pure functions of `outputs` and `reference_outputs`. Missing
`outputs` (the target raised) returns score `None` with a comment.

| Key | Kind | Scoring | Skipped (`None`) when |
|---|---|---|---|
| `correctness` | LLM judge | label `correct` / `partially_correct` / `incorrect` → 1 / 0.5 / 0, comparing the cleaned answer to `reference_answer` | answer empty → 0, not skipped |
| `groundedness` | LLM judge | label `grounded` / `partially_grounded` / `ungrounded` → 1 / 0.5 / 0, comparing the cleaned answer to `evidence` | `evidence` empty, or answer empty |
| `retrieval_hit` | no LLM | 1 if any non-thumbnail context matches: `file_name` basename equal to `source_file` (case-insensitive) and, when `page` is given, `str(page_label) == str(page)`; else 0 | no `source_file` |
| `reciprocal_rank` | no LLM | `1 / rank` of the first matching non-thumbnail context (1-based), 0 when none matches; the mean is MRR | no `source_file` |
| `citation_validity` | no LLM | `matched / claimed` | mode `off`, or `claimed == 0` |

- Judge prompts ask for a label plus a one-line reason. Parse the label with a
  tolerant regex rather than structured output, since the default LLM may be
  Gemini or Ollama. If parsing fails, return score `None` and put the raw judge
  text in the comment.
- `page_label` is only set by page-aware loaders (PDF, DOCX, HTML, Excel, OCR
  and similar). A dataset `page` for a plain-text file never matches. The loader
  documents this.
- PDF indexing also stores one "Page thumbnail" document per page
  (`metadata["type"] == "thumbnail"`, with `file_name` and `page_label`), and
  vector search can return it. `VectorRetrieval` relabels such docs as
  `type == "image"` (`kotaemon/indices/vectorindex.py:259-261`), keeping the
  text "Page thumbnail". Retrieval evaluators skip a context whose type is
  `thumbnail` or whose text is "Page thumbnail", or a contentless thumbnail
  would count as a hit. Image docs linked to a chunk keep the chunk text and
  still count.
- For PDFs, the default reader (`PDFThumbnailReader`,
  `kotaemon/loaders/pdf_loader.py:81-93`) sets `page_label` to the PDF's own
  page label from pypdf, not the physical page position. It also **drops**
  every page whose label is not an integer, such as roman-numeral front matter.
  Verified on a 4-page PDF labelled `i, ii, 1, 2`: only the last two pages were
  indexed, as `'1'` and `'2'`. So a dataset `page` must be the page label, and
  questions about pages with non-integer labels can never be retrieved. Most
  PDFs define no labels, and then the label is the position plus 1. The
  dataset docs state this. Sources with 0-based page indices, such as
  FinanceBench, need converting with `pypdf.PdfReader(f).page_labels[i]`.

#### `run.py`

CLI: `python -m ktem.evals.run`

| Flag | Meaning |
|---|---|
| `--dataset PATH` | CSV or JSONL file (required) |
| `--dataset-name NAME` | LangSmith dataset name (default: file stem) |
| `--user USERNAME` | whose files and settings to use (required when user management is on) |
| `--index NAME_OR_ID` | index to evaluate (default: first `FileIndex`) |
| `--citation {highlight,inline,off}` | default: the user's setting |
| `--llm-scoring` | keep the LLM relevance scorer on |
| `--experiment-prefix STR` | passed to `evaluate` |
| `--max-concurrency N` | default 2; always passed explicitly, never `None` |
| `--prune` | delete LangSmith examples no longer in the file |

- Calls `langsmith.evaluation.evaluate(target, data=dataset_name,
  evaluators=[...], experiment_prefix=..., max_concurrency=..., metadata=...,
  client=...)`. 0.1.147 has no top-level `langsmith.evaluate`. `evaluate` prints
  the experiment URL itself when it starts.
- Experiment metadata: LLM name, index, citation mode, retrieval mode, top-k,
  reranking on/off, git SHA. Experiments can then be compared in LangSmith.
- Prints the experiment URL and a per-metric mean at the end.

### 4. Configuration and dependencies

- Declare `langsmith>=0.1.147,<0.2` as a direct dependency of `ktem`, since
  `ktem.evals` imports it. This is the version already installed, so nothing new
  is downloaded. Regenerate `uv.lock` (CI runs `uv sync --frozen`). The
  `kotaemon` shim imports it lazily and needs no declaration.
- `.env.example` gains a commented LangSmith block:
  - `LANGSMITH_API_KEY`: required for evals.
  - `LANGSMITH_TRACING=true`: only needed to trace the live app; evals are
    traced regardless.
  - `LANGSMITH_PROJECT`: project for live-app traces.
  - `LANGSMITH_ENDPOINT`: for EU or self-hosted instances.
  - A note that prompts, retrieved chunks and answers are sent to LangSmith.
  These values reach langsmith through `export_langsmith_env()` (see the shim),
  since nothing else copies `.env` into `os.environ`.
- No behaviour change for users who set none of these.

### 5. Errors and testing

Errors:
- Missing `LANGSMITH_API_KEY`: exit with a clear message before any work.
- Unknown user, unknown index, or zero files for that user in that index: exit
  with a clear message.
- Target failure on a row: recorded on that row by `evaluate()`. Evaluators
  return `None` for it and the run continues.
- Judge parse failure: score `None`, raw judge text in the comment.

Tests. `kotaemon` tests go in `libs/kotaemon/tests/`. `ktem` tests go in the
existing `libs/ktem/ktem_tests/` (excluded from the package build), and
`.github/workflows/unit-test.yaml` gains a step that runs them from the repo
root, since CI currently only runs `libs/kotaemon`. That step skips the legacy
`ktem_tests/test_qa.py`, which no longer imports (`No module named 'index'`).
The other 27 ktem tests pass today.

- Shim:
  - no-op when `langsmith` import fails
  - `run_in_context` carries a `ContextVar` into a `Thread` and into a
    `ThreadPoolExecutor`
  - a `@traceable` generator keeps its `return` value under `yield from`
- `FileIndex.get_selected_ids`: `all`, `select`, `disabled`, `user_id=None`,
  and private vs public index, with no UI built.
- Dataset: CSV and JSONL, optional columns, duplicate and empty-question
  rejection, deterministic ids, and the create / update / prune diff against a
  mocked client.
- Cleaning: think tags, inline markers, `CITATION LIST` tail.
- Citation facts: highlight and inline fixtures, including `citation=None`.
- Each evaluator with a stubbed judge, including parse failure, missing outputs
  and empty evidence.
- CLI smoke test with a mocked LangSmith client and a stub pipeline (no network).

Manual end-to-end check (not in CI): `eval_data/uda_wiki10/` (gitignored) holds
10 Wikipedia PDFs on different topics from the UDA benchmark's `wiki_nq`
subset, plus `dataset.jsonl` with 25 questions. 22 rows carry a `page` checked
against `PDFThumbnailReader` output. Upload the PDFs to File Collection as one
user, then run the CLI against them with a real LangSmith key. The articles are
a 2017–2018 snapshot. One row ("most recent justice") has a document answer
that differs from what an LLM knows today, which shows whether answers come
from retrieval.

## Risks

- **Judge bias.** The same model that generated the answer may inflate scores.
  A separate judge model is a possible follow-up.
- **Cost.** Each row costs:
  - one answer call
  - one citation call (highlight mode)
  - two judge calls
  - one scorer call per retrieved doc, only if `--llm-scoring` is given

  Concurrency above 2 may hit provider rate limits.
- **Live-app tracing volume.** With `LANGSMITH_TRACING=true`, every chat is
  traced, including retrieved document text.
- **LangSmith 0.1.x.** Newer LangSmith features (for example, newer evaluator
  helpers) are unavailable until LangChain is upgraded to 0.3.
- **Settings drift.** Results depend on the user's saved settings at run time.
  They are recorded in the experiment metadata so runs can be compared.
