# Northstar: Baseline RAG vs Hybrid RAG + LoRA

A local comparison of a frozen base language model and the same model with a genuinely trained LoRA adapter. Both systems receive the same retrieved policy evidence and question prompt.

**Current conclusion: revise the hybrid; do not deploy this version as a reliable support assistant.** Citation-format and accepted-answer metrics improve, but incorrect accepted answers and near-domain abstention failures remain. See [the decision report](reports/evaluation_report.md).

## Architecture

- `data_loading.py`: validates six policies, 16 supervised examples and 14 evaluation cases.
- `retrieval.py`: paragraph-aware chunking, local MiniLM embeddings and in-memory Qdrant.
- `prompts.py`: shared native chat prompt and fallback response.
- `training.py`: real local LoRA optimization and saved adapter.
- `generation.py`: baseline/adapter loading, generation, citation validation and status.
- `evaluation.py`: shared retrieval metrics and separate answer metrics.
- `main.py`: runs the benchmark and writes the JSON trace.

Application modules live in the repository root. `solution/` is the supplied reference implementation, not the application described here.

## Setup

Use Python 3.13.17 or a newer 3.13 patch release and uv >=0.13.5, as required by the repository configuration. From the project root:

```powershell
uv sync
```

This creates `.venv` and installs the locked dependencies, including the default development group with pytest. Select that interpreter in the editor.

The checked environment used Python 3.13.18, torch 2.14.1, transformers 5.20.2, peft 0.21.2, sentence-transformers 6.2.0, qdrant-client 1.20.0, pydantic 2.14.1 and pytest 9.1.1.

Initial model loading requires access to public Hugging Face models; cached models can run offline. This implementation requires no hosted API, Qdrant server, API key or `.env` configuration. Its settings are Python constants. The template's `.env` workflow applies to the reference solution.

If uv is not recognized, repair/install it before creating a fresh environment. If the project environment already exists with dependencies, use its Python directly below. Run commands at the PowerShell prompt, not Python's `>>>` prompt.

## Train Once

```powershell
.\.venv\Scripts\python.exe training.py
```

Artifacts:

```text
artifacts/support-lora-adapter/
    adapter_config.json
    adapter_model.safetensors
    training_summary.json
    tokenizer.json
    tokenizer_config.json
    chat_template.jinja
```

Base: `HuggingFaceTB/SmolLM2-135M-Instruct`, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. CPU float32; eight epochs; batch one; learning rate 0.0002. LoRA targets q_proj/v_proj, rank 8, alpha 16, dropout 0.05. Trainable parameters: 460,800 of 134,975,808.

Prompt tokens are masked from loss; answer and end-of-turn tokens are supervised. A real backward/optimizer loop updates the adapter while the base remains frozen. The saved run took 88.36 seconds for the loop and reduced mean training loss from 1.4989 to 0.5070. This alone does not establish useful generalization.

## Compare and Reload the Adapter

```powershell
.\.venv\Scripts\python.exe main.py
```

This reloads the adapter for hybrid inference and does not retrain. Both model instances run on CPU. Each question is retrieved once, and the same selected evidence is passed to both.

Output: `reports/evaluation_results.json`, containing configuration, metric definitions/counts, retrieved chunks/scores, selected evidence, raw/public answers, statuses, reasons and aggregate/per-case metrics.

The Markdown report is a reviewed interpretation of the JSON. `main.py` does not regenerate it automatically; update it when settings or measured results change.

## Test

Place the supplied file in `tests/test_project.py` and run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: **16 passed**. The explicit `tests` path selects this implementation's component tests, not the separate `solution/tests` suite.

Tests cover validation failures, chunk metadata/overlap, real Qdrant search with controlled synthetic vectors, weak-evidence abstention, citation rules, metrics, shared evidence and JSON serialization. They do not download models or use external APIs. ML imports still require the installed dependencies and may take several seconds.

Passing component tests does not establish semantic answer correctness or resolve the real benchmark's near-domain unsupported failures.

## Status

- `answered`: nonempty generated answer with citation IDs drawn from supplied evidence.
- `abstained`: no evidence passes the threshold, or the model returns the exact fallback.
- `rejected`: missing/unknown citations, empty answer or generation-limit truncation.

Abstained/rejected public responses use:

> The approved material is insufficient. Escalate for manual review.

The original raw answer and distinct status remain in the trace. Rejections are not counted as correct abstentions. A valid citation ID does not establish that the cited text supports every claim.

## Configuration and Metrics

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; normalized, 384 dimensions.
- Chunks: up to 90 words, overlap 15 for long paragraphs.
- Qdrant local in-memory index, cosine similarity; rebuilt each run.
- Retrieval top 3; retain scores >=0.30.
- Shared native-chat prompt; greedy generation, at most 100 new tokens.
- Recall/MRR use original top-3 chunk ranks before thresholding and the 10 supported cases.
- Accepted relevance uses required-term coverage on 10 supported cases; blocked answers score zero. Raw coverage is diagnostic.
- Faithfulness is a citation/forbidden-phrase proxy over 14 cases, not semantic verification.
- Style checks reply length, first-sentence length and selected hedges; abstentions score one and rejections zero. Useful next actions are not verified.
- Decision accuracy rewards answered supported questions or abstained unsupported questions; all rejections score zero.
- Mean answer-call latency follows one warm-up per model and excludes loading/retrieval/indexing. It includes abstentions without generation.

## Saved Results

| Metric | Baseline RAG | Hybrid RAG + LoRA |
|---|---:|---:|
| Shared recall@3 | 100% | 100% |
| Shared MRR@3 | 1.0000 | 1.0000 |
| Accepted-answer term coverage | 0% | 50% |
| Raw term coverage | 50% | 75% |
| Faithfulness proxy | 14.29% | 57.14% |
| Style proxy | 14.29% | 71.43% |
| Decision accuracy | 14.29% | 64.29% |
| Mean answer-call latency | 1.1093 s | 1.2516 s |

Baseline: 12 rejected, two abstained. Hybrid: eight answered, four rejected, two abstained.

Some accepted hybrid answers contradict policy; the lexical faithfulness proxy misses some of them. Supported and near-domain unsupported similarity scores overlap, so simply raising the threshold cannot separate all observed cases.

The 0.30 threshold is preliminary and was fixed before the benchmark. Do not tune on these inspected cases and describe the same scores as a new held-out evaluation.

## Artifacts and Scope

Keep generated adapters, environment files and JSON traces out of Git. The supplied `.gitignore` excludes `artifacts/` and `.env`; add `reports/*.json` for the application's traces. Commit source, tests and the reviewed `reports/evaluation_report.md`.

Policies, training examples and evaluation labels remain separate and unchanged. No optional hosted judge, GUI or additional model condition is included.

See [project-description.md](project-description.md) for the assignment and [reports/evaluation_report.md](reports/evaluation_report.md) for all cases, failures, threshold analysis, costs, acceptance gaps and the five project questions.
