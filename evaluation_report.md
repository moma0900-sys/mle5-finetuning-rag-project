# Baseline RAG vs Hybrid RAG + LoRA: Evaluation Report

Prepared on 2026-10-29 from the saved reports/evaluation_results.json and the current source. No new training, tuning or model generation was performed to produce this document.

## Decision

**REVISE the hybrid design; do not treat the current assistant as ready for customer use.** The adapter improves accepted-answer coverage and citation formatting on this small benchmark, but still produces incorrect policy claims and fails to abstain reliably on unsupported questions close to the support domain. Beating a poorly performing baseline does not establish sufficient reliability.

The implementation and benchmark are available. The acceptance criterion that unsupported questions abstain is **not fully satisfied**. Rejected answers and valid citation IDs are not treated as proof of correctness.

## Data and Comparison Design

- Six approved Markdown documents, 16 training examples, 14 separate evaluation cases: 10 supported and four unsupported.
- Policy IDs are filename stems retained through chunking, Qdrant storage, retrieval and citation checking.
- Pydantic validates required fields. Duplicate training/evaluation IDs, unknown source IDs and exact question overlap after case/whitespace normalization are rejected.
- Supplied training examples, policies and evaluation labels were not changed. Exact-overlap checks do not establish semantic independence: both datasets share the same policies.
- Every evaluation question is retrieved once. Both generators receive the same selected evidence and native chat prompt. Expected sources, required terms and unsupported labels are used only for scoring.
- Both paths use the same generation limits and validation. The intended model difference is the loaded adapter.

## Reproduction Environment

Windows; Python 3.13.18; torch 2.14.1; transformers 5.20.2; peft 0.21.2; sentence-transformers 6.2.0; qdrant-client 1.20.0; pydantic 2.14.1; pytest 9.1.1. Dependencies are declared in pyproject.toml and locked in uv.lock.

From the repository root:

```powershell
.\.venv\Scripts\python.exe training.py
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

Train once to create the adapter, not before every evaluation. Place the supplied test_project.py in tests/ before running the third command.

## Complete Configuration

### Retrieval

- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Embedding revision: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
- CPU; normalized 384-dimensional embeddings.
- Paragraph-aware chunks: maximum 90 words, overlap 15 for paragraphs needing multiple windows. The six documents produce 16 chunks.
- Qdrant local in-memory collection northstar_hybrid; cosine distance; integer point IDs.
- Payload: chunk_id, doc_id, title, text, position.
- Top-k: 3; retain each hit scoring >= 0.30; the same threshold applies to both systems.
- The in-memory vector index is rebuilt on each run.

### Base Model and Generation

- Base model: HuggingFaceTB/SmolLM2-135M-Instruct
- Revision: 12fd25f77366fa6b3b4b768ec3050bf629380bac
- Native tokenizer chat template with system/user messages and assistant generation marker.
- User content: approved context blocks prefixed with [doc_id], followed by the question.
- System message: Answer only from approved context. Start with the direct answer, then give the next action. Keep the answer concise. Cite every policy claim with source IDs in square brackets. If the context is insufficient, reply exactly: The approved material is insufficient. Escalate for manual review.
- CPU float32; greedy (do_sample=False); max_new_tokens=100; evaluation/inference mode; cache enabled.
- Reaching the generation limit without EOS causes rejection.
- Runtime validation rejects empty answers, missing citation IDs and IDs outside selected evidence. It does not verify claim entailment.
- No selected evidence, or the exact insufficient-material response, yields abstained. Rejected outputs keep rejected status; their public answer is replaced with the fallback and their raw answer is retained.

Fallback:

> The approved material is insufficient. Escalate for manual review.

### Real Local Adapter Training

- Adapter path: C:\Users\moma0\Documents\Weiterbildung\00_GitHub_Repo\Modul5\mle5-finetuning-rag-project\artifacts\support-lora-adapter
- PEFT causal-LM LoRA; targets q_proj and v_proj; rank 8; alpha 16; dropout 0.05; bias none.
- Frozen base; only LoRA parameters optimized; no quantization.
- Trainable: 460,800 / 134,975,808 (0.341%).
- Maximum sequence 512 tokens; overlong examples raise an error. Batch size one, no padding. Prompt labels are -100; answer and end-of-turn tokens are supervised.
- Eight epochs over 16 examples: 128 optimizer updates. AdamW, learning rate 0.0002, default betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01; seed 42; reshuffled examples each epoch.
- CPU float32, up to four threads; no scheduler or gradient accumulation.
- Training-loop time: 88.36 seconds, excluding loading, tokenization and saving.
- Mean per-example epoch losses: 1.4989, 1.2979, 1.0839, 0.9414, 0.8291, 0.7235, 0.6226, 0.5070.
- Adapter and tokenizer are saved separately from the base. Successful reload is evidenced by the actual benchmark. Decreasing loss alone does not establish held-out correctness.

## Metric Rubric

- **recall_at_k**: Mean fraction of expected document IDs found in the raw top-k chunks, before thresholding; supported cases only.
- **mrr_at_k**: Mean reciprocal rank of the first expected source in the raw top-k chunks; missing sources score 0. Supported cases only.
- **answer_relevance**: Required-term coverage in accepted answers; supported cases only. Abstained and rejected supported answers score 0.
- **raw_term_coverage**: Diagnostic required-term coverage before rejection; supported cases only. No generation scores 0.
- **faithfulness**: Citation/forbidden-term proxy, not semantic verification. Supported cases require an accepted answer, nonempty citations limited to supplied evidence and expected sources, and no forbidden phrase. Unsupported cases require abstention.
- **style_compliance**: Mean of three checks: at most 80 words, first sentence at most 30 words, and no maybe/probably/perhaps. Abstentions score 1, even if unnecessary; rejected answers score 0. Does not verify a next action or full style compliance.
- **decision_accuracy**: Answered supported cases or abstained unsupported cases score 1; every rejection scores 0. Not factual correctness.
- **latency_seconds**: Mean answer-call wall time, including tokenization, generation and validation; excludes retrieval, model loading and indexing. Includes abstentions without generation; one untimed warm-up per model precedes the benchmark.

Retrieval, accepted-answer relevance and raw coverage use the 10 supported cases. Faithfulness proxy, style proxy, decision accuracy and mean latency use all 14. Denominators are recorded per metric in the JSON.

Required-term matching is case-insensitive, whitespace-normalized and bounded by word boundaries. It misses some synonyms/negations. Citation faithfulness is a lexical/source-ID proxy, not a claim-entailment test. The style proxy does not verify a useful next action, neutrality or specificity; unnecessary abstentions still score 1 for style, with decision accuracy assessed separately.

## Aggregate Results

| Shared retrieval metric | Result | Cases |
|---|---:|---:|
| Recall@3 | 100.00% | 10 |
| MRR@3 (chunk ranks) | 1.0000 | 10 |

| Answer metric | Baseline RAG | Hybrid RAG + LoRA |
|---|---:|---:|
| answer_relevance | 0.00% | 50.00% |
| raw_term_coverage | 50.00% | 75.00% |
| faithfulness | 14.29% | 57.14% |
| style_compliance | 14.29% | 71.43% |
| decision_accuracy | 14.29% | 64.29% |
| latency_seconds | 1.1093 | 1.2516 |

| Outcome | Baseline RAG | Hybrid RAG + LoRA |
|---|---:|---:|
| answered | 0 | 8 |
| abstained | 2 | 2 |
| rejected | 12 | 4 |

The hybrid accepts seven supported answers and one unsupported answer. Both systems abstain correctly on the two distant out-of-domain questions. Baseline decision accuracy is 2/14; hybrid is 9/14. Acceptance is a format/decision outcome, not proof of correct content.

## Per-Case Comparison

| Case | Expected | Baseline status / relevance | Hybrid status / relevance |
|---|---|---|---|
| eval-refund-ticket | answer | rejected / 0.00% | rejected / 0.00% |
| eval-duplicate-payment | answer | rejected / 0.00% | answered / 100.00% |
| eval-authenticator | answer | rejected / 0.00% | answered / 100.00% |
| eval-passport | answer | rejected / 0.00% | answered / 0.00% |
| eval-packed-address | answer | rejected / 0.00% | rejected / 0.00% |
| eval-standard-shipping | answer | rejected / 0.00% | answered / 100.00% |
| eval-annual-refund | answer | rejected / 0.00% | answered / 50.00% |
| eval-pause | answer | rejected / 0.00% | answered / 100.00% |
| eval-outage-updates | answer | rejected / 0.00% | rejected / 0.00% |
| eval-log-secrets | answer | rejected / 0.00% | answered / 50.00% |
| eval-bank-chargeback | abstain | rejected / N/A | rejected / N/A |
| eval-shipping-compensation | abstain | rejected / N/A | answered / N/A |
| eval-cafeteria | abstain | abstained / N/A | abstained / N/A |
| eval-bicycle | abstain | abstained / N/A | abstained / N/A |

### Evidence and Raw Outputs for Every Case

These measured outputs include incorrect statements and are retained for analysis, not as policy recommendations. The public output for abstained/rejected results is the fallback above.

#### eval-refund-ticket

Question: A digital workshop starts in 12 hours. Can the customer still receive a refund?

Expected sources: refunds.

Raw top-3: refunds-1 (0.7433); refunds-0 (0.4754); refunds-2 (0.4065).

Selected evidence: refunds-1, refunds-0, refunds-2.

**baseline_rag**: rejected; citations: none; latency: 2.0791 s.

Raw output:

> Yes, the customer can still receive a refund. The refund window is 30 calendar days before the workshop starts, and the customer can request a refund within those 30 days. The refund window is 30 days after the order is placed, so the customer can request a refund within that timeframe.

Validation reason: Missing source citations.

**hybrid_rag_lora**: rejected; citations: none; latency: 1.3257 s.

Raw output:

> Yes, the digital workshop tickets are refundable until 24 hours before the workshop starts. Requests received inside the final 24 hours are outside the refund window.

Validation reason: Missing source citations.

#### eval-duplicate-payment

Question: What must support collect for a duplicate payment investigation?

Expected sources: refunds.

Raw top-3: refunds-2 (0.5246); refunds-0 (0.3592); refunds-1 (0.3209).

Selected evidence: refunds-2, refunds-0, refunds-1.

**baseline_rag**: rejected; citations: none; latency: 3.0131 s.

Raw output:

> To support collect for a duplicate payment investigation, the following must be present:
> 
> 1. A valid invoice number and payment reference
> 2. Both payment references
> 3. A complete duplicate-payment request within three business days
> 4. A complete refund within 30 calendar days
> 5. A valid digital workshop ticket within 24 hours before the workshop starts
> 6. A valid digital workshop ticket within 24 hours before the workshop starts
> 
> If any of these must be present,

Validation reason: Answer reached the generation limit without ending.

**hybrid_rag_lora**: answered; citations: refunds; latency: 1.3311 s.

Raw output:

> Collect the invoice number and both payment references to ensure a complete refund. Requests received inside the final 24 hours are outside the refund window. [refunds]

Validation reason: Citation IDs passed validation.

#### eval-authenticator

Question: A customer lost their authenticator. What information and process should support use?

Expected sources: account-security.

Raw top-3: account-security-1 (0.6071); account-security-2 (0.3983); response-style-0 (0.3621).

Selected evidence: account-security-1, account-security-2, response-style-0.

**baseline_rag**: rejected; citations: none; latency: 0.9259 s.

Raw output:

> The customer should use the secure upload portal to reset their password, sign out other sessions, and enable multi-factor authentication.

Validation reason: Missing source citations.

**hybrid_rag_lora**: answered; citations: account-security; latency: 1.9063 s.

Raw output:

> Ask for the registered email address and send a secure recovery link. If a customer lost access to an authenticator, use the secure upload portal to reset the password and enable multi-factor authentication. Follow the process to ensure the authenticator is secure. [account-security]

Validation reason: Citation IDs passed validation.

#### eval-passport

Question: Should a customer paste a passport scan into chat to recover an account?

Expected sources: account-security.

Raw top-3: account-security-1 (0.6059); account-security-2 (0.3627); incidents-1 (0.3593).

Selected evidence: account-security-1, account-security-2, incidents-1.

**baseline_rag**: rejected; citations: none; latency: 0.6604 s.

Raw output:

> Yes, a customer should paste a passport scan into chat to recover an account.

Validation reason: Missing source citations.

**hybrid_rag_lora**: answered; citations: incidents; latency: 0.6943 s.

Raw output:

> Password recovery is the best way to ensure account recovery. [incidents]

Validation reason: Citation IDs passed validation.

#### eval-packed-address

Question: Can support change a delivery address after the warehouse packed the order?

Expected sources: shipping.

Raw top-3: shipping-1 (0.8412); shipping-0 (0.2891); response-style-0 (0.2364).

Selected evidence: shipping-1.

**baseline_rag**: rejected; citations: none; latency: 0.5629 s.

Raw output:

> Yes, support can change a delivery address after the warehouse packs the order.

Validation reason: Missing source citations.

**hybrid_rag_lora**: rejected; citations: none; latency: 0.8688 s.

Raw output:

> Yes, the address can be changed before the warehouse packs the order. After packing, support cannot change the address.

Validation reason: Missing source citations.

#### eval-standard-shipping

Question: What is the usual standard shipping time?

Expected sources: shipping.

Raw top-3: shipping-0 (0.6538); incidents-0 (0.2485); refunds-0 (0.2281).

Selected evidence: shipping-0.

**baseline_rag**: rejected; citations: none; latency: 0.4131 s.

Raw output:

> The standard shipping time is three to five business days.

Validation reason: Missing source citations.

**hybrid_rag_lora**: answered; citations: shipping; latency: 0.8474 s.

Raw output:

> Standard shipping usually takes three to five business days. Express shipping usually takes one to two business days. [shipping]

Validation reason: Citation IDs passed validation.

#### eval-annual-refund

Question: Can an annual subscription receive a partial refund after 45 days?

Expected sources: subscriptions.

Raw top-3: subscriptions-1 (0.7052); refunds-0 (0.5723); subscriptions-0 (0.4839).

Selected evidence: subscriptions-1, refunds-0, subscriptions-0.

**baseline_rag**: rejected; citations: none; latency: 0.6471 s.

Raw output:

> No, an annual subscription cannot receive a partial refund after 45 days.

Validation reason: Missing source citations.

**hybrid_rag_lora**: answered; citations: subscriptions; latency: 1.3901 s.

Raw output:

> No, an annual subscription cannot receive a partial refund after the 30-day refund window. The purchase is fully refundable within 30 calendar days. [subscriptions]

Validation reason: Citation IDs passed validation.

#### eval-pause

Question: How often and for how long can an active subscription be paused?

Expected sources: subscriptions.

Raw top-3: subscriptions-2 (0.8908); subscriptions-0 (0.6718); subscriptions-1 (0.4828).

Selected evidence: subscriptions-2, subscriptions-0, subscriptions-1.

**baseline_rag**: rejected; citations: none; latency: 1.5390 s.

Raw output:

> The approved context states that an active subscription can be paused once per calendar quarter for up to 30 days. The pause begins on the date selected by the customer, and the pause ends on the date selected by the customer.

Validation reason: Missing source citations.

**hybrid_rag_lora**: answered; citations: subscriptions; latency: 1.6252 s.

Raw output:

> [subscriptions] An active subscription may be paused once per calendar quarter for up to 30 days. The pause begins on the date selected by the customer.

Validation reason: Citation IDs passed validation.

#### eval-outage-updates

Question: How frequently should updates be published during a confirmed outage?

Expected sources: incidents.

Raw top-3: incidents-0 (0.7079); subscriptions-2 (0.3666); subscriptions-0 (0.3169).

Selected evidence: incidents-0, subscriptions-2, subscriptions-0.

**baseline_rag**: rejected; citations: none; latency: 2.0613 s.

Raw output:

> The approved context states that updates should be published at least every 30 minutes until recovery. However, the context does not specify the frequency of updates. It is unclear whether the context is providing a specific frequency or a general guideline. Therefore, the answer is not specified.

Validation reason: Missing source citations.

**hybrid_rag_lora**: rejected; citations: none; latency: 1.2834 s.

Raw output:

> Publish status updates at least every 30 minutes until recovery. Direct customers to the public status page for the latest confirmed information.

Validation reason: Missing source citations.

#### eval-log-secrets

Question: May diagnostic logs sent to support contain authentication tokens?

Expected sources: incidents.

Raw top-3: incidents-1 (0.6541); account-security-1 (0.3440); account-security-2 (0.3202).

Selected evidence: incidents-1, account-security-1, account-security-2.

**baseline_rag**: rejected; citations: none; latency: 0.5014 s.

Raw output:

> Yes, diagnostic logs may contain authentication tokens.

Validation reason: Missing source citations.

**hybrid_rag_lora**: answered; citations: incidents; latency: 2.4878 s.

Raw output:

> Yes, diagnostic logs may include password, authentication tokens, and one-time passcodes. For account recovery, ask for the registered email address and send a secure upload portal. For suspicious login, request resetting the password, signing out other sessions, and enabling multi-factor authentication. [incidents]

Validation reason: Citation IDs passed validation.

#### eval-bank-chargeback

Question: How many days will my external bank take to resolve a chargeback for a duplicate payment?

Expected sources: none (unsupported).

Raw top-3: refunds-2 (0.6621); refunds-0 (0.6048); refunds-1 (0.4727).

Selected evidence: refunds-2, refunds-0, refunds-1.

**baseline_rag**: rejected; citations: none; latency: 1.1460 s.

Raw output:

> The answer is 30 calendar days. The process typically takes 30 business days to resolve a chargeback for a duplicate payment.

Validation reason: Missing source citations.

**hybrid_rag_lora**: rejected; citations: none; latency: 1.9874 s.

Raw output:

> The standard purchase is refundable within 30 calendar days when the customer provides the order ID. The digital workshop tickets are outside the refund window. Requests received inside the final 24 hours are outside the refund window.

Validation reason: Missing source citations.

#### eval-shipping-compensation

Question: What compensation amount is guaranteed if express shipping arrives late?

Expected sources: none (unsupported).

Raw top-3: shipping-0 (0.5486); shipping-1 (0.3784); refunds-1 (0.3609).

Selected evidence: shipping-0, shipping-1, refunds-1.

**baseline_rag**: rejected; citations: none; latency: 1.9812 s.

Raw output:

> The provided context does not specify the amount of compensation that can be guaranteed if express shipping arrives late. It only mentions that express shipping can be refunded up to three business days after the workshop starts.

Validation reason: Missing source citations.

**hybrid_rag_lora**: answered; citations: shipping; latency: 1.7745 s.

Raw output:

> Express shipping usually arrives within the 24-hour window. Express shipping usually takes one to two business days. [shipping]

Validation reason: Citation IDs passed validation.

#### eval-cafeteria

Question: Where can I find next week's cafeteria menu?

Expected sources: none (unsupported).

Raw top-3: subscriptions-1 (0.1321); refunds-1 (0.1230); subscriptions-0 (0.1037).

Selected evidence: none.

**baseline_rag**: abstained; citations: none; latency: 0.0000 s.

Raw output:

> No generation: no selected evidence.

Validation reason: No evidence passed the retrieval threshold.

**hybrid_rag_lora**: abstained; citations: none; latency: 0.0000 s.

Raw output:

> No generation: no selected evidence.

Validation reason: No evidence passed the retrieval threshold.

#### eval-bicycle

Question: Can I reserve a bicycle parking space near the office?

Expected sources: none (unsupported).

Raw top-3: shipping-1 (0.2078); refunds-1 (0.1360); subscriptions-2 (0.1170).

Selected evidence: none.

**baseline_rag**: abstained; citations: none; latency: 0.0000 s.

Raw output:

> No generation: no selected evidence.

Validation reason: No evidence passed the retrieval threshold.

**hybrid_rag_lora**: abstained; citations: none; latency: 0.0000 s.

Raw output:

> No generation: no selected evidence.

Validation reason: No evidence passed the retrieval threshold.

## Failure and Fragile-Success Analysis

### 1. Incorrect security advice passes the citation proxy

On eval-log-secrets the hybrid says diagnostic logs may include passwords, authentication tokens and one-time passcodes, citing [incidents]. The actual policy explicitly prohibits these. Runtime validation accepts the supplied citation ID. The forbidden label is 'authentication tokens may be included'; different generated wording escapes that lexical check. The faithfulness proxy therefore scores an incorrect answer positively. This is a measured false positive.

### 2. Domain-related unsupported questions are not reliably declined

eval-shipping-compensation asks for a guaranteed compensation amount absent from the sources. The hybrid instead gives timing, including an unsupported '24-hour window', and cites [shipping]. It is accepted despite not answering the question. For eval-bank-chargeback, both systems generate answers about internal refunds rather than an external bank timeline; citation validation rejects them instead of explicit abstention. Correct abstention on cafeteria/bicycle questions does not establish near-domain answerability.

### 3. Coverage and citation formatting can mask content errors

The baseline standard-shipping reply gives the expected interval but lacks [shipping], so accepted relevance is 0 despite raw coverage 1. The hybrid workshop-refund answer begins 'Yes' for a workshop starting in 12 hours, then states the 24-hour exclusion. It achieves raw term coverage 1 despite the contradictory opening and is rejected for missing citations. Raw coverage distinguishes some formatting failures but is not logical correctness.

### 4. The right source document is not sufficient

Every supported expected document appears at rank 1, yet the hybrid passport answer cites [incidents] and gives generic recovery advice instead of secure-upload guidance. Related and irrelevant passages can coexist in the selected context. Document-ID recall does not establish complete passage coverage or correct model use.

### 5. Rubric and label limits

The required term 'not' is not found inside 'cannot', although the hybrid annual-refund answer expresses a negation; this depresses lexical coverage. The packed-address case requires 'carrier', while the policy qualifies carrier contact as an option after dispatch, not automatically after packing. Labels were preserved and this nuance documented rather than silently changing scores.

## Evidence Threshold

The common 0.30 threshold was fixed before this benchmark as a preliminary hypothesis. Earlier express-shipping versus unrelated-weather demonstrations were smoke checks, not an independent calibration study.

The lowest supported top score is 0.5246; the highest unsupported top score is 0.6621.

Unsupported bank and compensation questions score 0.6621 and 0.5486, overlapping supported scores. A single similarity threshold cannot separate these observed cases: rejecting the bank question by score also removes some supported questions. These are diagnostics of the frozen threshold, not a calibration set for retuning and claiming a fresh held-out result.

## Training, Inference and Maintenance Costs

- Actual optimization-loop time: 88.36 s. A small trainable fraction reduces update/optimizer storage, but the frozen base still participates in computation.
- Adapter weight file: 1,858,776 bytes; tokenizer/configuration files add storage.
- Mean answer-call latency: baseline 1.1093 s; hybrid 1.2516 s on this run. Both averages include two no-generation abstentions.
- Timings include tokenization, generation and validation after one warm-up per model; they exclude loading, retrieval and indexing. Variance, throughput, peak memory and energy were not measured.
- Both model instances are resident during comparison. The adapter needs its matching base, tokenizer and prompt. It is not a standalone model.
- Maintenance adds adapter/data versioning, retraining, validation and checks for stale policy behavior. No hosted service/API cost is involved.

## Tests and Scope

The accompanying tests/test_project.py contains 16 cases after parametrization. It covers required fields, data separation, duplicate evaluation IDs, unknown sources, chunk overlap/metadata, real Qdrant storage/search with controlled vectors, weak-evidence abstention, citations, ranks, word boundaries, rejection semantics, shared evidence and JSON serialization.

The 16 tests were executed against the current project modules from a staged test file outside the repository using the documented virtual environment. An explicit temporary folder was needed because this execution environment could not create pytest's default temporary directory. Copy the supplied file to tests/test_project.py and rerun locally.

Synthetic embedding vectors and fake generators isolate component contracts. Passing those tests does not prove embedding semantics, correctness, LoRA training quality or reliable near-domain abstention. Actual adapter inference is evidenced separately by the saved benchmark.

## Acceptance Status

| Requirement | Status |
|---|---|
| Separate policy/training/evaluation artifacts | Implemented and validated |
| Metadata survives chunking/vector storage | Implemented; component test passes |
| Real vector database | Qdrant local mode |
| Actual local LoRA optimization and saved adapter | Completed; reloaded for benchmark |
| Same retrieval, prompt and cases | Implemented; shared-evidence test passes |
| Unsupported questions abstain | **Partially met: near-domain cases fail** |
| Reject citations outside selected evidence | Implemented; component tests pass |
| Separate retrieval/answer metrics | Saved JSON and this report |
| Failures and adapter trade-offs analyzed | Documented here |
| Tests from documented environment | 16 component tests pass; copy staged file into repository |

## Answers to the Five Project Questions

1. **Which behaviors changed?** Accepted supported answers rise from zero to seven; raw term coverage from 50% to 75%. Citation formatting improves. Unsupported compensation still receives an accepted answer; both distant unsupported cases abstain in both systems.
2. **Evidence use, style, both or neither?** Citation formatting and the limited style proxy improve. Reliable semantic evidence use is not established: accepted security advice contradicts policy. Useful next actions and overall style are not verified by the proxy.
3. **Which retrieval errors limit both?** Similarity retrieves related but insufficient evidence for near-domain unsupported questions. Document recall cannot prove passage completeness or answerability. Both paths inherit the same selected evidence.
4. **Does improvement justify complexity?** Keep the adapter as an educational experiment, but do not promote this hybrid as a reliable support assistant. Accepted contradictions and unsupported answers outweigh the modest training cost.
5. **Which next experiment reduces uncertainty?** Predefine fresh supported paraphrases, near-domain unsupported questions and counterfactual policy changes. Manually score claim support and citation completeness for the frozen baseline and adapter with shared retrieval/prompt. These 14 cases are now inspected; do not tune on them and call the same evaluation untouched.

## Remaining Work

1. Copy the supplied tests and documentation into the repository and rerun the documented test command.
2. Keep current near-domain abstention and contradictory-answer failures visible. Full behavioral acceptance remains incomplete.
3. If full behavioral acceptance is required before submission, address answerability and claim support using independent development questions, then evaluate on untouched cases. Merely appending citations or relabeling rejections would not fix the failures.
