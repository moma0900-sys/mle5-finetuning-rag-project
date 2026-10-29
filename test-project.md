# Explanation of the Tests in `test_project.py`

`test_project.py` contains **11 test functions that produce 16 test cases**. Two functions run with multiple inputs.

The tests check individual components of the pipeline.

## 1. Are the supplied datasets valid and separate?

**Test function:** `test_supplied_data_are_valid_and_questions_are_separate`

The test checks that:

- Documents, training examples and evaluation cases are present.
- The same question does not appear in both training and evaluation after normalizing capitalization and whitespace.
- Every declared training source exists.

### Example

```text
Training:   "How long does shipping take?"
Evaluation: "HOW LONG DOES SHIPPING TAKE?"
```

These questions are recognized as identical → the test would fail.

**Limitation:** This comparison does not detect differently worded questions with the same meaning.

## 2. Is a training example without an answer rejected?

**Test function:** `test_training_answer_is_required`

An intentionally incomplete record is submitted:

```python
{
    "id": "example-1",
    "context": "[shipping] Policy",
    "question": "How long does shipping take?",
    "reference_ids": ["shipping"],
    # answer is missing
}
```

**Expected result:** Pydantic raises a `ValidationError`.

In this test, the expected error means the test passes. This is expressed using:

```python
with pytest.raises(ValidationError):
```

## 3. Are duplicate evaluation IDs detected?

**Test function:** `test_duplicate_evaluation_ids_are_rejected`

A small test dataset is intentionally duplicated:

```text
Case 1: id = "test-case"
Case 2: id = "test-case"
```

**Expected result:** `load_data()` raises an error reporting duplicate IDs.

This prevents two evaluation cases from accidentally sharing the same identifier.

## 4. Are unknown sources detected?

**Test function:** `test_unknown_source_is_rejected`

This function runs with two variants:

| Test case | Intentionally introduced problem | Expected result |
|---|---|---|
| Training | An answer cites `[unknown-source]` | Error reporting an unknown source |
| Evaluation | `expected_sources` contains `"unknown-source"` | Error reporting an unknown source |

If only `shipping.md` exists, a record must not reference a nonexistent source.

**Important:** These changes are made in temporary test files. The original datasets are not modified.

## 5. Does chunking preserve text, overlap and metadata?

**Test function:** `test_chunking_preserves_overlap_metadata_and_positions`

### Input

```text
Document shipping: one two three four five six
Document account:  alpha beta
```

### Settings

```python
max_words = 4
overlap_words = 1
```

### Expected output

| Chunk ID | Text | Position |
|---|---|---:|
| `shipping-0` | `one two three four` | 0 |
| `shipping-1` | `four five six` | 1 |
| `account-0` | `alpha beta` | 0 |

The test checks that:

- Each chunk contains at most four words.
- The word `four` overlaps.
- Positions restart at zero for the next document.
- Titles and document IDs are preserved.

## 6. Do Qdrant storage and search preserve metadata?

**Test function:** `test_real_qdrant_index_preserves_payload`

The test deliberately uses simple vectors:

```text
Shipping document: [1, 0]
Password document: [0, 1]
Password question: [0, 1]
```

The question is:

```text
How do I protect my password?
```

**Expected result:** Qdrant returns the password document first because its vector exactly matches the query vector.

The test also checks the returned information:

```text
doc_id:    account
chunk_id:  account-0
title:     Account
position:  0
text:      Never share your password.
score:     approximately 1.0
```

**Qdrant actually runs.** The embedding model is replaced with a simple test implementation.

The test therefore checks the database integration and metadata, not the embedding model’s ability to understand language.

## 7. Are weak results excluded, and does the system abstain without evidence?

**Test function:** `test_weak_hits_produce_abstention_without_loading_a_model`

With a minimum score of `0.30`:

```text
Result A: 0.29 → exclude
Result B: 0.30 → retain
```

Next, only the weak result is used. After filtering, the evidence list is empty:

```python
evidence = []
```

### Expected output

```text
status:     abstained
answer:     Standard insufficient-evidence response
raw_answer: empty
```

No language model is loaded for this test. It checks the early exit before generation.

## 8. Do citation validation and the fallback response work?

**Test function:** `test_citation_and_fallback_validation`

This function runs with five variants.

Only `shipping` is provided as an allowed source:

| Model output | Expected status |
|---|---|
| `One to two days. [shipping]` | `answered` |
| `One to two days. [unknown-source]` | `rejected` |
| `One to two days.` | `rejected` |
| Empty text | `rejected` |
| Exact insufficient-evidence fallback response | `abstained` |

The first case only proves that an allowed source ID was included.

**Limitation:** It does not check whether the cited source actually supports the answer.

## 9. Are ranking metrics and word boundaries evaluated correctly?

**Test function:** `test_ranking_and_word_boundaries`

The expected source is `shipping`. Search returns:

```text
Rank 1: account
Rank 2: shipping
```

### Expected metrics

```text
Recall@3 = 1.0 → the expected source was retrieved
MRR@3    = 0.5 → the first relevant source is at rank 2: 1 / 2
```

The test also checks term matching:

```python
contains_term("noteworthy", "not")           # False
contains_term("It is NOT permitted.", "not") # True
```

This prevents `not` from being incorrectly counted inside another word.

## 10. Is a rejected answer scored correctly?

**Test function:** `test_rejection_does_not_count_as_a_correct_abstention`

### Input

```text
status = rejected
```

The function checks two situations:

| Question | Result | Decision accuracy |
|---|---|---:|
| Supported | `rejected` | 0 |
| Unsupported | `rejected` | 0 |

Rejection because of missing citations, for example, is different from abstaining because the available information is insufficient.

Additionally, a rejected answer to a supported question receives **0 for accepted-answer term coverage**.

## 11. Do both systems receive the same evidence, and can the report be saved?

**Test function:** `test_evaluation_shares_evidence_and_creates_serializable_report`

Simple test objects replace retrieval and the two language models.

### Example workflow

```text
Question → Run retrieval once → Evidence list
                                  ├→ Baseline
                                  └→ Hybrid
```

The test checks that:

- The question is retrieved only once.
- Both generators receive the same evidence list.
- Retrieving the expected source produces recall `1.0`.
- A predefined answer containing all required terms produces relevance `1.0`.
- The report can be saved as JSON and loaded again.
- The case ID and metric definitions are preserved.

No real model answers are generated here. The test checks the evaluation workflow and report format.

## Why are there 16 test cases?

The 16 test cases consist of:

- Nine test functions executed once.
- Two variants for unknown sources.
- Five variants for answer validation.

```text
9 + 2 + 5 = 16
```

Multiple `assert` statements within one function do not automatically count as separate test cases.

## What does “16 passed” mean?

**“16 passed” means that the tested components behave as expected.**

It does not mean that the real language model answers every question correctly.

Answer correctness, whether citations actually support the claims, and reliable abstention on unsupported questions must still be assessed using the actual evaluation outputs.