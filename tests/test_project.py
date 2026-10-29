import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import data_loading
import retrieval
from data_loading import EvaluationCase, TrainingExample, load_data
from evaluation import (
    answer_metrics, contains_term, evaluate, retrieval_metrics,
)
from generation import AnswerGenerator, validate_answer
from prompts import FALLBACK_ANSWER
from retrieval import MIN_SCORE, Retriever, SearchHit, chunk_documents, select_evidence


def make_hit(source="shipping", score=0.8):
    return SearchHit(
        chunk_id=f"{source}-0",
        doc_id=source,
        title=source.title(),
        text="Express shipping takes one to two business days.",
        position=0,
        score=score,
    )


def make_case(unsupported=False):
    return EvaluationCase(
        id="test-case",
        question="How long does express shipping take?",
        expected_sources=[] if unsupported else ["shipping"],
        required_terms=[] if unsupported else ["one to two"],
        forbidden_terms=[],
        should_abstain=unsupported,
    )


@pytest.fixture
def small_dataset(tmp_path, monkeypatch):
    for folder in ("knowledge_base", "training", "evaluation"):
        (tmp_path / "data" / folder).mkdir(parents=True)
    (tmp_path / "data/knowledge_base/shipping.md").write_text(
        "# Shipping\n\nExpress shipping takes one to two business days.",
        encoding="utf-8",
    )
    example = {
        "id": "train-example",
        "context": "[shipping] Express shipping takes one to two business days.",
        "question": "Tell me about express delivery.",
        "answer": "One to two business days. [shipping]",
        "reference_ids": ["shipping"],
    }
    (tmp_path / "data/training/instruction_examples.jsonl").write_text(
        json.dumps(example), encoding="utf-8",
    )
    path = tmp_path / "data/evaluation/cases.json"
    path.write_text(json.dumps([make_case().model_dump()]), encoding="utf-8")
    monkeypatch.setattr(data_loading, "ROOT", tmp_path)
    return tmp_path


def test_supplied_data_are_valid_and_questions_are_separate():
    documents, training, cases = load_data()
    assert documents and training and cases
    normalized = lambda text: " ".join(text.casefold().split())
    assert not (
        {normalized(item.question) for item in training}
        & {normalized(item.question) for item in cases}
    )
    assert all(
        set(item.reference_ids) <= set(documents) for item in training
    )


def test_training_answer_is_required():
    with pytest.raises(ValidationError):
        TrainingExample.model_validate({
            "id": "missing-answer", "context": "[shipping] Policy",
            "question": "A question", "reference_ids": ["shipping"],
        })


def test_duplicate_evaluation_ids_are_rejected(small_dataset):
    path = small_dataset / "data/evaluation/cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(cases + cases), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate IDs"):
        load_data()


@pytest.mark.parametrize("layer", ["training", "evaluation"])
def test_unknown_source_is_rejected(small_dataset, layer):
    if layer == "training":
        path = small_dataset / "data/training/instruction_examples.jsonl"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["answer"] = "An answer. [unknown-source]"
        path.write_text(json.dumps(record), encoding="utf-8")
    else:
        path = small_dataset / "data/evaluation/cases.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        records[0]["expected_sources"] = ["unknown-source"]
        path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown sources"):
        load_data()


def test_chunking_preserves_overlap_metadata_and_positions():
    chunks = chunk_documents(
        {
            "shipping": "# Shipping\n\none two three four five six",
            "account": "# Account\n\nalpha beta",
        },
        max_words=4,
        overlap_words=1,
    )
    assert [chunk.text for chunk in chunks] == [
        "one two three four", "four five six", "alpha beta",
    ]
    assert [chunk.position for chunk in chunks] == [0, 1, 0]
    assert [chunk.chunk_id for chunk in chunks] == [
        "shipping-0", "shipping-1", "account-0",
    ]
    assert chunks[0].title == "Shipping"
    assert chunks[2].doc_id == "account"


def test_real_qdrant_index_preserves_payload(monkeypatch):
    # Controlled vectors isolate database storage/search from embedding quality.
    class FixedEmbeddings:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            return np.array([
                [0.0, 1.0] if "password" in text.lower() else [1.0, 0.0]
                for text in texts
            ], dtype=np.float32)

    monkeypatch.setattr(retrieval, "SentenceTransformer", FixedEmbeddings)
    retriever = Retriever({
        "shipping": "# Shipping\n\nDelivery takes two days.",
        "account": "# Account\n\nNever share your password.",
    })
    try:
        hit = retriever.search("How do I protect my password?")[0]
        assert hit.doc_id == "account"
        assert hit.chunk_id == "account-0"
        assert hit.title == "Account"
        assert hit.position == 0
        assert hit.text == "Never share your password."
        assert hit.score == pytest.approx(1.0)
    finally:
        retriever.close()


def test_weak_hits_produce_abstention_without_loading_a_model():
    weak = make_hit(score=MIN_SCORE - 0.01)
    boundary = make_hit(score=MIN_SCORE)
    assert select_evidence([weak, boundary]) == [boundary]
    evidence = select_evidence([weak])

    generator = AnswerGenerator.__new__(AnswerGenerator)
    result = generator.answer("An unrelated question", evidence)
    assert result["status"] == "abstained"
    assert result["answer"] == FALLBACK_ANSWER
    assert result["raw_answer"] == ""


@pytest.mark.parametrize(
    "text,status",
    [
        ("One to two days. [shipping]", "answered"),
        ("One to two days. [unknown-source]", "rejected"),
        ("One to two days.", "rejected"),
        ("", "rejected"),
        (FALLBACK_ANSWER, "abstained"),
    ],
)
def test_citation_and_fallback_validation(text, status):
    assert validate_answer(text, [make_hit()])[0] == status


def test_ranking_and_word_boundaries():
    metrics = retrieval_metrics(
        make_case(), [make_hit("account"), make_hit()],
    )
    assert metrics == {"recall_at_k": 1.0, "mrr_at_k": 0.5}
    assert not contains_term("noteworthy", "not")
    assert contains_term("It is NOT permitted.", "not")


def test_rejection_does_not_count_as_a_correct_abstention():
    result = {
        "status": "rejected", "answer": FALLBACK_ANSWER,
        "raw_answer": "One to two days.", "citations": [],
        "latency_seconds": 0.1,
    }
    for unsupported in (False, True):
        metrics = answer_metrics(make_case(unsupported), result, [make_hit()])
        assert metrics["decision_accuracy"] == 0
    assert answer_metrics(make_case(), result, [make_hit()])["answer_relevance"] == 0


def test_evaluation_shares_evidence_and_creates_serializable_report(tmp_path):
    class FakeRetriever:
        def __init__(self):
            self.questions = []

        def search(self, question):
            self.questions.append(question)
            return [make_hit()]

    class FakeGenerator:
        def __init__(self):
            self.received = []

        def answer(self, question, evidence):
            self.received.append((question, evidence))
            return {
                "status": "answered",
                "answer": "One to two business days. [shipping]",
                "raw_answer": "One to two business days. [shipping]",
                "citations": ["shipping"],
                "reason": "Test result",
                "latency_seconds": 0.1,
            }

    retriever = FakeRetriever()
    first, second = FakeGenerator(), FakeGenerator()
    case = make_case()
    report = evaluate([case], retriever, {"baseline": first, "hybrid": second})

    assert retriever.questions == [case.question]
    assert first.received[0][1] is second.received[0][1]
    assert report["retrieval"]["metrics"]["recall_at_k"] == 1.0
    assert report["systems"]["hybrid"]["metrics"]["answer_relevance"] == 1.0

    output = tmp_path / "evaluation_results.json"
    output.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    restored = json.loads(output.read_text(encoding="utf-8"))
    assert restored["systems"]["baseline"]["results"][0]["id"] == case.id
    assert "faithfulness" in restored["metric_definitions"]

