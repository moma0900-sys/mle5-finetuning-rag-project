import json
import re
from pathlib import Path

from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent


class TrainingExample(BaseModel):
    id: str = Field(min_length=1)
    context: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    reference_ids: list[str] = Field(min_length=1)


class EvaluationCase(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_sources: list[str]
    required_terms: list[str]
    forbidden_terms: list[str]
    should_abstain: bool


def load_data():
    documents = {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "data/knowledge_base").glob("*.md"))
    }

    training = [
        TrainingExample.model_validate_json(line)
        for line in (ROOT / "data/training/instruction_examples.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    cases = [
        EvaluationCase.model_validate(case)
        for case in json.loads(
            (ROOT / "data/evaluation/cases.json").read_text(encoding="utf-8")
        )
    ]

    if not documents or not training or not cases:
        raise ValueError(
            "Documents, training data and evaluation cases are required."
        )

    # IDs innerhalb der beiden Datensätze prüfen
    for name, records in [("training", training), ("evaluation", cases)]:
        ids = [record.id for record in records]

        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs in {name} data.")

    known_sources = set(documents)

    # Deklarierte und tatsächlich zitierte Trainingsquellen prüfen
    for example in training:
        cited_sources = set(re.findall(
            r"\[([^\[\]\r\n]+)\]",
            example.context + "\n" + example.answer,
        ))

        unknown = (
            set(example.reference_ids) | cited_sources
        ) - known_sources

        if unknown:
            raise ValueError(
                f"{example.id}: unknown sources {sorted(unknown)}"
            )

    # Erwartete Evaluationsquellen prüfen
    for case in cases:
        unknown = set(case.expected_sources) - known_sources

        if unknown:
            raise ValueError(
                f"{case.id}: unknown sources {sorted(unknown)}"
            )

    # Exakt gleiche Fragen nach Vereinheitlichung erkennen
    training_questions = {
        " ".join(example.question.casefold().split())
        for example in training
    }

    evaluation_questions = {
        " ".join(case.question.casefold().split())
        for case in cases
    }

    if training_questions & evaluation_questions:
        raise ValueError("Training and evaluation questions overlap.")

    return documents, training, cases