import json

from data_loading import ROOT, load_data
from evaluation import evaluate
from generation import MAX_NEW_TOKENS, AnswerGenerator
from retrieval import (
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    MIN_SCORE,
    TOP_K,
    Retriever,
    select_evidence,
)
from training import ADAPTER_PATH


def main():
    documents, training, cases = load_data()

    print(f"Training examples: {len(training)}")
    print(f"Evaluation cases: {len(cases)}")

    generators = {
        "baseline_rag": AnswerGenerator(use_adapter=False),
        "hybrid_rag_lora": AnswerGenerator(use_adapter=True),
    }

    retriever = Retriever(documents)

    try:
        # Ein Aufwärmaufruf pro Modell vor der Zeitmessung
        question = "How long does express shipping take?"
        evidence = select_evidence(
            retriever.search(question)
        )

        for generator in generators.values():
            generator.answer(question, evidence)

        report = evaluate(
            cases,
            retriever,
            generators,
        )

    finally:
        retriever.close()

    report["configuration"] = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": EMBEDDING_REVISION,
        "top_k": TOP_K,
        "min_score": MIN_SCORE,
        "max_chunk_words": 90,
        "overlap_words": 15,
        "threshold_status": (
            "Preliminary threshold, fixed before this benchmark; "
            "not calibrated on held-out cases."
        ),
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "device": "cpu",
        "training": json.loads(
            (ADAPTER_PATH / "training_summary.json").read_text(
                encoding="utf-8"
            )
        ),
    }

    report_path = ROOT / "reports/evaluation_results.json"
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\nShared retrieval:")

    for name, value in report["retrieval"]["metrics"].items():
        print(f"  {name}: {value:.4f}")

    for name, system in report["systems"].items():
        print(f"\n{name}:")

        for metric, value in system["metrics"].items():
            formatted = (
                "N/A"
                if value is None
                else f"{value:.4f}"
            )
            print(f"  {metric}: {formatted}")

        print(
            f"  Status counts: {system['status_counts']}"
        )

    print(f"\nResults saved to: {report_path}")


if __name__ == "__main__":
    main()