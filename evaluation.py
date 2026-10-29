import re
from collections import Counter
from statistics import mean

from retrieval import TOP_K, select_evidence


def contains_term(text, term):
    """Begriff suchen, ohne beispielsweise 'not' in 'noteworthy' zu finden."""
    text = " ".join(text.casefold().split())
    term = " ".join(term.casefold().split())

    return re.search(
        r"(?<!\w)" + re.escape(term) + r"(?!\w)",
        text,
    ) is not None


def term_coverage(text, terms):
    if not terms:
        return None

    found = sum(
        contains_term(text, term)
        for term in terms
    )
    return found / len(terms)


def retrieval_metrics(case, hits):
    # Für unbeantwortbare Fragen gibt es keine erwartete Quelle
    if case.should_abstain:
        return {
            "recall_at_k": None,
            "mrr_at_k": None,
        }

    expected = set(case.expected_sources)

    if not expected:
        raise ValueError(
            f"{case.id}: supported case has no expected sources."
        )

    hits = hits[:TOP_K]
    retrieved = {hit.doc_id for hit in hits}

    reciprocal_rank = next(
        (
            1 / rank
            for rank, hit in enumerate(hits, start=1)
            if hit.doc_id in expected
        ),
        0.0,
    )

    return {
        "recall_at_k": (
            len(expected & retrieved) / len(expected)
        ),
        "mrr_at_k": reciprocal_rank,
    }


def answer_metrics(case, result, evidence):
    status = result["status"]
    answered = status == "answered"
    answer = result["answer"]

    citations = set(result["citations"])
    allowed = {hit.doc_id for hit in evidence}

    if case.should_abstain:
        relevance = None
        raw_relevance = None
        faithful = status == "abstained"

    else:
        relevance = (
            term_coverage(answer, case.required_terms)
            if answered
            else 0.0
        )

        # Diagnose: Inhalt auch vor einer Zurückweisung betrachten
        raw_relevance = term_coverage(
            result["raw_answer"],
            case.required_terms,
        )

        faithful = (
            answered
            and bool(citations)
            and citations <= allowed
            and citations <= set(case.expected_sources)
            and not any(
                contains_term(answer, term)
                for term in case.forbidden_terms
            )
        )

    # Einfache, ausdrücklich begrenzte Stilprüfung
    if status == "abstained":
        style = 1.0

    elif not answered:
        style = 0.0

    else:
        first_sentence = re.split(
            r"[.!?]", answer, maxsplit=1
        )[0]

        checks = [
            len(answer.split()) <= 80,
            len(first_sentence.split()) <= 30,
            not any(
                contains_term(answer, hedge)
                for hedge in ("maybe", "probably", "perhaps")
            ),
        ]

        style = sum(checks) / len(checks)

    # Zurückweisung ist eine eigene Kategorie und zählt nicht als Verzicht
    correct = (
        status == "abstained"
        if case.should_abstain
        else answered
    )

    return {
        "answer_relevance": relevance,
        "raw_term_coverage": raw_relevance,
        "faithfulness": int(faithful),
        "style_compliance": style,
        "decision_accuracy": int(correct),
        "latency_seconds": result["latency_seconds"],
    }


def summarize(rows):
    averages = {}
    counts = {}

    for name in rows[0]["metrics"]:
        values = [
            row["metrics"][name]
            for row in rows
            if row["metrics"][name] is not None
        ]

        averages[name] = mean(values) if values else None
        counts[name] = len(values)

    return {
        "metrics": averages,
        "metric_case_counts": counts,
        "results": rows,
    }


def evaluate(cases, retriever, generators):
    if not cases:
        raise ValueError("Evaluation cases must not be empty.")

    retrieval_rows = []
    system_rows = {name: [] for name in generators}

    for position, case in enumerate(cases, start=1):
        hits = retriever.search(case.question)
        evidence = select_evidence(hits)

        retrieval_rows.append({
            "id": case.id,
            "question": case.question,
            "expected_sources": case.expected_sources,
            "should_abstain": case.should_abstain,
            "hits": [hit.model_dump() for hit in hits],
            "evidence": [hit.model_dump() for hit in evidence],
            "metrics": retrieval_metrics(case, hits),
        })

        for name, generator in generators.items():
            result = generator.answer(case.question, evidence)

            system_rows[name].append({
                "id": case.id,
                "expected_abstain": case.should_abstain,
                "expected_sources": case.expected_sources,
                "required_terms": case.required_terms,
                "forbidden_terms": case.forbidden_terms,
                **result,
                "metrics": answer_metrics(
                    case, result, evidence
                ),
            })

            print(
                f"[{position}/{len(cases)}] {case.id} | "
                f"{name}: {result['status']}",
                flush=True,
            )

    systems = {}

    for name, rows in system_rows.items():
        systems[name] = summarize(rows)
        systems[name]["status_counts"] = dict(
            Counter(row["status"] for row in rows)
        )

    return {
        "question_set": "data/evaluation/cases.json",
        "metric_definitions": {
            "recall_at_k": (
                "Mean fraction of expected document IDs found in the raw "
                "top-k chunks, before thresholding; supported cases only."
            ),
            "mrr_at_k": (
                "Mean reciprocal rank of the first expected source in the "
                "raw top-k chunks; missing sources score 0. "
                "Supported cases only."
            ),
            "answer_relevance": (
                "Required-term coverage in accepted answers; supported "
                "cases only. Abstained and rejected supported answers "
                "score 0."
            ),
            "raw_term_coverage": (
                "Diagnostic required-term coverage before rejection; "
                "supported cases only. No generation scores 0."
            ),
            "faithfulness": (
                "Citation/forbidden-term proxy, not semantic verification. "
                "Supported cases require an accepted answer, nonempty "
                "citations limited to supplied evidence and expected "
                "sources, and no forbidden phrase. Unsupported cases "
                "require abstention."
            ),
            "style_compliance": (
                "Mean of three checks: at most 80 words, first sentence "
                "at most 30 words, and no maybe/probably/perhaps. "
                "Abstentions score 1, even if unnecessary; rejected "
                "answers score 0. Does not verify a next action or "
                "full style compliance."
            ),
            "decision_accuracy": (
                "Answered supported cases or abstained unsupported cases "
                "score 1; every rejection scores 0. "
                "Not factual correctness."
            ),
            "latency_seconds": (
                "Mean answer-call wall time, including tokenization, "
                "generation and validation; excludes retrieval, model "
                "loading and indexing. Includes abstentions without "
                "generation; one untimed warm-up per model precedes "
                "the benchmark."
            ),
        },
        "retrieval": summarize(retrieval_rows),
        "systems": systems,
    }