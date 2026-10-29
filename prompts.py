FALLBACK_ANSWER = (
    "The approved material is insufficient. Escalate for manual review."
)

SYSTEM_MESSAGE = (
    "Answer only from approved context. Start with the direct answer, "
    "then give the next action. Keep the answer concise. "
    "Cite every policy claim with source IDs in square brackets. "
    f"If the context is insufficient, reply exactly: {FALLBACK_ANSWER}"
)


def build_messages(question, context):
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": f"Approved context:\n{context}\n\nQuestion: {question}",
        },
    ]


def build_prompt(tokenizer, question, context):
    return tokenizer.apply_chat_template(
        build_messages(question, context),
        tokenize=False,
        add_generation_prompt=True,
    )