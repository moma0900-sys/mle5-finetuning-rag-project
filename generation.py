import os
import re
from time import perf_counter

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import FALLBACK_ANSWER, build_prompt
from training import ADAPTER_PATH, MODEL_NAME, MODEL_REVISION


MAX_NEW_TOKENS = 100


def validate_answer(raw_answer, evidence):
    if raw_answer == FALLBACK_ANSWER:
        return (
            "abstained",
            [],
            "Model reported insufficient evidence.",
        )

    citations = sorted(set(
        re.findall(r"\[([^\[\]\r\n]+)\]", raw_answer)
    ))
    allowed_sources = {hit.doc_id for hit in evidence}

    if not raw_answer:
        return "rejected", citations, "Empty answer."

    if not citations:
        return "rejected", citations, "Missing source citations."

    if set(citations) - allowed_sources:
        return (
            "rejected",
            citations,
            "Citation was not in the supplied evidence.",
        )

    return "answered", citations, "Citation IDs passed validation."


class AnswerGenerator:
    def __init__(self, use_adapter=False):
        torch.set_num_threads(min(4, os.cpu_count() or 1))

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            dtype=torch.float32,
        )

        if use_adapter:
            if not (
                ADAPTER_PATH / "adapter_model.safetensors"
            ).is_file():
                raise FileNotFoundError(
                    "Train and save the adapter first."
                )

            model = PeftModel.from_pretrained(
                model,
                ADAPTER_PATH,
                is_trainable=False,
            )

        self.model = model.to("cpu").eval()
        self.model.config.use_cache = True

    def answer(self, question, evidence):
        started = perf_counter()

        # Ohne geeignete Belege keine Antwort generieren
        if not evidence:
            return {
                "status": "abstained",
                "answer": FALLBACK_ANSWER,
                "raw_answer": "",
                "citations": [],
                "reason": (
                    "No evidence passed the retrieval threshold."
                ),
                "latency_seconds": perf_counter() - started,
            }

        context = "\n\n".join(
            f"[{hit.doc_id}] {hit.text}"
            for hit in evidence
        )

        prompt = build_prompt(
            self.tokenizer,
            question,
            context,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_length = inputs["input_ids"].shape[1]

        if (
            input_length + MAX_NEW_TOKENS
            > self.model.config.max_position_embeddings
        ):
            raise ValueError(
                "Prompt and answer exceed the model context limit."
            )

        # Antwort erzeugen, ohne Modellparameter zu verändern
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Nur die neu erzeugte Antwort auslesen
        new_tokens = output[0, input_length:]

        raw_answer = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        status, citations, reason = validate_answer(
            raw_answer,
            evidence,
        )

        # Abgebrochene Antworten ebenfalls zurückweisen
        if (
            len(new_tokens) == MAX_NEW_TOKENS
            and new_tokens[-1].item()
            != self.tokenizer.eos_token_id
        ):
            status = "rejected"
            reason = (
                "Answer reached the generation limit without ending."
            )

        return {
            "status": status,
            "answer": (
                raw_answer
                if status == "answered"
                else FALLBACK_ANSWER
            ),
            "raw_answer": raw_answer,
            "citations": citations,
            "reason": reason,
            "latency_seconds": perf_counter() - started,
        }