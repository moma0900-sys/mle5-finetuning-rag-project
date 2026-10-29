import json
import os
import random
from time import perf_counter

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from data_loading import ROOT, load_data
from prompts import SYSTEM_MESSAGE, build_messages, build_prompt


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
ADAPTER_PATH = ROOT / "artifacts/support-lora-adapter"

MAX_SEQUENCE_LENGTH = 512
EPOCHS = 8
LEARNING_RATE = 2e-4
SEED = 42


def encode_example(tokenizer, example):
    prompt = build_prompt(
        tokenizer, example.question, example.context
    )

    messages = build_messages(example.question, example.context)
    messages.append({
        "role": "assistant",
        "content": example.answer,
    })

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )

    input_ids = encoded["input_ids"]

    if input_ids.shape[1] > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"{example.id}: training example is too long."
        )

    if input_ids[0, :len(prompt_ids)].tolist() != prompt_ids:
        raise ValueError(
            f"{example.id}: prompt token boundary changed."
        )

    # Nur Antworttokens einschließlich Endmarkierung bewerten
    labels = input_ids.clone()
    labels[:, :len(prompt_ids)] = -100

    if not (labels != -100).any():
        raise ValueError(
            f"{example.id}: no answer tokens to train."
        )

    return {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


def train_adapter():
    set_seed(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    _, examples, _ = load_data()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Alle Beispiele vor dem Training prüfen und tokenisieren
    batches = [
        encode_example(tokenizer, example)
        for example in examples
    ]

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        dtype=torch.float32,
    )
    base_model.config.pad_token_id = tokenizer.eos_token_id
    base_model.config.use_cache = False
    base_model.loss_type = "ForCausalLM"

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)
    model.to("cpu")

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    total = sum(p.numel() for p in model.parameters())

    if not trainable or not all(
        "lora" in name
        for name, p in model.named_parameters()
        if p.requires_grad
    ):
        raise ValueError(
            "Expected only LoRA parameters to be trainable."
        )

    print(f"Training examples: {len(batches)}")
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    print(f"Trainable percentage: {100 * trainable / total:.3f}%")

    optimizer = torch.optim.AdamW(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=LEARNING_RATE,
    )

    epoch_losses = []
    started = perf_counter()
    model.train()

    for epoch in range(EPOCHS):
        random.shuffle(batches)
        losses = []

        for batch in batches:
            loss = model(**batch).loss

            if not torch.isfinite(loss):
                raise RuntimeError("Training loss is not finite.")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.detach()))

        mean_loss = sum(losses) / len(losses)
        epoch_losses.append(mean_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}: loss={mean_loss:.4f}"
        )

    training_seconds = perf_counter() - started

    # Adapter und Tokenizer speichern
    ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
    model.peft_config["default"].revision = MODEL_REVISION

    model.save_pretrained(
        ADAPTER_PATH,
        save_embedding_layers=False,
    )
    tokenizer.save_pretrained(ADAPTER_PATH)

    # Geforderte Einstellungen und Ergebnisse dokumentieren
    summary = {
        "base_model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "prompt_format": "native-chat",
        "system_message": SYSTEM_MESSAGE,
        "lora_target_modules": ["q_proj", "v_proj"],
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "batch_size": 1,
        "device": "cpu",
        "trainable_parameters": trainable,
        "total_parameters": total,
        "epoch_losses": epoch_losses,
        "training_seconds": training_seconds,
        "adapter_path": str(ADAPTER_PATH),
    }

    (ADAPTER_PATH / "training_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Adapter saved to: {ADAPTER_PATH}")


if __name__ == "__main__":
    train_adapter()