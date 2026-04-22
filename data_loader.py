"""
Data loader for semantic idea compression.

Goal: preserve the full propositional content of the input at minimal bit cost.
This is NOT summarization. The decoder target is the source itself — the model
must learn to reconstruct all ideas, arguments, and logical structure, just more
efficiently encoded.

Using WikiLarge's `source` column (complex Wikipedia sentences) for both encoder
input and decoder target. The `target` (simplified) column is intentionally ignored:
simplification is semantically lossy and trains the model to drop exactly the
complex reasoning structure we want to preserve.

For longer chains of reasoning, switch `split="train"` to a paragraph-level dataset
(e.g. arXiv abstracts, Wikipedia paragraphs, BillSum) and adjust MAX_LENGTH.
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BartTokenizer

import config


def get_dataloader(split: str = "train"):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    dataset = load_dataset("eilamc14/wikilarge-clean", split=split)
    # Keep only source sentences long enough to contain real ideas.
    # Short sentences (<= 10 tokens) rarely contain complex reasoning.
    dataset = dataset.filter(
        lambda x: len(x["source"].strip().split()) > 10
    )

    # Cap dataset size so each epoch is tractable on a single GPU.
    # Full WikiLarge (~296k examples) → 35k steps/epoch → 420k steps total.
    # At 4k steps/epoch: 12 epochs = 48k steps total (~4-6 hours on T4).
    # Increase MAX_TRAIN_SAMPLES for final training runs.
    MAX_TRAIN_SAMPLES = 32_000
    if split == "train" and len(dataset) > MAX_TRAIN_SAMPLES:
        dataset = dataset.select(range(MAX_TRAIN_SAMPLES))

    def tokenize_function(examples):
        # Encoder input: the source text (complex, information-rich)
        inputs = tokenizer(
            examples["source"],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_LENGTH,
        )

        # Decoder target: the SAME source text.
        # The model must reconstruct all propositional content from the bottleneck.
        # If it can't, it hasn't captured the ideas — the loss correctly penalises this.
        targets = tokenizer(
            examples["source"],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_LENGTH,
        )

        # Mask padding positions from the CE loss
        inputs["labels"] = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
            for seq in targets["input_ids"]
        ]
        return inputs

    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized = tokenized.remove_columns(["source", "target"])
    tokenized.set_format("torch")

    dataloader = DataLoader(
        tokenized,
        batch_size=config.BATCH_SIZE,
        shuffle=(split == "train"),
    )

    return dataloader, tokenizer
