"""Data utilities for chunking, embedding generation, and dataset preparation."""

import numpy as np


def chunk_text(text, max_chars=380):
    """Split text into chunks of at most max_chars, breaking only at word boundaries."""
    words = text.split()
    if not words:
        return []

    chunks = []
    current_words = []
    for word in words:
        if current_words and len(" ".join(current_words + [word])) > max_chars:
            chunks.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words.append(word)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def chunk_dataset(examples, max_chars=380):
    """Chunk a batch of examples for use with datasets.map(batched=True)."""
    all_chunks = []
    all_ids = []
    all_titles = []

    for i, text in enumerate(examples["text"]):
        chunks = chunk_text(text, max_chars)
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{examples['id'][i]}_{j}")
            all_titles.append(examples["title"][i])

    return {"id": all_ids, "title": all_titles, "text": all_chunks}


def generate_embeddings_batched(texts, model, tokenizer, device, batch_size=16):
    """Generate embeddings for a list of texts using a DPR context encoder."""
    import torch

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embeddings = model(**inputs).pooler_output
            all_embeddings.append(embeddings.float().cpu().numpy())

    return np.vstack(all_embeddings)
