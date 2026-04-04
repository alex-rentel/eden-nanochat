"""
BOS-aligned bestfit packing dataloader for pretraining.
Every row starts with BOS. Documents packed using best-fit to minimize cropping.
100% utilization (no padding), ~35% tokens cropped at T=2048.
"""

import numpy as np
import mlx.core as mx
import pyarrow.parquet as pq

from nanochat_mlx.dataset import list_parquet_files


def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """Infinite iterator over document batches from parquet files."""
    parquet_paths = list_parquet_files(warn_on_legacy=(split == "train"))
    assert len(parquet_paths) != 0, "No dataset parquet files found. Run: python -m nanochat_mlx.dataset -n 8"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    epoch = resume_epoch

    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            if first_pass and resume_rg_idx is not None and pq_idx == resume_pq_idx:
                rg_idx = resume_rg_idx + 1
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None
            else:
                rg_idx = 0
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += 1
            pq_idx += 1
        first_pass = False
        epoch += 1


def data_loader_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_batch_size=128,
    resume_state_dict=None,
    buffer_size=1000
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    For each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly
    """
    assert split in ["train", "val"]

    row_capacity = T + 1  # +1 because we split into inputs[:-1] and targets[1:]
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        for tokens in token_lists:
            if len(tokens) > 1:  # Skip empty docs (BOS-only)
                doc_buffer.append(tokens)

    # Pre-allocate numpy buffer for row construction
    row_buffer = np.zeros((B, row_capacity), dtype=np.int32)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Refill buffer if needed
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Best-fit: find largest doc that fits
                # TODO: optimize to O(log n) with sorted container if buffer_size > 10000
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = doc
                    pos += doc_len
                else:
                    # No doc fits: crop shortest to fill exactly
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                    pos += remaining

        # Split into inputs and targets
        inputs = mx.array(row_buffer[:, :-1])
        targets = mx.array(row_buffer[:, 1:])
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        yield inputs, targets, state_dict


def data_loader_simple(tokenizer, B, T, split, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in data_loader_bos_bestfit(tokenizer, B, T, split, **kwargs):
        yield inputs, targets
