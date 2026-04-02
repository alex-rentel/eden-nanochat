"""
Train a BPE tokenizer on the pretraining data.
    python -m scripts.tok_train
"""

import os
import time
import argparse
import numpy as np

from nanochat_mlx.tokenizer import HuggingFaceTokenizer, SPECIAL_TOKENS
from nanochat_mlx.common import get_base_dir
from nanochat_mlx.dataset import parquets_iter_batched

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='Max characters to train on')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Max chars per document')
parser.add_argument('--vocab-size', type=int, default=32768, help='Vocabulary size')
args = parser.parse_args()

print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")


def text_iterator():
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return


t0 = time.time()
tokenizer = HuggingFaceTokenizer.train_from_iterator(text_iterator(), args.vocab_size)
train_time = time.time() - t0
print(f"Training time: {train_time:.2f}s")

# Save
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# Sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Special chars: @#$%^&*()"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text, f"Roundtrip failed: {decoded!r} != {test_text!r}"
print("Tokenizer roundtrip test passed!")

# Save token-to-bytes mapping for BPB evaluation
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_bytes = []
for token_id in range(vocab_size):
    token_str = tokenizer.decode([token_id])
    if token_str in special_set:
        token_bytes.append(0)
    else:
        token_bytes.append(len(token_str.encode("utf-8")))

token_bytes = np.array(token_bytes, dtype=np.int32)
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.npy")
np.save(token_bytes_path, token_bytes)
print(f"Saved token_bytes to {token_bytes_path}")

# Stats
nonzero = token_bytes[token_bytes > 0].astype(np.float32)
print(f"Token bytes - min: {nonzero.min():.0f}, max: {nonzero.max():.0f}, mean: {nonzero.mean():.2f}")
