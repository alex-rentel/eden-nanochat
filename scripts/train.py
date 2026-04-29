"""
Pretrain base model. Run as:
    python -m scripts.train --depth=4
"""

import argparse

from nanochat_mlx.train import run_training

parser = argparse.ArgumentParser(description="Pretrain base model")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = auto)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="data:param ratio for training horizon")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=4, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens (-1 = auto)")
parser.add_argument("--matrix-lr", type=float, default=0.001, help="learning rate for AdamW")
parser.add_argument("--embedding-lr", type=float, default=0.01, help="learning rate for embeddings")
parser.add_argument("--weight-decay", type=float, default=0.1, help="weight decay")
parser.add_argument("--warmup-steps", type=int, default=40, help="LR warmup steps")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR fraction")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=524288, help="tokens to evaluate val loss on")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")

args = parser.parse_args()
run_training(args)
