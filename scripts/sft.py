"""
Supervised Fine-Tuning. Run as:
    python -m scripts.sft --depth=4
"""

import argparse
from nanochat_mlx.sft import run_sft

parser = argparse.ArgumentParser(description="SFT fine-tuning")
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--device-batch-size", type=int, default=4, help="batch size")
parser.add_argument("--num-iterations", type=int, default=500, help="number of SFT steps")
parser.add_argument("--learning-rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("--max-examples", type=int, default=50000, help="max training examples")
parser.add_argument("--save-every", type=int, default=100, help="save every N steps")

args = parser.parse_args()
run_sft(args)
