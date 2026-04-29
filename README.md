# mlx-nanochat

**Train small tool-calling models on your Mac, then scale to HPC with eden-models.**

A proper MLX port of [Karpathy's nanochat](https://github.com/karpathy/nanochat) -- the minimal, end-to-end ChatGPT training pipeline for Apple Silicon.

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> *"The goal of nanochat is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000."* -- Andrej Karpathy

## What this is

A single-file-ish, readable, hackable codebase that trains a ChatGPT-class model from scratch on your Mac. No PyTorch. No CUDA. Just MLX and Apple Silicon.

```
Raw data --> Tokenizer --> Pretrain --> SFT --> Chat
   All on your Mac. All in one repo. ~$0 in cloud costs.
```

## Pipeline

```
+-------------------+     +-------------+     +----------+
| Data Sources      |     | Pretrain    |     | SFT      |
|                   |     |             |     |          |
| - ClimbMix-400B   +---->+ Base model  +---->+ ChatML   |
| - Flywheel export |     | (depth=N)   |     | SmolTalk |
| - Custom JSONL    |     +-------------+     +----+-----+
+-------------------+                              |
                                                   v
                                            +------+------+
                                            | Chat / Eval |
                                            +-------------+
```

## One complexity dial

```bash
python -m scripts.train --depth=4    # Tiny model, 10 minutes, fits 8GB Mac
python -m scripts.train --depth=12   # Medium model, 2 hours, needs 16GB
python -m scripts.train --depth=24   # GPT-2 class, 8 hours, needs 64GB
python -m scripts.train --depth=32   # Best quality, 24 hours, needs 64GB+
```

`--depth` automatically sets: model width, attention heads, learning rate, batch size, training duration. You don't configure anything else.

**Reference target:** Gemma 4 E4B (2560 hidden, 34 layers, ~4B params) -- train a smaller model locally, compare to E4B baseline, then scale up on HPC.

## What's different from the original

| | karpathy/nanochat | mlx-nanochat |
|---|---|---|
| Backend | PyTorch + CUDA | **MLX** (Apple native) |
| Hardware | 8x H100 for GPT-2 class | **1x M1 Max** (slower but works) |
| Flash Attention | FA3 kernels | **MLX additive masks** (functionally equivalent) |
| Optimizer | Muon + AdamW | **AdamW** (Muon MLX port in progress) |
| SFT formats | SmolTalk | **SmolTalk + ChatML + tool-calling** |
| Training cost | ~$100 (8x H100 rental) | **$0** (your Mac) |

## Full Pipeline

```bash
# 1. Download training data (FineWeb subset)
python -m nanochat_mlx.dataset -n 8     # 8 shards, ~800MB

# 2. Train BPE tokenizer
python -m scripts.tok_train              # vocab_size=32768

# 3. Pretrain base model
python -m scripts.train --depth=12       # ~2 hours on M1 Max

# 4. Supervised fine-tuning (SmolTalk)
python -m scripts.sft --depth=12         # ~30 minutes

# 5. Chat with your model
python -m scripts.chat --depth=12
```

### Tool-calling SFT (ChatML format)

```bash
# Import data from eden-flywheel
python scripts/import_flywheel.py \
    --flywheel-db ~/.config/eden-flywheel/flywheel.db \
    --output data/tool_calling.jsonl \
    --min-quality 0.6

# Or use any ChatML JSONL file
python -m scripts.sft --depth=12 --data data/tool_calling.jsonl --format chatml
```

ChatML format supports multi-turn conversations with tool calls:

```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "List files in the current directory"},
  {"role": "assistant", "content": "<tool_call>{\"name\": \"bash\", \"arguments\": {\"command\": \"ls -la\"}}</tool_call>"},
  {"role": "tool", "content": "total 48\ndrwxr-xr-x  12 user ..."},
  {"role": "assistant", "content": "Here are the files in your current directory: ..."}
]}
```

Loss is only computed on assistant turns. System, user, and tool messages are masked.

### Train a small model and compare to Gemma 4 E4B

```bash
# Train a 500M-param model with tool-calling data
python -m scripts.train --depth=8 --num-iterations=1000
python -m scripts.sft --depth=8 --data data/tool_calling.jsonl --format chatml

# Chat with it
python -m scripts.chat --depth=8 --top-p 0.95
```

## Training Performance

| Mac | depth=4 | depth=12 | depth=24 |
|---|---|---|---|
| M1 8GB | ~10 min | ~4 hrs | OOM |
| M1 Max 64GB | ~5 min | ~2 hrs | ~8 hrs |
| M4 16GB | ~4 min | ~1.5 hrs | ~10 hrs |
| M4 Max 128GB | ~3 min | ~45 min | ~4 hrs |

## Benchmark Results

*Coming soon -- eval harness integration in progress.*

## Credits

**Original concept and implementation:** [Andrej Karpathy](https://github.com/karpathy/nanochat) (MIT License)

The nanochat architecture, scaling laws, training methodology, and the "one complexity dial" design are all Karpathy's work. This is a community MLX port adapted for Apple Silicon training with added tool-calling support.

**Also referenced:**
- [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) -- Earlier MLX port
- [NeuroArchitect/nanochat-mlx](https://github.com/NeuroArchitect/nanochat-mlx) -- Another MLX port
- [Doriandarko/MLX-GRPO](https://github.com/Doriandarko/MLX-GRPO) -- GRPO training on MLX
- Apple MLX team -- framework and examples

## License

MIT -- matching the original nanochat license.
