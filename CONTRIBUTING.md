# Contributing to mlx-nanochat

This is an MLX port of [Karpathy's nanochat](https://github.com/karpathy/nanochat) — the minimal end-to-end ChatGPT training pipeline, adapted for Apple Silicon. Focused, hackable, end-to-end on a Mac.

## Dev environment

Apple Silicon required (MLX is Apple-Silicon-only). Python 3.11–3.13.

```bash
git clone https://github.com/alex-rentel/eden-nanochat.git
cd eden-nanochat
uv sync                           # or: pip install -e ".[dev]"
```

`uv` is the canonical install path (the repo ships a `uv.lock`); `pip install -e ".[dev]"` works fine too.

## Running checks locally

```bash
ruff check nanochat_mlx/ tests/ scripts/
pyright
pytest tests/ -q
```

CI (`.github/workflows/test.yml`) runs all three: ruff on Ubuntu as a fast pre-gate, then pyright + pytest on macos-14 across Python 3.11/3.12/3.13.

## Layout

| Path | What lives here |
|---|---|
| `nanochat_mlx/gpt.py` | Model definition (depth-controlled GPT) |
| `nanochat_mlx/train.py` | Pretraining loop (the depth dial) |
| `nanochat_mlx/sft.py` | Supervised fine-tuning (SmolTalk + ChatML + tool-calling) |
| `nanochat_mlx/engine.py` | Inference engine for chat / eval |
| `nanochat_mlx/dataset.py` | ClimbMix shard download + parsing |
| `nanochat_mlx/dataloader.py` | BOS-aligned best-fit packer |
| `nanochat_mlx/tokenizer.py` | tiktoken wrapper + ChatML special tokens |
| `nanochat_mlx/chatml.py` | ChatML format encoder / loss-mask builder |
| `nanochat_mlx/common.py` | Shared paths and helpers |
| `scripts/` | CLI entry points (train, sft, chat, tok_train, etc.) |
| `tests/` | pytest suite — see `tests/test_engine.py` etc. |

## Style

- Tool config lives in `pyproject.toml` (`[tool.ruff]`, `[tool.pyright]`, `[tool.pytest.ini_options]`).
- Pyright runs at "basic" mode with the noisy MLX-stub-related rules off (`reportAttributeAccessIssue`, `reportIndexIssue`, the Optional-narrowing family). `reportArgumentType` stays at error.
- Ruff: `E,F,W,I,B,UP` rule set; line length 110.
- 4-space indent for `*.py`, 2-space for yaml/json (see `.editorconfig`).

## Issues / questions

Open an issue at https://github.com/alex-rentel/eden-nanochat/issues. For training-config questions (depth, lr, batch sizes), the table in the README's "One complexity dial" section is the contract — anything that doesn't match it is a bug.
