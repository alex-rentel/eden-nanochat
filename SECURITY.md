# Security policy

## Reporting a vulnerability

Email **alex@renaissanceintelligence.ai** with the details. Avoid filing a public GitHub issue for anything you believe could be exploited — open a private channel first.

A useful report includes:

- The version you reproduced against (`pyproject.toml` `version` + commit SHA).
- A minimal repro: training script invocation, dataset shard, the failure mode.
- Versions: `python --version`, `pip freeze | grep -E "mlx|tiktoken|numpy"`.

You should expect a first reply within a few days.

## Supported versions

Only `main` and the latest release get security fixes. Forks and historical branches are unsupported.

## Scope

In scope: anything that lets crafted training data crash the runtime, leak memory across batches, write outside the configured base directory, or execute arbitrary code via the dataset / SFT format parsers.

Out of scope:

- Training instabilities, NaN losses, divergence on a particular dataset — these are bugs or hyperparameter issues, not security issues. Open a public GitHub issue.
- Issues in MLX, tiktoken, or pyarrow themselves — report upstream.
- Anything that requires the attacker to already have write access to the user's `~/.cache/nanochat-mlx/` (or wherever `get_base_dir()` resolves to).
