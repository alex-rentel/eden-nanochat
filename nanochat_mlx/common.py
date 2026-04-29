"""
Common utilities for mlx-nanochat.
"""

import os

BANNER = r"""
              _         _         _
  _ __   __ _| |_ __   ___   ___| |__   __ _| |_
 | '_ \ / _` | '_ \ / _ \ / __| '_ \ / _` | __|
 | | | | (_| | | | | (_) | (__| | | | (_| | |_
 |_| |_|\__,_|_| |_|\___/ \___|_| |_|\__,_|\__|  MLX
"""


def print_banner():
    print(BANNER)


def get_base_dir():
    """Return base directory for all nanochat data (checkpoints, tokenizer, etc.)."""
    base = os.environ.get("NANOCHAT_DIR", os.path.expanduser("~/.nanochat"))
    os.makedirs(base, exist_ok=True)
    return base
