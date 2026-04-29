"""Tests for inference engine: nucleus sampling."""

import mlx.core as mx

from nanochat_mlx.engine import sample_next_token


def test_nucleus_sampling():
    """Bug 5: Verify nucleus (top-p) sampling works."""
    mx.random.seed(42)
    # Create logits with a clear dominant token
    logits = mx.array([[0.0] * 100])
    logits = logits.at[0, 0].add(10.0)  # Token 0 is dominant

    # With top_p=0.5, should almost always pick token 0
    samples = []
    for _ in range(20):
        token = sample_next_token(logits, temperature=1.0, top_p=0.5)
        samples.append(token.item())

    assert samples.count(0) > 15, f"Expected token 0 to dominate with top_p=0.5, got {samples}"


def test_top_k_sampling():
    """Verify top-k sampling restricts to k candidates."""
    mx.random.seed(42)
    logits = mx.zeros((1, 100))
    logits = logits.at[0, 0].add(5.0)
    logits = logits.at[0, 1].add(4.0)
    logits = logits.at[0, 2].add(3.0)

    samples = set()
    for _ in range(50):
        token = sample_next_token(logits, temperature=1.0, top_k=3)
        samples.add(token.item())

    assert samples.issubset({0, 1, 2}), f"top_k=3 should only produce tokens 0-2, got {samples}"


def test_greedy_sampling():
    """Temperature=0 should be deterministic argmax."""
    logits = mx.array([[1.0, 5.0, 2.0, 3.0]])
    token = sample_next_token(logits, temperature=0.0)
    assert token.item() == 1
