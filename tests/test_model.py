"""Tests for GPT model: parameter counting, learnable scalars, presets."""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.utils

from nanochat_mlx.gpt import GPT, GPTConfig, build_model, MODEL_PRESETS, has_ve


def _materialize(*arrays):
    """Wrapper for MLX array materialization."""
    mx.eval(*arrays)


def test_learnable_scalars_in_parameters():
    """Bug 1: Verify all learnable scalars appear in the parameter tree."""
    model = build_model(depth=4)
    params = dict(mlx.utils.tree_flatten(model.parameters()))
    param_keys = set(params.keys())

    for name in ["smear_lambda", "backout_lambda", "resid_lambdas", "x0_lambdas"]:
        assert name in param_keys, f"{name} not found in parameter tree"


def test_learnable_scalars_update_after_training_step():
    """Bug 1: Verify learnable scalars change after one training step."""
    model = build_model(depth=4)

    # Capture initial values
    initial_resid = model.resid_lambdas.tolist()
    initial_x0 = model.x0_lambdas.tolist()
    initial_smear = model.smear_lambda.tolist()
    initial_backout = model.backout_lambda.tolist()

    # Run one training step
    loss_fn = lambda model, x, y: model(x, targets=y)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    B, T = 2, 64
    x = mx.random.randint(0, model.config.vocab_size, shape=(B, T))
    y = mx.random.randint(0, model.config.vocab_size, shape=(B, T))

    loss, grads = loss_and_grad_fn(model, x, y)

    import mlx.optimizers as optim
    optimizer = optim.AdamW(learning_rate=0.01)
    optimizer.update(model, grads)
    _materialize(model.parameters(), optimizer.state)

    # Check at least some scalars changed
    changed = 0
    if model.resid_lambdas.tolist() != initial_resid:
        changed += 1
    if model.x0_lambdas.tolist() != initial_x0:
        changed += 1
    if model.smear_lambda.tolist() != initial_smear:
        changed += 1
    if model.backout_lambda.tolist() != initial_backout:
        changed += 1

    assert changed > 0, "No learnable scalars changed after training step"


def test_num_params_accuracy():
    """Bug 4: Verify num_params() matches expected count from config math."""
    config = GPTConfig(
        sequence_len=128,
        vocab_size=256,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
    )
    model = GPT(config, pad_vocab_size_to=64)
    model.init_weights()
    _materialize(model.parameters())

    reported = model.num_params()

    # Compute expected params manually
    n_embd = config.n_embd
    head_dim = n_embd // config.n_head
    kv_dim = config.n_kv_head * head_dim
    padded_vocab = ((config.vocab_size + 63) // 64) * 64

    expected = 0
    # wte embedding
    expected += padded_vocab * n_embd
    # lm_head
    expected += n_embd * padded_vocab
    # Per block
    for i in range(config.n_layer):
        # Attention: q, k, v, proj (all bias=False)
        expected += n_embd * (config.n_head * head_dim)  # c_q
        expected += n_embd * kv_dim  # c_k
        expected += n_embd * kv_dim  # c_v
        expected += n_embd * n_embd  # c_proj
        # MLP: fc + proj
        expected += n_embd * (4 * n_embd)  # c_fc
        expected += (4 * n_embd) * n_embd  # c_proj
        # VE gate (if applicable)
        if has_ve(i, config.n_layer):
            expected += 12 * config.n_kv_head  # ve_gate weight

    # Value embeddings
    for i in range(config.n_layer):
        if has_ve(i, config.n_layer):
            expected += padded_vocab * kv_dim

    # Learnable scalars
    expected += config.n_layer  # resid_lambdas
    expected += config.n_layer  # x0_lambdas
    expected += 1  # smear_lambda
    expected += 1  # backout_lambda
    expected += 24 * 1  # smear_gate weight (Linear(24, 1, bias=False))

    tolerance = 0.01
    ratio = abs(reported - expected) / max(expected, 1)
    assert ratio < tolerance, (
        f"num_params() reports {reported:,} but expected ~{expected:,} "
        f"(difference: {abs(reported - expected):,}, ratio: {ratio:.4f})"
    )


def test_gemma4_e4b_preset_exists():
    """Phase 3: Verify Gemma 4 E4B preset is defined."""
    assert "gemma4-e4b" in MODEL_PRESETS
    preset = MODEL_PRESETS["gemma4-e4b"]
    assert preset.n_embd == 2560
    assert preset.n_layer == 34
    assert preset.n_head == 16
    assert preset.n_kv_head == 8


def test_model_forward_backward():
    """Smoke test: model runs forward and backward without error."""
    model = build_model(depth=4)
    B, T = 2, 64
    x = mx.random.randint(0, model.config.vocab_size, shape=(B, T))
    y = mx.random.randint(0, model.config.vocab_size, shape=(B, T))

    loss = model(x, targets=y)
    _materialize(loss)
    assert loss.item() > 0

    # Inference mode
    logits, cache = model(x[:, :8])
    _materialize(logits)
    assert logits.shape == (2, 8, model.config.vocab_size)
