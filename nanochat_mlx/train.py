"""
Training loop for mlx-nanochat.
Uses MLX's value_and_grad for automatic differentiation and AdamW optimizer.
Note: mx.eval() is MLX's array materialization function, not Python's eval() builtin.
"""

import json
import math
import os
import time
from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

from nanochat_mlx.common import get_base_dir, print_banner
from nanochat_mlx.dataloader import data_loader_bos_bestfit
from nanochat_mlx.gpt import build_model
from nanochat_mlx.tokenizer import get_token_bytes, get_tokenizer


def loss_fn(model, x, y):
    """Compute cross-entropy loss."""
    return model(x, targets=y)


def evaluate_bpb(model, val_loader, eval_steps, token_bytes):
    """Evaluate validation bits-per-byte."""
    total_loss = 0.0
    total_bytes = 0
    total_tokens = 0

    for step_i, (x, y, _) in enumerate(val_loader):
        if step_i >= eval_steps:
            break
        loss = model(x, targets=y)
        mx.eval(loss)  # mx.eval materializes MLX lazy arrays

        B, T = y.shape
        valid = y != -1
        n_valid = mx.sum(valid).item()
        total_tokens += n_valid
        total_loss += loss.item() * n_valid

        if token_bytes is not None:
            flat_y = y.reshape(-1)
            flat_valid = valid.reshape(-1)
            byte_counts = token_bytes[flat_y] * flat_valid.astype(mx.int32)
            total_bytes += mx.sum(byte_counts).item()

    if total_bytes > 0:
        avg_loss = total_loss / max(total_tokens, 1)
        bpb = avg_loss / math.log(2) * (total_tokens / total_bytes)
        return bpb
    else:
        return total_loss / max(total_tokens, 1) / math.log(2)


def save_checkpoint(checkpoint_dir, step, model, meta):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    weights_path = os.path.join(checkpoint_dir, f"step_{step:06d}.safetensors")
    meta_path = os.path.join(checkpoint_dir, f"step_{step:06d}_meta.json")

    flat_params = dict(mlx.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(weights_path, flat_params)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved checkpoint to {weights_path}")


def load_checkpoint(checkpoint_dir, step, model):
    """Load model checkpoint."""
    weights_path = os.path.join(checkpoint_dir, f"step_{step:06d}.safetensors")
    meta_path = os.path.join(checkpoint_dir, f"step_{step:06d}_meta.json")

    weights = mx.load(weights_path)
    model.load_weights(list(weights.items()))

    with open(meta_path) as f:
        meta = json.load(f)
    return meta


def run_training(args):
    """Main training function."""
    print_banner()

    # Load tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    # Build model from depth
    print(f"Building model with depth={args.depth}...")
    model = build_model(
        depth=args.depth,
        aspect_ratio=args.aspect_ratio,
        head_dim=args.head_dim,
        max_seq_len=args.max_seq_len,
        vocab_size=vocab_size,
    )
    config = model.config
    print(f"Model config: {json.dumps(asdict(config), indent=2)}")

    # Count parameters
    nparams = sum(p.size for _, p in mlx.utils.tree_flatten(model.parameters()))
    print(f"Total parameters: {nparams:,}")

    # Calculate training horizon from scaling laws
    d12_dim = ((12 * args.aspect_ratio + args.head_dim - 1) // args.head_dim) * args.head_dim
    d12_scaling_params = 12 * 12 * d12_dim * d12_dim + d12_dim * vocab_size
    D_REF = args.target_param_data_ratio * d12_scaling_params
    B_REF = 2**19

    target_tokens = int(args.target_param_data_ratio * nparams)
    total_batch_size = args.total_batch_size
    if total_batch_size == -1:
        batch_size_ratio = target_tokens / max(D_REF, 1)
        predicted = B_REF * batch_size_ratio ** 0.383
        total_batch_size = 2 ** round(math.log2(max(predicted, 256)))
        print(f"Auto-computed batch size: {total_batch_size:,} tokens")

    tokens_per_step = args.device_batch_size * args.max_seq_len
    grad_accum_steps = max(1, total_batch_size // tokens_per_step)
    actual_batch_size = tokens_per_step * grad_accum_steps

    if args.num_iterations > 0:
        num_iterations = args.num_iterations
    else:
        num_iterations = max(1, target_tokens // actual_batch_size)
    total_tokens = actual_batch_size * num_iterations
    print(f"Training for {num_iterations:,} iterations, {total_tokens:,} tokens")
    print(f"Batch size: {actual_batch_size:,} tokens (grad accum: {grad_accum_steps})")

    # Learning rates scaled by model dim
    dmodel_lr_scale = (config.n_embd / 768) ** -0.5
    matrix_lr = args.matrix_lr * dmodel_lr_scale

    # Create optimizer
    optimizer = optim.AdamW(
        learning_rate=matrix_lr,
        betas=[0.9, 0.95],
        weight_decay=args.weight_decay,
    )

    # LR schedule
    def get_lr_multiplier(it):
        warmup = args.warmup_steps
        warmdown = round(args.warmdown_ratio * num_iterations)
        if it < warmup:
            return (it + 1) / warmup
        elif it <= num_iterations - warmdown:
            return 1.0
        else:
            progress = (num_iterations - it) / warmdown
            return progress * 1.0 + (1 - progress) * args.final_lr_frac

    # Create data loader
    train_loader = data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len, split="train"
    )

    # Try to load token_bytes for BPB evaluation
    try:
        token_bytes = get_token_bytes()
    except (AssertionError, FileNotFoundError):
        token_bytes = None
        print("Warning: token_bytes not found, BPB evaluation unavailable")

    # Checkpoint directory
    base_dir = get_base_dir()
    output_dirname = f"d{args.depth}"
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)

    # Training loop
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    smooth_train_loss = 0.0
    total_training_time = 0.0
    step = 0

    print("\nStarting training...")
    x, y, dataloader_state = next(train_loader)

    while step <= num_iterations:
        last_step = step == num_iterations

        # Evaluation
        if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
            val_loader = data_loader_bos_bestfit(
                tokenizer, args.device_batch_size, args.max_seq_len, split="val"
            )
            eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len))
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            print(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")

        # Save checkpoint
        if last_step or (step > 0 and args.save_every > 0 and step % args.save_every == 0):
            meta = {
                "step": step,
                "model_config": asdict(config),
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": actual_batch_size,
            }
            save_checkpoint(checkpoint_dir, step, model, meta)

        if last_step:
            break

        # Training step
        t0 = time.time()

        # Update learning rate
        lrm = get_lr_multiplier(step)
        optimizer.learning_rate = mx.array(matrix_lr * lrm)

        # Gradient accumulation
        if grad_accum_steps == 1:
            loss, grads = loss_and_grad_fn(model, x, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)  # mx.eval materializes MLX lazy arrays
            x, y, dataloader_state = next(train_loader)
        else:
            total_loss = mx.array(0.0)
            accumulated_grads = None

            for _micro_step in range(grad_accum_steps):
                loss, grads = loss_and_grad_fn(model, x, y)
                total_loss = total_loss + loss

                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = mlx.utils.tree_map(
                        lambda a, b: a + b, accumulated_grads, grads
                    )
                # Materialize per micro-step to prevent unbounded graph growth
                mx.eval(total_loss, *[v for _, v in mlx.utils.tree_flatten(accumulated_grads)])
                x, y, dataloader_state = next(train_loader)

            accumulated_grads = mlx.utils.tree_map(
                lambda g: g / grad_accum_steps, accumulated_grads
            )
            loss = total_loss / grad_accum_steps
            optimizer.update(model, accumulated_grads)
            mx.eval(model.parameters(), optimizer.state, loss)  # mx.eval materializes MLX lazy arrays

        t1 = time.time()
        dt = t1 - t0

        # Logging
        train_loss_f = loss.item()
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased = smooth_train_loss / (1 - ema_beta**(step + 1))

        if step > 10:
            total_training_time += dt

        pct = 100 * step / num_iterations
        tok_per_sec = int(actual_batch_size / dt)

        eta_str = ""
        if step > 10:
            avg_dt = total_training_time / (step - 10)
            eta_sec = avg_dt * (num_iterations - step)
            eta_str = f" | eta: {eta_sec/60:.1f}m"

        epoch_info = f"pq:{dataloader_state['pq_idx']} rg:{dataloader_state['rg_idx']}"
        print(f"step {step:05d}/{num_iterations:05d} ({pct:.1f}%) | loss: {debiased:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,} | {epoch_info}{eta_str}")

        step += 1

    print(f"\nTotal training time: {total_training_time/60:.2f}m")
    return model
