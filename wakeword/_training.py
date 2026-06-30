"""Model architecture + training loop. Mirrors dscripka's reference:
2-layer FC + LayerNorm, no dropout, lr=1e-4, warmup+cosine schedule,
hard-negative mining, negative-weight ramp."""
from __future__ import annotations
import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


def build_neg_weight_schedule(epochs: int,
                              start: float = 1.0,
                              end: float = 8.0) -> list[float]:
    """Linear ramp of the negative-class weight from start at epoch 0 to end
    at the final epoch. Pushes the model to be more confident on negatives
    later in training.

    History: v0.7.1 tried end=2.0 (with epochs=30, low=0.05) and the model
    catastrophically undertrained (28% holdout FP, positives collapsed to
    min=0.06). Reverted to v0.7's end=8.0 pending diagnostic-curve evidence
    on where to actually land."""
    if epochs <= 1:
        return [end]
    step = (end - start) / (epochs - 1)
    return [start + i * step for i in range(epochs)]


def hard_negative_filter(preds_sigmoid: torch.Tensor,
                         labels: torch.Tensor,
                         low: float = 0.001,
                         high: float = 0.999) -> torch.Tensor:
    """Boolean mask selecting samples that should contribute to the gradient.
    Drops trivial samples: labels==0 with pred<low (easy negs already classified)
    and labels==1 with pred>high (easy positives already classified).

    History: v0.7.1 tried low=0.05 alongside other softening; model
    catastrophically undertrained. Reverted to v0.7's 0.001 pending
    diagnostic-curve evidence."""
    labels_flat = labels.view(-1)
    preds_flat = preds_sigmoid.view(-1)
    is_easy_neg = (labels_flat == 0) & (preds_flat < low)
    is_easy_pos = (labels_flat == 1) & (preds_flat > high)
    return ~(is_easy_neg | is_easy_pos)


class WakewordModel(nn.Module):
    """Reference DNN: Flatten -> Linear -> LayerNorm -> ReLU -> Linear ->
    LayerNorm -> ReLU -> Linear -> [Sigmoid if inference else raw logit].
    Defaults match dscripka's training_models.ipynb middle ground:
    layer_dim=128, no dropout."""

    def __init__(self, layer_dim: int = 128, inference: bool = False):
        super().__init__()
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 96, layer_dim),
            nn.LayerNorm(layer_dim), nn.ReLU(),
            nn.Linear(layer_dim, layer_dim),
            nn.LayerNorm(layer_dim), nn.ReLU(),
            nn.Linear(layer_dim, 1),
        )
        self._inference = inference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        if self._inference:
            return torch.sigmoid(out)
        return out

    def as_inference(self) -> "WakewordModel":
        """Return a clone wired for sigmoid output, suitable for ONNX export."""
        clone = WakewordModel(
            layer_dim=self.body[1].out_features, inference=True
        )
        clone.body.load_state_dict(self.body.state_dict())
        return clone


def _warmup_then_cosine(optimizer, num_steps: int,
                        warmup_frac: float = 0.1,
                        min_lr: float = 1e-5):
    """LR schedule: warmup over first warmup_frac of steps from min_lr to
    base_lr, then cosine decay back to min_lr."""
    base_lr = optimizer.param_groups[0]["lr"]
    warmup_steps = max(1, int(num_steps * warmup_frac))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return min_lr + (base_lr - min_lr) * (step / warmup_steps)
        progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cos

    return lr_at


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    layer_dim: int = 128,
    log_every: int = 5,
    print_fn: Callable[[str], None] = print,
    on_epoch_end: Callable[[int, "WakewordModel"], None] | None = None,
    pos_weight: float | None = None,
) -> WakewordModel:
    """Train the wakeword classifier with hard-negative mining + neg-weight ramp.
    Returns the inference-wired model (sigmoid output).

    pos_weight: if None, defaults to n_neg/n_pos (our historical setting, which
    inverts the canonical openWakeWord loss balance — see v0.9 research findings).
    Pass pos_weight=1.0 to match dscripka's reference recipe, where negative
    dominance is achieved via the neg_w ramp alone.
    """
    torch.manual_seed(42)
    model = WakewordModel(layer_dim=layer_dim, inference=False)

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float().unsqueeze(1)

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both positive and negative samples.")
    pos_weight_value = pos_weight if pos_weight is not None else (n_neg / n_pos)
    print_fn(f"  Class imbalance: pos={n_pos}, neg={n_neg}, pos_weight={pos_weight_value:.2f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps = epochs * max(1, (len(X) + batch_size - 1) // batch_size)
    lr_at = _warmup_then_cosine(optimizer, total_steps)

    neg_weight_sched = build_neg_weight_schedule(epochs)

    # Static pos_weight handles class imbalance; neg_w is applied per-sample below.
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value]),
        reduction="none",
    )

    step = 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        epoch_loss, n_batches, n_kept_total, n_total = 0.0, 0, 0, 0

        neg_w = neg_weight_sched[epoch]

        for i in range(0, len(X_t), batch_size):
            idx = perm[i:i + batch_size]
            xb = X_t[idx]
            yb = y_t[idx]

            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            step += 1

            logits = model(xb)
            with torch.no_grad():
                preds_sig = torch.sigmoid(logits)
                mask = hard_negative_filter(preds_sig, yb)

            losses = loss_fn(logits, yb).view(-1)

            # Apply neg_w as a per-sample multiplier on negative-label rows ONLY.
            # =1 for positives, =neg_w for negatives. Ramps 1.0→4.0 over epochs
            # to push the model to be more confident on negatives later in training.
            neg_mask = (yb.view(-1) == 0).float()
            sample_weights = 1.0 + (neg_w - 1.0) * neg_mask
            losses = losses * sample_weights

            kept = losses[mask]
            n_total += len(losses)
            n_kept_total += int(mask.sum())
            if len(kept) == 0:
                continue
            loss = kept.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % log_every == 0 or epoch == 0:
            kept_pct = 100 * n_kept_total / max(1, n_total)
            print_fn(f"  Epoch {epoch+1:3d}/{epochs}  "
                     f"loss={epoch_loss / max(1, n_batches):.4f}  "
                     f"kept={kept_pct:.1f}%  neg_w={neg_w:.2f}  "
                     f"lr={optimizer.param_groups[0]['lr']:.5f}")

        if on_epoch_end is not None:
            on_epoch_end(epoch + 1, model)

    return model.as_inference()
