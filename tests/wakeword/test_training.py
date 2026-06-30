import numpy as np
import pytest

# Skip the whole file when torch isn't installed (CI runs without the training
# stack — see requirements-server-text.txt). Local dev installs torch for
# wakeword/train.py and these tests run normally.
torch = pytest.importorskip("torch")

from wakeword._training import (
    WakewordModel, hard_negative_filter, build_neg_weight_schedule,
)


def test_wakeword_model_takes_16x96_returns_scalar():
    model = WakewordModel(layer_dim=128)
    x = torch.zeros(4, 16, 96)
    out = model(x)
    assert out.shape == (4, 1)


def test_wakeword_model_inference_outputs_sigmoid_in_0_1():
    model = WakewordModel(layer_dim=128, inference=True)
    x = torch.randn(8, 16, 96)
    out = model(x)
    assert (out >= 0).all() and (out <= 1).all()


def test_hard_negative_filter_drops_easy_negatives():
    # Easy negatives = (label=0, sigmoid pred close to 0) — should be dropped
    # Easy positives = (label=1, sigmoid pred close to 1) — should be dropped
    # Hard samples in middle — should be kept
    preds_sigmoid = torch.tensor([0.0005, 0.5, 0.9999, 0.3, 0.7])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])
    mask = hard_negative_filter(preds_sigmoid, labels, low=0.001, high=0.999)
    # idx 0: neg, p=0.0005 < 0.001 → drop
    # idx 1: neg, p=0.5 → keep
    # idx 2: pos, p=0.9999 > 0.999 → drop
    # idx 3: pos, p=0.3 → keep (hard pos)
    # idx 4: neg, p=0.7 → keep (hard neg)
    expected = torch.tensor([False, True, False, True, True])
    assert torch.equal(mask, expected)


def test_neg_weight_schedule_ramps_up():
    sched = build_neg_weight_schedule(epochs=50, start=1.0, end=4.0)
    assert len(sched) == 50
    assert sched[0] == 1.0
    assert sched[-1] == 4.0
    # Monotonic non-decreasing
    for i in range(1, 50):
        assert sched[i] >= sched[i-1]
