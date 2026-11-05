"""Deliberately tiny and transparent demonstration model.

The reference classifier encodes y = 1 iff 2*x0 + x1 > 0, as two logits.
The base checkpoint is corrupted by changing one parameter. No training occurs.
"""
from __future__ import annotations

import torch
from torch import nn


class ToyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def reference_model() -> ToyClassifier:
    model = ToyClassifier()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[-2.0, -1.0], [2.0, 1.0]]))
        model.linear.bias.zero_()
    model.eval()
    return model


def corrupted_model() -> ToyClassifier:
    model = reference_model()
    with torch.no_grad():
        model.linear.weight[1, 0] = -2.0
    return model


def model_from_state(state: dict[str, torch.Tensor]) -> ToyClassifier:
    model = ToyClassifier()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
