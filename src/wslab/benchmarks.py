"""Pure forward-pass scoring. No weight fitting or gradients."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
from torch.nn import functional as F
from .data import Samples
from .models import ToyClassifier


@dataclass(frozen=True)
class Metrics:
    split: str
    accuracy: float
    cross_entropy: float
    count: int
    correct: int
    group_accuracy: dict[str, float]
    confusion: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(model: ToyClassifier, data: Samples) -> Metrics:
    model.eval()
    with torch.inference_mode():
        logits = model(data.x)
        pred = torch.argmax(logits, dim=1)
        correct = pred.eq(data.y)
        n = len(data.y)
        ce = float(F.cross_entropy(logits, data.y).item())
        groups = {}
        for gid, name in ((0, "x0_negative"), (1, "x0_nonnegative")):
            mask = data.groups == gid
            groups[name] = float(correct[mask].float().mean().item())
        confusion = {
            f"true_{actual}_pred_{guessed}": int(((data.y == actual) & (pred == guessed)).sum().item())
            for actual in (0, 1) for guessed in (0, 1)
        }
        return Metrics(data.split, float(correct.float().mean().item()), ce,
                       n, int(correct.sum().item()), groups, confusion)
