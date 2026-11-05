import torch
from wslab.models import reference_model, corrupted_model
from wslab.data import samples
from wslab.benchmarks import evaluate


def test_reference_fits_known_rule_and_corruption_hurts():
    val = samples("validation")
    ref = evaluate(reference_model(), val)
    damaged = evaluate(corrupted_model(), val)
    assert ref.accuracy == 1.0
    assert damaged.accuracy < 0.85
    assert damaged.accuracy < ref.accuracy


def test_data_reproducible_separate_splits():
    a = samples("validation", seed=19)
    b = samples("validation", seed=19)
    test = samples("test", seed=19)
    assert torch.equal(a.x, b.x) and torch.equal(a.y, b.y)
    assert not torch.equal(a.x, test.x)
    assert test.y.dtype == torch.int64


def test_no_trainable_computation_in_evaluate():
    model = corrupted_model()
    for p in model.parameters():
        p.grad = None
    evaluate(model, samples("audit"))
    assert all(p.grad is None for p in model.parameters())
