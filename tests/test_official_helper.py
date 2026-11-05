"""Unit tests for the optional official-hash helper, not for Qwen performance."""
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'verify_official_qwen.py'
spec = importlib.util.spec_from_file_location('verify_official_qwen_helper', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_bfloat16_hash_uses_raw_bytes_not_float32_conversion():
    tensor = torch.tensor([1.0, 0.1, -12.0], dtype=torch.bfloat16)
    raw_sha = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
    assert mod.tensor_hash(tensor) == raw_sha
    assert raw_sha != hashlib.sha256(tensor.float().numpy().tobytes()).hexdigest()


def test_shard_reader_selects_expected_tensors(tmp_path):
    pytest.importorskip('safetensors')
    from safetensors.torch import save_file
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    save_file({'wanted': tensor, 'unused': torch.zeros(1)}, str(tmp_path / 'part.safetensors'))
    tensors = mod.read_named_tensors(tmp_path, {'wanted'})
    assert set(tensors) == {'wanted'} and torch.equal(tensors['wanted'], tensor)
    with pytest.raises(ValueError, match='Missing tensors'):
        mod.read_named_tensors(tmp_path, {'missing'})
