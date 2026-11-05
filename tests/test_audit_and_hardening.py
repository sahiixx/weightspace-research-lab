"""Tests for reproducibility, deliberate rejection, and checkpoint integrity."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from wslab.audit import run_negative_controls, run_sweep
from wslab.checkpoints import fingerprint
from wslab.models import corrupted_model
from wslab.safety import PatchRejected, apply_checked
from wslab.search import propose
from wslab.data import samples


def test_sweep_is_reproducible_and_gates_holdout(tmp_path):
    one = run_sweep(tmp_path / 'one', base_seed=10, seeds=3, count=256)
    two = run_sweep(tmp_path / 'two', base_seed=10, seeds=3, count=256)
    assert one == two
    assert all(r['selection_split'] == 'validation' and r['changed_elements'] == 1 for r in one['runs'])
    assert all(r['test']['after'] > r['test']['before'] for r in one['runs'])
    assert all(r['audit']['after'] > r['audit']['before'] for r in one['runs'])
    # The validation-selected coefficient may overfit a small sample: do not require perfection.


def test_negative_controls_and_original_unchanged(tmp_path):
    before = fingerprint(corrupted_model().state_dict())
    report = run_negative_controls(tmp_path)
    assert report['passed'] and report['controls'] == 10
    assert report['rejected'] == 10 and report['base_unchanged']
    assert fingerprint(corrupted_model().state_dict()) == before


def test_boolean_schema_version_rejected():
    base = corrupted_model().state_dict()
    patch, _ = propose(base, samples('validation', count=128))
    patch['schema_version'] = True
    with pytest.raises(PatchRejected, match='schema'):
        apply_checked(base, patch)


def test_cli_apply_refuses_overwrite_original(tmp_path):
    from wslab.checkpoints import save_checkpoint, write_json
    base = corrupted_model().state_dict()
    base_path = tmp_path / 'original.pt'
    patch_path = tmp_path / 'patch.json'
    save_checkpoint(base_path, base)
    patch, _ = propose(base, samples('validation', count=128))
    write_json(patch_path, patch)
    result = subprocess.run([sys.executable, '-m', 'wslab', 'apply',
                             '--checkpoint', str(base_path), '--patch', str(patch_path),
                             '--out', str(base_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'overwrite' in result.stderr.lower()
    assert fingerprint(base) == fingerprint(corrupted_model().state_dict())


def test_negative_controls_cli(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'wslab', 'negative-controls',
                             '--out', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)['rejected'] == 10
