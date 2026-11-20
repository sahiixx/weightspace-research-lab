"""Offline publication policy checks; never access GitHub in tests."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'publish_to_github.sh'


def run_bash(*args):
    bash = shutil.which('bash')
    if bash is None:
        pytest.skip('Bash is required for publisher checks')
    # stdin avoids Windows paths in WSL; bytes preserve LF through Windows pipes.
    script = SCRIPT.read_text(encoding='utf-8').encode('utf-8')
    return subprocess.run([bash, *args], input=script, capture_output=True, timeout=30)


def test_git_publisher_has_valid_bash_syntax():
    result = run_bash('-n')
    assert result.returncode == 0, result.stderr


def test_git_publisher_refuses_invalid_repository_without_network():
    result = run_bash('-s', '--', 'invalid-repository-name')
    assert result.returncode == 2
    assert b'Usage' in result.stderr


def test_git_publisher_refuses_invalid_visibility_without_network():
    result = run_bash('-s', '--', 'owner/repository', '--unlisted')
    assert result.returncode == 2
    assert b'Visibility must be' in result.stderr


def test_github_ci_is_explicitly_read_only():
    workflow = (ROOT / '.github' / 'workflows' / 'ci.yml').read_text()
    assert 'permissions:\n  contents: read' in workflow
    assert 'actions/upload-artifact@' in workflow
    action_refs = re.findall(r'uses:\s+(\S+)', workflow)
    assert action_refs
    assert all(re.fullmatch(r'actions/[a-z-]+@[0-9a-f]{40}', ref) for ref in action_refs)
    assert 'bash scripts/run_all.sh runs/ci-audit' in workflow
