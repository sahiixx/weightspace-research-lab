"""Offline publication policy checks; never access GitHub in tests."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'publish_to_github.sh'


def test_git_publisher_has_valid_bash_syntax():
    if not shutil.which('bash'):
        return
    result = subprocess.run(['bash', '-n', str(SCRIPT)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_git_publisher_refuses_invalid_repository_without_network():
    if not shutil.which('bash'):
        return
    result = subprocess.run(['bash', str(SCRIPT), 'invalid-repository-name'], text=True, capture_output=True)
    assert result.returncode == 2
    assert 'Usage' in result.stderr


def test_git_publisher_refuses_invalid_visibility_without_network():
    if not shutil.which('bash'):
        return
    result = subprocess.run(['bash', str(SCRIPT), 'owner/repository', '--unlisted'], text=True, capture_output=True)
    assert result.returncode == 2
    assert 'Visibility must be' in result.stderr


def test_github_ci_is_explicitly_read_only():
    workflow = (ROOT / '.github' / 'workflows' / 'ci.yml').read_text()
    assert 'permissions:\n  contents: read' in workflow
    assert 'actions/upload-artifact@v4' in workflow
    assert 'bash scripts/run_all.sh runs/ci-audit' in workflow
