"""Agent tools must not expose authorization or access private keys."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_codex_tools_contain_no_signing_or_apply_calls():
    for name in ("propose.py", "inspect_proposal.py"):
        data = (ROOT / "codex" / "tools" / name).read_text()
        assert "private_key" not in data
        assert "sign_patch(" not in data
        assert "verify_signed_patch(" not in data
        assert "wslab apply" not in data


def test_codex_inspector_fails_closed_on_missing_input(tmp_path):
    process = subprocess.run([sys.executable, str(ROOT / "codex/tools/inspect_proposal.py"),
                              "--checkpoint", str(tmp_path / "no.pt"),
                              "--patch", str(tmp_path / "no.json")],
                             capture_output=True, text=True)
    assert process.returncode != 0
    assert "NOT_APPROVED" in process.stdout
