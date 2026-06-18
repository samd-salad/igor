"""Verify boundary_check returns 0 on the current scaffolds."""
import subprocess
import sys


def test_boundary_check_passes_on_scaffold():
    result = subprocess.run(
        [sys.executable, "-m", "tools.boundary_check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "passed" in result.stdout.lower()
