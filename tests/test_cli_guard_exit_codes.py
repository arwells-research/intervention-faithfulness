# tests/test_cli_guard_exit_codes.py
"""
CLI smoke tests for Σ₂-I guard exit codes.

Contract:
- OK      -> exit code 0
- BOUNDARY-> exit code 2
- REFUSE  -> exit code 3

These are subprocess-level tests: they exercise __main__.py + cli.py parsing,
CSV loading, diagnosis, policy, JSON emission, and the final process exit code.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    # tests/ is directly under repo root per tree
    return Path(__file__).resolve().parents[1]


def _run_guard(csv_path: Path, extra_args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "intervention_faithfulness",
        "guard",
        "--csv",
        str(csv_path),
        "--fracture-threshold",
        "0.12",
        "--min-effective-samples",
        "200",
        "--no-recommend",
    ]
    if extra_args:
        cmd.extend(list(extra_args))
    return subprocess.run(
        cmd,
        cwd=str(_repo_root()),
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_guard_exit_ok(tmp_path: Path) -> None:
    """
    Faithful synthetic should be OK (0).
    """
    from intervention_faithfulness.core.FaithfulnessTest import FaithfulnessTest

    t = FaithfulnessTest.from_plugin("faithful_synthetic", None)
    csv_path = tmp_path / "faithful.csv"
    t.trials_df.to_csv(csv_path, index=False)

    p = _run_guard(csv_path)
    assert p.returncode == 0, f"expected exit 0, got {p.returncode}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"


def test_cli_guard_exit_refuse(tmp_path: Path) -> None:
    """
    Unfaithful-cut positive control should be REFUSE (3).

    We use unfaithful_cut_synthetic here because it is designed as a stable
    positive control for a high fracture under intervention.
    """
    from intervention_faithfulness.core.FaithfulnessTest import FaithfulnessTest

    t = FaithfulnessTest.from_plugin("unfaithful_cut_synthetic", None)
    csv_path = tmp_path / "unfaithful.csv"
    t.trials_df.to_csv(csv_path, index=False)

    p = _run_guard(csv_path, extra_args=["--no-envelope"])
    assert p.returncode == 3, f"expected exit 3, got {p.returncode}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"

def test_cli_guard_exit_boundary_underpowered(tmp_path: Path) -> None:
    """
    Underpowered inputs must never be OK. Expect BOUNDARY (2).
    """
    n = 40  # intentionally below min-effective-samples=200
    df = pd.DataFrame(
        {
            "trial_id": range(n),
            "intervention_id": ["I0"] * (n // 2) + ["I1"] * (n - n // 2),
            "outcome": [0.0] * n,
            "state_s": [0.0] * n,
        }
    )
    csv_path = tmp_path / "underpowered.csv"
    df.to_csv(csv_path, index=False)

    p = _run_guard(csv_path)
    assert p.returncode == 2, f"expected exit 2, got {p.returncode}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
