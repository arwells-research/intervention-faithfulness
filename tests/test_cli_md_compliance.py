from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "intervention_faithfulness", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_help_has_documented_flags() -> None:
    p = _run(["diagnose", "--help"])
    assert p.returncode == 0, f"diagnose --help failed\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    out = p.stdout
    for needle in ["--csv", "--out-dir", "--prefix"]:
        assert needle in out, f"missing {needle} in diagnose --help\nstdout:\n{out}"


def test_cli_diagnose_emits_expected_bundle_files(tmp_path: Path) -> None:
    # Create a stable CSV input
    from intervention_faithfulness.core.FaithfulnessTest import FaithfulnessTest

    t = FaithfulnessTest.from_plugin("faithful_synthetic", None)
    csv_path = tmp_path / "trials.csv"
    t.trials_df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    p = _run(
        [
            "diagnose",
            "--csv",
            str(csv_path),
            "--out-dir",
            str(out_dir),
            "--prefix",
            "demo",
            "--certificate-json",
        ]
    )
    assert p.returncode == 0, f"diagnose failed\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"

    assert (out_dir / "demo_diagnosis.json").exists(), _missing_file_debug(out_dir, p, "demo_diagnosis.json")
    assert (out_dir / "demo_certificate.json").exists(), _missing_file_debug(out_dir, p, "demo_certificate.json")

    expected_breakdown = out_dir / "demo_breakdown.csv"
    if not expected_breakdown.exists():
        # Show what was actually emitted so we can align naming to the contract.
        raise AssertionError(_missing_file_debug(out_dir, p, "demo_breakdown.csv"))


def _missing_file_debug(out_dir: Path, p: subprocess.CompletedProcess[str], missing_name: str) -> str:
    files = sorted([q.name for q in out_dir.glob("*")]) if out_dir.exists() else []
    return (
        f"missing expected output file: {missing_name}\n"
        f"out_dir: {out_dir}\n"
        f"out_dir contents: {files}\n"
        f"stdout:\n{p.stdout}\n"
        f"stderr:\n{p.stderr}\n"
    )

