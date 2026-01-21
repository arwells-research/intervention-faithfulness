import json
import pandas as pd
import re

from intervention_faithfulness.core.FaithfulnessTest import FaithfulnessTest, DiagnoseConfig


def test_export_artifacts_writes_bundle(tmp_path):
    n = 240
    df = pd.DataFrame(
        {
            "trial_id": list(range(n)),
            "intervention_id": ([0.0] * (n // 3)) + ([1.0] * (n // 3)) + ([2.0] * (n - 2 * (n // 3))),
            "state_x": [1] * n,
            "history_h": ([0] * (n // 2)) + ([1] * (n - (n // 2))),
        }
    )
    df["outcome"] = ([0.1] * (n // 2)) + ([0.9] * (n - (n // 2)))

    cfg = DiagnoseConfig(
        min_samples=20,
        tail_mode=False,
        recommend=True,
        recommend_top_k=5,
    )
    res = FaithfulnessTest(df).diagnose(cfg)

    out_dir = tmp_path / "bundle"
    paths = res.export_artifacts(
        str(out_dir),
        prefix="t0",
        include_trials=True,
        include_certificate=True,        
        envelope_table_kwargs={
            "x_col": "intervention_id",
            "y_col": None,
            "bins_x": 6,
            "bins_y": 1,
            "min_samples": 20,
            "faithfulness": True,
            "threshold": 0.7,
        },
        include_map=True,
        map_kwargs={
            "x_col": "intervention_id",
            "y_col": None,
            "bins_x": 6,
            "bins_y": 1,
            "min_samples": 20,
            "faithfulness": True,
        },
        include_recommendations=True,
    )

    assert "diagnosis_json" in paths
    assert "certificate_json" in paths
    assert "trials_csv" in paths
    assert "safe_envelope_csv" in paths
    assert "recommended_features_csv" in paths
    assert "faithfulness_map_png" in paths

    # Files exist and are non-empty
    for k, p in paths.items():
        fp = out_dir / (p.split("/")[-1])
        assert fp.exists(), f"missing {k}"
        assert fp.stat().st_size > 0, f"empty {k}"

    # Certificate should include artifact hashes
    cert = json.load(open(paths["certificate_json"], "r", encoding="utf-8"))
    hashes = cert["provenance"]["artifact_hashes"]
    assert "diagnosis_json_sha256" in hashes
    assert "trials_csv_sha256" in hashes

    # Basic sanity: sha256 hex length
    hex64 = re.compile(r"^[0-9a-f]{64}$")
    assert hex64.match(hashes["diagnosis_json_sha256"])
    assert hex64.match(hashes["trials_csv_sha256"])

    # Diagnosis should embed trials table hash
    diag = json.load(open(paths["diagnosis_json"], "r", encoding="utf-8"))
    assert "trials_table_sha256" in diag
    assert hex64.match(diag["trials_table_sha256"])

    # diagnosis JSON parses
    with open(paths["diagnosis_json"], "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert "fracture_score" in payload