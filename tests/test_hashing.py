import json
import re
import pandas as pd

from intervention_faithfulness.core.FaithfulnessTest import FaithfulnessTest


def test_certificate_includes_hashes_for_diagnosis_and_trials(tmp_path):
    df = pd.DataFrame(
        {
            "trial_id": [1, 2, 3, 4],
            "intervention_id": ["A", "A", "B", "B"],
            "outcome": [0.1, 0.2, 0.3, 0.4],
            "state_x": [1, 1, 1, 1],
            "history_h": [0, 1, 0, 1],
        }
    )

    res = FaithfulnessTest(df).diagnose()
    paths = res.export_artifacts(
        str(tmp_path),
        prefix="h0",
        include_trials=True,
        include_certificate=True,
        include_map=False,
    )

    hex64 = re.compile(r"^[0-9a-f]{64}$")

    cert = json.load(open(paths["certificate_json"], "r", encoding="utf-8"))
    hashes = cert["provenance"]["artifact_hashes"]

    assert "diagnosis_json_sha256" in hashes
    assert "trials_csv_sha256" in hashes
    assert hex64.match(hashes["diagnosis_json_sha256"])
    assert hex64.match(hashes["trials_csv_sha256"])

    diag = json.load(open(paths["diagnosis_json"], "r", encoding="utf-8"))
    assert "trials_table_sha256" in diag
    assert hex64.match(diag["trials_table_sha256"])