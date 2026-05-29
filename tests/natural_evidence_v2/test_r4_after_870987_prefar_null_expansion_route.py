from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/natural_evidence_v2/validate_r4_after_870987_prefar_null_expansion_route.py"


def test_prefar_null_route_validation_passes_current_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "validation"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(out),
        ],
        check=True,
    )
    summary = json.loads((out / "route_validation_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS_R4_AFTER_870987_PREFAR_NULL_EXPANSION_ROUTE_PLAN_ONLY_NO_SUBMIT"
    assert summary["additional_control_blocks_required_per_arm"] == {
        "raw": 160,
        "task_only": 160,
        "wrong_key": 160,
        "wrong_payload": 160,
    }
    assert summary["organic_null_target_block_equivalent"] == 256
    assert summary["slurm_allowed"] is False
