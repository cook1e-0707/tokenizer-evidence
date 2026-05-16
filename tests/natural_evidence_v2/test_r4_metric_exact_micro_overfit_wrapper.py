from __future__ import annotations

from pathlib import Path


WRAPPER = Path("scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch")


def test_micro_overfit_wrapper_exposes_surface_margin_loss_mode() -> None:
    text = WRAPPER.read_text(encoding="utf-8")

    assert 'SURFACE_MARGIN_LOSS_MODE="${SURFACE_MARGIN_LOSS_MODE:-mass_relu}"' in text
    assert 'echo "surface_margin_loss_mode=$SURFACE_MARGIN_LOSS_MODE"' in text
    assert '--surface-margin-loss-mode "$SURFACE_MARGIN_LOSS_MODE"' in text


def test_micro_overfit_wrapper_exposes_task_ce_weight() -> None:
    text = WRAPPER.read_text(encoding="utf-8")

    assert 'TASK_CE_WEIGHT="${TASK_CE_WEIGHT:-1.0}"' in text
    assert 'echo "task_ce_weight=$TASK_CE_WEIGHT"' in text
    assert '--task-ce-weight "$TASK_CE_WEIGHT"' in text


def test_micro_overfit_wrapper_remains_h200_pomplun() -> None:
    text = WRAPPER.read_text(encoding="utf-8")

    assert "#SBATCH --partition=pomplun" in text
    assert "#SBATCH --account=cs_yinxin.wan" in text
    assert "#SBATCH --qos=pomplun" in text
    assert "#SBATCH --gres=gpu:h200:1" in text
    assert "#SBATCH --time=30-00:00:00" in text
