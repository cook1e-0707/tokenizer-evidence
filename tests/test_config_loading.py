from pathlib import Path

import pytest

from src.infrastructure.config import ConfigError, load_config, save_resolved_config
from src.infrastructure.paths import discover_repo_root


def test_config_loading_supports_includes_runtime_and_overrides(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    config = load_config(
        repo_root / "configs" / "experiment" / "exp_alignment.yaml",
        overrides=[
            "run.seed=99",
            "run.method_name=baseline_kgw",
            "runtime.resources.mem_gb=48",
        ],
    )
    assert config.seed == 99
    assert config.method_name == "baseline_kgw"
    assert config.runtime.resources.mem_gb == 48
    assert len(config.source_config_paths) >= 3

    output_path = tmp_path / "config.resolved.yaml"
    save_resolved_config(config, output_path)
    assert output_path.exists()


def test_config_validation_rejects_missing_sections(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("model:\n  name: missing-run\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Missing required section: run"):
        load_config(bad_config)


def test_config_validation_rejects_conflicting_output_roots(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad_output_root.yaml"
    bad_config.write_text(
        "\n".join(
            [
                "run:",
                "  experiment_name: exp_alignment",
                "  mode: train",
                "  method: our_method",
                "  seed: 1",
                "  output_root: results/raw",
                "model:",
                "  name: tiny-debug",
                "data:",
                "  name: synthetic-smoke",
                "runtime:",
                "  output_root: somewhere_else",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="run.output_root and runtime.output_root disagree"):
        load_config(bad_config)
