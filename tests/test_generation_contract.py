import sys
import types
from pathlib import Path

import pytest
import yaml

from src.core.bucket_mapping import BucketLayout, FieldBucketSpec, save_bucket_layout
from src.core.canonical_contract import (
    CanonicalContractError,
    ensure_train_eval_config_alignment,
    teacher_forced_sanity_check,
)
from src.infrastructure.config import load_experiment_config
from src.infrastructure.paths import discover_repo_root
from src.training.dataset import TrainingExample
from src.training.hf_causal_lm import run_minimal_hf_causal_lm_training


def _write_frozen_catalog(path: Path, *, include_topic: bool = True) -> Path:
    fields = [
        FieldBucketSpec(
            field_name="SECTION",
            buckets={0: ("news",), 1: ("report",), 2: ("guide",), 3: ("update", "review")},
        ),
    ]
    if include_topic:
        fields.append(
            FieldBucketSpec(
                field_name="TOPIC",
                buckets={0: ("market",), 1: ("travel",), 2: ("health",), 3: ("science", "climate")},
            )
        )
    layout = BucketLayout(
        fields=tuple(fields),
        catalog_name="generation-contract-test-catalog",
        provenance={
            "catalog_status": "frozen",
            "freeze_status": "strict_passed",
            "tokenizer_name": "gpt2",
            "tokenizer_backend": "huggingface",
            "tokenizer_revision_source": "gpt2",
            "source_catalog": str(path.with_name("source.yaml")),
            "freeze_timestamp": "20260418T000000Z",
            "git_commit": "abc123",
        },
    )
    save_bucket_layout(layout, path)
    return path


def _write_experiment_config(path: Path, *, catalog_path: Path, experiment_name: str) -> Path:
    payload = {
        "run": {
            "experiment_name": experiment_name,
            "mode": "train" if experiment_name == "exp_train" else "eval",
            "method": "our_method",
            "seed": 17,
        },
        "model": {
            "name": "tiny-debug",
            "family": "synthetic",
            "tokenizer_name": "synthetic-tokenizer",
            "max_length": 128,
        },
        "data": {
            "name": "real-pilot",
            "carrier_catalog_path": str(catalog_path),
        },
        "train": {
            "target_mode": "canonical_evidence",
            "generation_prompt": "Emit canonical ownership evidence only:",
            "generation_max_new_tokens": 16,
            "generation_stop_strings": ["\n\n"],
        },
        "eval": {
            "verification_mode": "canonical_render",
            "render_format": "canonical_v1",
            "payload_text": "OK",
        },
        "runtime": {
            "output_root": str(path.parent / "results"),
            "launcher_mode": "local",
            "resources": {
                "partition": "Intel",
                "num_gpus": 0,
                "cpus": 2,
                "mem_gb": 8,
                "time_limit": "00:30:00",
            },
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_train_and_eval_schema_alignment_fails_loudly_on_catalog_divergence(tmp_path: Path) -> None:
    train_catalog = _write_frozen_catalog(tmp_path / "train_catalog.yaml", include_topic=True)
    eval_catalog = _write_frozen_catalog(tmp_path / "eval_catalog.yaml", include_topic=False)
    train_config = load_experiment_config(
        _write_experiment_config(tmp_path / "train.yaml", catalog_path=train_catalog, experiment_name="exp_train")
    )
    eval_config = load_experiment_config(
        _write_experiment_config(tmp_path / "eval.yaml", catalog_path=eval_catalog, experiment_name="exp_eval")
    )

    with pytest.raises(CanonicalContractError, match="canonical contract mismatch"):
        ensure_train_eval_config_alignment(train_config, eval_config, tmp_path)


def test_repo_batch26_train_and_eval_configs_share_canonical_contract() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    train_config = load_experiment_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_train__gpt2__v1.yaml"
    )
    eval_config = load_experiment_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_eval__gpt2__v1.yaml"
    )

    ensure_train_eval_config_alignment(train_config, eval_config, repo_root)


def test_teacher_forced_canonical_block_is_accepted(tmp_path: Path) -> None:
    catalog_path = _write_frozen_catalog(tmp_path / "catalog.yaml")
    config = load_experiment_config(
        _write_experiment_config(tmp_path / "train.yaml", catalog_path=catalog_path, experiment_name="exp_train")
    )

    bundle, result = teacher_forced_sanity_check(config, tmp_path)

    assert result.success is True
    assert result.decoded_payload == "OK"
    assert bundle.contract.field_names == ("SECTION", "TOPIC")
    assert bundle.contract.block_count == 4


def test_canonical_generation_is_deterministic_and_length_bounded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeLoss:
        def detach(self):
            return self

        def cpu(self):
            return self

        def item(self):
            return 0.123

        def backward(self):
            return None

    class FakeTensor:
        def __init__(self, payload):
            self.payload = payload

        def to(self, _device):
            return self

        def __getitem__(self, index):
            return self.payload[index]

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeOptimizer:
        def zero_grad(self, set_to_none=True):
            return None

        def step(self):
            return None

    class FakeTokenizer:
        last_instance = None

        def __init__(self):
            self.pad_token = None
            self.eos_token = "<eos>"
            self.pad_token_id = 0
            self.last_prompt = ""

        @classmethod
        def from_pretrained(cls, _name):
            cls.last_instance = cls()
            return cls.last_instance

        def __call__(self, text, **_kwargs):
            if isinstance(text, str):
                self.last_prompt = text
            return {
                "input_ids": FakeTensor([[1, 2, 3]]),
                "attention_mask": FakeTensor([[1, 1, 1]]),
            }

        def decode(self, _tokens, skip_special_tokens=True):
            return f"{self.last_prompt} SECTION=report; TOPIC=market\n\nTrailing prose"

        def save_pretrained(self, path: Path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeModel:
        last_instance = None

        def __init__(self):
            self.config = types.SimpleNamespace(pad_token_id=None)
            self.generate_kwargs = None

        @classmethod
        def from_pretrained(cls, _name):
            cls.last_instance = cls()
            return cls.last_instance

        def to(self, _device):
            return self

        def train(self):
            return None

        def eval(self):
            return None

        def parameters(self):
            return []

        def __call__(self, **_kwargs):
            return types.SimpleNamespace(loss=FakeLoss())

        def save_pretrained(self, path: Path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "config.json").write_text("{}", encoding="utf-8")

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            return [[10, 11, 12]]

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.device = lambda name: name
    fake_torch.no_grad = lambda: FakeNoGrad()
    fake_torch.optim = types.SimpleNamespace(AdamW=lambda params, lr: FakeOptimizer())

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = FakeModel
    fake_transformers.AutoTokenizer = FakeTokenizer

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    result = run_minimal_hf_causal_lm_training(
        model_name_or_path="gpt2",
        max_length=64,
        dataset=[
            TrainingExample(
                prompt="Emit canonical ownership evidence only:",
                target_symbols=(),
                metadata={"completion": "SECTION=report; TOPIC=market\n\n"},
            )
        ],
        batch_size=1,
        epochs=1,
        learning_rate=1.0e-4,
        run_dir=tmp_path,
        require_cuda=False,
        generation_prompt="Emit canonical ownership evidence only:",
        generation_max_new_tokens=5,
        generation_stop_strings=("\n\n",),
    )

    assert result.generated_text == "SECTION=report; TOPIC=market"
    assert FakeModel.last_instance is not None
    assert FakeModel.last_instance.generate_kwargs["do_sample"] is False
    assert FakeModel.last_instance.generate_kwargs["max_new_tokens"] == 5
