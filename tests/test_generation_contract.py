import json
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

from src.core.bucket_mapping import BucketLayout, FieldBucketSpec, save_bucket_layout
from src.core.canonical_contract import (
    CanonicalContractError,
    build_canonical_evidence_bundle,
    ensure_train_eval_config_alignment,
    teacher_forced_sanity_check,
)
from src.core.contextual_alignment import audit_contextual_field_values
from src.core.scaffolded_completion import (
    FOUNDATION_ARTIFACT_FORMAT,
    FOUNDATION_FIELDWISE_PROMPT_CONTRACT,
    build_fieldwise_generation_plan,
    build_scaffolded_completion_target,
    evaluate_foundation_completion,
    parse_scaffolded_completion,
    render_foundation_slot_values,
)
from src.infrastructure.config import load_experiment_config
from src.infrastructure.paths import discover_repo_root
from src.training.dataset import TrainingExample
from src.training.hf_causal_lm import run_minimal_hf_causal_lm_training
from src.evaluation.report import EvalRunSummary, load_result_json


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


def _load_eval_script_module() -> object:
    repo_root = discover_repo_root(Path(__file__).parent)
    script_path = repo_root / "scripts" / "eval.py"
    spec = importlib.util.spec_from_file_location("test_eval_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_repo_batch28_qwen_train_and_eval_configs_share_canonical_contract() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    bridge_train_config = load_experiment_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_train__qwen2_5_3b__v1.yaml"
    )
    bridge_eval_config = load_experiment_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_eval__qwen2_5_3b__v1.yaml"
    )
    main_train_config = load_experiment_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_train__qwen2_5_7b__v1.yaml"
    )
    main_eval_config = load_experiment_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_eval__qwen2_5_7b__v1.yaml"
    )

    ensure_train_eval_config_alignment(bridge_train_config, bridge_eval_config, repo_root)
    ensure_train_eval_config_alignment(main_train_config, main_eval_config, repo_root)


def test_repo_batch28_model_roles_remain_split_between_bridge_and_repair() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    bridge_train_config = load_experiment_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_train__qwen2_5_3b__v1.yaml"
    )
    main_train_config = load_experiment_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_train__qwen2_5_7b__v1.yaml"
    )

    assert bridge_train_config.train.target_mode == "scaffolded_canonical_completion"
    assert main_train_config.train.target_mode == "fieldwise_constrained_slot_completion"
    assert main_train_config.train.probe_payload_texts


def test_repo_qwen7b_foundation_config_uses_small_contextual_probe_stage() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    foundation_train_config = load_experiment_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_train__qwen2_5_7b__foundation_v1.yaml"
    )

    assert foundation_train_config.train.target_mode == "foundation_fieldwise_constrained_slot_completion"
    assert foundation_train_config.train.probe_block_count == 1
    assert len(foundation_train_config.train.probe_payload_texts) == 16
    assert foundation_train_config.data.carrier_catalog_path.endswith(
        "real_pilot_catalog__qwen2_5_7b_foundation__v1.yaml"
    )


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


def test_scaffolded_completion_reconstructs_canonical_evidence(tmp_path: Path) -> None:
    catalog_path = _write_frozen_catalog(tmp_path / "catalog.yaml")
    config = load_experiment_config(
        _write_experiment_config(tmp_path / "train.yaml", catalog_path=catalog_path, experiment_name="exp_train")
    )
    bundle = build_canonical_evidence_bundle(config, tmp_path)
    scaffold = build_scaffolded_completion_target(
        bundle,
        instruction="Output exactly one carrier value per line for each slot and nothing else.",
    )

    parsed = parse_scaffolded_completion(
        "\n".join(scaffold.expected_slot_values),
        layout=BucketLayout.from_dict(yaml.safe_load(catalog_path.read_text(encoding="utf-8"))),
        slot_field_names=scaffold.slot_field_names,
        expected_slot_values=scaffold.expected_slot_values,
    )

    assert parsed.valid_canonical_block_count == bundle.contract.block_count
    assert parsed.value_slot_exact_rate == 1.0
    assert parsed.parse_success_rate == 1.0
    assert parsed.first_divergence_position is None
    assert parsed.reconstructed_text == bundle.rendered.text


def test_eval_accepts_scaffolded_slot_value_artifact(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    catalog_path = _write_frozen_catalog(tmp_path / "catalog.yaml")
    train_config = load_experiment_config(
        _write_experiment_config(tmp_path / "train.yaml", catalog_path=catalog_path, experiment_name="exp_train")
    )
    eval_config_path = _write_experiment_config(
        tmp_path / "eval.yaml",
        catalog_path=catalog_path,
        experiment_name="exp_eval",
    )
    bundle = build_canonical_evidence_bundle(train_config, tmp_path)
    scaffold = build_scaffolded_completion_target(
        bundle,
        instruction="Output exactly one carrier value per line for each slot and nothing else.",
    )

    generated_values_path = tmp_path / "generated_values.txt"
    generated_values_path.write_text("\n".join(scaffold.expected_slot_values), encoding="utf-8")
    eval_input_path = tmp_path / "eval_input.json"
    eval_input_path.write_text(
        json.dumps(
            {
                "schema_name": "train_eval_input",
                "source_train_run_id": "train-run",
                "payload_text": "OK",
                "generated_text_path": str(generated_values_path),
                "generated_artifact_format": "scaffolded_slot_values",
                "expected_slot_values": list(scaffold.expected_slot_values),
                "canonical_contract": bundle.contract.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    eval_payload = yaml.safe_load(eval_config_path.read_text(encoding="utf-8"))
    eval_payload["data"]["eval_path"] = str(eval_input_path)
    eval_config_path.write_text(yaml.safe_dump(eval_payload, sort_keys=False), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/eval.py",
            "--config",
            str(eval_config_path),
            "--force",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    eval_summary_path = sorted((tmp_path / "results").rglob("eval_summary.json"))[0]
    eval_summary = load_result_json(eval_summary_path)
    assert isinstance(eval_summary, EvalRunSummary)
    assert eval_summary.accepted is True
    assert eval_summary.verifier_success is True
    assert eval_summary.diagnostics["generated_artifact_format"] == "scaffolded_slot_values"
    assert eval_summary.diagnostics["value_slot_exact_rate"] == 1.0
    assert eval_summary.diagnostics["decode_success_rate"] == 1.0


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
            if cls.last_instance is None:
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
            if cls.last_instance is None:
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


def test_lora_training_mode_uses_peft_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeLoss:
        def detach(self):
            return self

        def cpu(self):
            return self

        def item(self):
            return 0.321

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
            if cls.last_instance is None:
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
            return f"{self.last_prompt} market\ntravel"

        def save_pretrained(self, path: Path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")

        def encode(self, text: str, add_special_tokens: bool = False):
            return [len(text)]

    class FakeModel:
        last_instance = None

        def __init__(self):
            self.config = types.SimpleNamespace(pad_token_id=None)
            self.generate_kwargs = None

        @classmethod
        def from_pretrained(cls, _name):
            if cls.last_instance is None:
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
            (Path(path) / "adapter_config.json").write_text("{}", encoding="utf-8")

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            return [[10, 11]]

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.device = lambda name: name
    fake_torch.no_grad = lambda: FakeNoGrad()
    fake_torch.optim = types.SimpleNamespace(AdamW=lambda params, lr: FakeOptimizer())

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = FakeModel
    fake_transformers.AutoTokenizer = FakeTokenizer

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def _get_peft_model(model, config):
        model.lora_config = config
        return model

    fake_peft = types.ModuleType("peft")
    fake_peft.LoraConfig = FakeLoraConfig
    fake_peft.TaskType = types.SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
    fake_peft.get_peft_model = _get_peft_model

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    result = run_minimal_hf_causal_lm_training(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        max_length=64,
        dataset=[
            TrainingExample(
                prompt="Output exactly one carrier value per line.",
                target_symbols=(),
                metadata={"completion": "market\ntravel\n\n"},
            )
        ],
        batch_size=1,
        epochs=1,
        learning_rate=1.0e-4,
        run_dir=tmp_path,
        require_cuda=False,
        generation_prompt="Output exactly one carrier value per line.",
        generation_max_new_tokens=6,
        generation_stop_strings=("\n\n",),
        generation_bad_words=("SECTION", "TOPIC"),
        adapter_mode="lora",
        lora_target_modules=("q_proj", "k_proj"),
    )

    assert result.generated_text == "market\ntravel"
    assert FakeModel.last_instance is not None
    assert FakeModel.last_instance.lora_config.kwargs["target_modules"] == ["q_proj", "k_proj"]
    assert FakeModel.last_instance.generate_kwargs["bad_words_ids"] == [[7], [5]]


def test_fieldwise_constrained_single_token_decoding_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeLoss:
        def detach(self):
            return self

        def cpu(self):
            return self

        def item(self):
            return 0.111

        def backward(self):
            return None

    class FakeTensor:
        def __init__(self, payload):
            self.payload = payload

        def to(self, _device):
            return self

        def __getitem__(self, index):
            return self.payload[index]

        def tolist(self):
            return self.payload

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
        value_token_ids = {
            "news": 10,
            "report": 11,
            "guide": 12,
            "update": 13,
            "review": 14,
            "market": 21,
            "travel": 22,
            "health": 23,
            "science": 24,
            "climate": 25,
        }

        def __init__(self):
            self.pad_token = None
            self.eos_token = "<eos>"
            self.pad_token_id = 0
            self.eos_token_id = 99
            self.prompt_to_id: dict[str, int] = {}
            self.prompt_id_to_expected_token: dict[int, int] = {}
            self.id_to_text: dict[int, str] = {}
            self.vocab_size = 128

        @classmethod
        def from_pretrained(cls, _name):
            if cls.last_instance is None:
                cls.last_instance = cls()
            return cls.last_instance

        def _prompt_id(self, prompt: str) -> int:
            if prompt not in self.prompt_to_id:
                self.prompt_to_id[prompt] = 1000 + len(self.prompt_to_id)
                self.id_to_text[self.prompt_to_id[prompt]] = prompt
            return self.prompt_to_id[prompt]

        def encode(self, text: str, add_special_tokens: bool = False):
            for value, token_id in self.value_token_ids.items():
                if text.endswith(value):
                    prompt = text[: -len(value)]
                    if prompt in self.prompt_to_id:
                        return [self.prompt_to_id[prompt], token_id]
            return [self._prompt_id(text)]

        def __call__(self, text, **_kwargs):
            if isinstance(text, list):
                rows = [self.encode(item) for item in text]
                width = max(len(row) for row in rows)
                padded = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
                mask = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
                return {
                    "input_ids": FakeTensor(padded),
                    "attention_mask": FakeTensor(mask),
                }
            encoded = self.encode(text)
            return {
                "input_ids": FakeTensor([encoded]),
                "attention_mask": FakeTensor([[1] * len(encoded)]),
            }

        def decode(self, tokens, skip_special_tokens: bool = True):
            if isinstance(tokens, int):
                tokens = [tokens]
            inverse = {token_id: value for value, token_id in self.value_token_ids.items()}
            return "".join(
                self.id_to_text.get(int(token_id), inverse.get(int(token_id), ""))
                for token_id in tokens
            )

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
            (Path(path) / "adapter_config.json").write_text("{}", encoding="utf-8")

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            input_row = kwargs["input_ids"][0]
            prompt_id = int(input_row[0])
            allowed = kwargs["prefix_allowed_tokens_fn"](0, input_row)
            expected_token = FakeTokenizer.last_instance.prompt_id_to_expected_token[prompt_id]
            assert expected_token in allowed
            return FakeTensor([[prompt_id, expected_token]])

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

    catalog_path = _write_frozen_catalog(tmp_path / "catalog.yaml")
    train_config = load_experiment_config(
        _write_experiment_config(tmp_path / "train.yaml", catalog_path=catalog_path, experiment_name="exp_train")
    )
    bundle = build_canonical_evidence_bundle(train_config, tmp_path)
    plan = build_fieldwise_generation_plan(
        bundle,
        instruction="Output exactly one allowed carrier value for the requested slot.",
        prompt_contract_name=FOUNDATION_FIELDWISE_PROMPT_CONTRACT,
        max_blocks=1,
    )
    tokenizer = FakeTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    for slot_target in plan.slot_targets:
        prompt_id = tokenizer._prompt_id(slot_target.prompt)
        tokenizer.prompt_id_to_expected_token[prompt_id] = tokenizer.value_token_ids[slot_target.expected_value]
    FakeTokenizer.last_instance = tokenizer

    result = run_minimal_hf_causal_lm_training(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        max_length=128,
        dataset=[
            TrainingExample(
                prompt=slot_target.prompt,
                target_symbols=(),
                metadata={"completion": slot_target.expected_value},
            )
            for slot_target in plan.slot_targets
        ],
        batch_size=1,
        epochs=1,
        learning_rate=1.0e-4,
        run_dir=tmp_path,
        require_cuda=False,
        adapter_mode="full",
        fieldwise_generation_plan=plan,
    )

    assert result.generated_text == "\n".join(plan.expected_slot_values)
    assert result.generation_diagnostics["per_slot_exact_rate"] == 1.0
    assert result.generation_diagnostics["parse_success_rate"] == 1.0
    assert result.generation_diagnostics["decode_success_rate"] == 1.0
    assert result.generation_diagnostics["contextual_carrier_audit"]["is_context_safe"] is True
    assert FakeModel.last_instance is not None
    assert FakeModel.last_instance.generate_kwargs["max_new_tokens"] == 1
    assert FakeModel.last_instance.generate_kwargs["do_sample"] is False
    assert "prefix_allowed_tokens_fn" in FakeModel.last_instance.generate_kwargs


def test_contextual_carrier_audit_uses_exact_slot_prefix_not_prefix_stable_diff() -> None:
    class PrefixShiftTokenizer:
        def __init__(self):
            self.id_to_text = {
                1: "SECTION=",
                7: "news",
                8: "report",
            }

        def encode(self, text: str) -> list[int]:
            if text == "SECTION=":
                return [1]
            if text == "SECTION=news":
                return [99]
            if text == "SECTION=report":
                return [98]
            raise AssertionError(text)

        def decode(self, token_ids):
            if tuple(token_ids) == (1, 7):
                return "SECTION=news"
            if tuple(token_ids) == (1, 8):
                return "SECTION=report"
            return "".join(self.id_to_text.get(int(token_id), "") for token_id in token_ids)

    audit = audit_contextual_field_values(
        field_allowed_values={"SECTION": ("news", "report")},
        exact_slot_prefixes={"SECTION": "SECTION="},
        tokenizer=PrefixShiftTokenizer(),
        prompt_contract_name=FOUNDATION_FIELDWISE_PROMPT_CONTRACT,
    )

    assert audit.is_context_safe is True
    assert audit.valid_token_map["SECTION"]["SECTION="]["news"] == 7
    assert audit.valid_token_map["SECTION"]["SECTION="]["report"] == 8


def test_contextual_carrier_audit_fails_when_value_is_not_single_next_token_in_context() -> None:
    class SparseTokenizer:
        def __init__(self):
            self.id_to_text = {1: "SECTION=", 8: "report"}

        def encode(self, text: str) -> list[int]:
            if text == "SECTION=":
                return [1]
            raise AssertionError(text)

        def decode(self, token_ids) -> str:
            if tuple(token_ids) == (1, 8):
                return "SECTION=report"
            return "".join(self.id_to_text.get(int(token_id), "") for token_id in token_ids)

    audit = audit_contextual_field_values(
        field_allowed_values={"SECTION": ("news", "report")},
        exact_slot_prefixes={"SECTION": "SECTION="},
        tokenizer=SparseTokenizer(),
        prompt_contract_name=FOUNDATION_FIELDWISE_PROMPT_CONTRACT,
    )

    assert audit.is_context_safe is False
    assert audit.valid_token_map["SECTION"]["SECTION="] == {"report": 8}


def test_foundation_completion_measures_metrics_and_renders_canonical_block(tmp_path: Path) -> None:
    class PrefixTokenizer:
        def __init__(self):
            self.id_to_text = {1: "SECTION=", 2: "TOPIC=", 7: "news", 21: "market", 22: "travel"}

        def encode(self, text: str) -> list[int]:
            if text == "SECTION=":
                return [1]
            if text == "TOPIC=":
                return [2]
            raise AssertionError(text)

        def decode(self, token_ids) -> str:
            token_tuple = tuple(int(token_id) for token_id in token_ids)
            mapping = {
                (1, 7): "SECTION=news",
                (2, 21): "TOPIC=market",
                (2, 22): "TOPIC=travel",
                (7,): "news",
                (21,): "market",
                (22,): "travel",
            }
            return mapping.get(token_tuple, "".join(self.id_to_text.get(token_id, "") for token_id in token_tuple))

    catalog_path = _write_frozen_catalog(tmp_path / "catalog.yaml")
    layout = BucketLayout.from_dict(yaml.safe_load(catalog_path.read_text(encoding="utf-8")))

    render_text, render_bucket_tuples = render_foundation_slot_values(
        slot_values=("news", "market"),
        layout=layout,
        slot_field_names=("SECTION", "TOPIC"),
    )
    assert render_text == "SECTION=news; TOPIC=market"
    assert render_bucket_tuples == ((0, 0),)

    result = evaluate_foundation_completion(
        "news\ntravel",
        layout=layout,
        expected_slot_values=("news", "market"),
        exact_slot_prefixes={"SECTION": "SECTION=", "TOPIC": "TOPIC="},
        tokenizer=PrefixTokenizer(),
        prompt_contract_name=FOUNDATION_FIELDWISE_PROMPT_CONTRACT,
        slot_field_names=("SECTION", "TOPIC"),
    )

    assert result.field_valid_rate == 1.0
    assert result.bucket_correct_rate == 0.5
    assert result.slot_exact_rate == 0.5
    assert result.per_field_accuracy == {"SECTION": 1.0, "TOPIC": 0.0}
    assert result.rendered_canonical_text == "SECTION=news; TOPIC=travel"
    assert result.foundation_gate_passed is False


def test_foundation_eval_path_reports_f1_metrics_and_passes_for_gold_slot_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    eval_script = _load_eval_script_module()

    class PrefixTokenizer:
        def __init__(self):
            self.id_to_text = {
                1: "SECTION=",
                2: "TOPIC=",
                7: "news",
                8: "report",
                9: "guide",
                10: "update",
                11: "review",
                21: "market",
                22: "travel",
                23: "health",
                24: "science",
                25: "climate",
            }

        def encode(self, text: str) -> list[int]:
            if text == "SECTION=":
                return [1]
            if text == "TOPIC=":
                return [2]
            raise AssertionError(text)

        def decode(self, token_ids) -> str:
            token_tuple = tuple(int(token_id) for token_id in token_ids)
            mapping = {
                (1, 7): "SECTION=news",
                (1, 8): "SECTION=report",
                (1, 9): "SECTION=guide",
                (1, 10): "SECTION=update",
                (1, 11): "SECTION=review",
                (2, 21): "TOPIC=market",
                (2, 22): "TOPIC=travel",
                (2, 23): "TOPIC=health",
                (2, 24): "TOPIC=science",
                (2, 25): "TOPIC=climate",
                (7,): "news",
                (8,): "report",
                (9,): "guide",
                (10,): "update",
                (11,): "review",
                (21,): "market",
                (22,): "travel",
                (23,): "health",
                (24,): "science",
                (25,): "climate",
            }
            return mapping.get(token_tuple, "")

    monkeypatch.setattr(eval_script, "load_tokenizer", lambda *_args, **_kwargs: PrefixTokenizer())

    catalog_path = _write_frozen_catalog(tmp_path / "catalog.yaml")
    eval_config_path = _write_experiment_config(
        tmp_path / "foundation_eval.yaml",
        catalog_path=catalog_path,
        experiment_name="exp_eval",
    )
    payload = yaml.safe_load(eval_config_path.read_text(encoding="utf-8"))
    payload["eval"]["verification_mode"] = "foundation_gate"
    generated_values_path = tmp_path / "generated_values.txt"
    generated_values_path.write_text("news\nmarket", encoding="utf-8")
    eval_input_path = tmp_path / "foundation_eval_input.json"
    eval_input_path.write_text(
        json.dumps(
            {
                "schema_name": "train_eval_input",
                "payload_text": "AA",
                "generated_text_path": str(generated_values_path),
                "generated_artifact_format": FOUNDATION_ARTIFACT_FORMAT,
                "expected_slot_values": ["news", "market"],
                "slot_field_names": ["SECTION", "TOPIC"],
                "exact_slot_prefixes": {"SECTION": "SECTION=", "TOPIC": "TOPIC="},
                "prompt_contract_name": FOUNDATION_FIELDWISE_PROMPT_CONTRACT,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    payload["data"]["eval_path"] = str(eval_input_path)
    eval_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_experiment_config(eval_config_path)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    verification_result, diagnostics = eval_script._run_our_method_eval(config, tmp_path, run_dir)

    assert verification_result.success is True
    assert diagnostics["field_valid_rate"] == 1.0
    assert diagnostics["bucket_correct_rate"] == 1.0
    assert diagnostics["slot_exact_rate"] == 1.0
    assert diagnostics["foundation_gate_passed"] is True
    assert (run_dir / "foundation_rendered_canonical.txt").read_text(encoding="utf-8") == "SECTION=news; TOPIC=market"


def test_promotion_gate_blocks_canonical_eval_when_foundation_summary_failed(tmp_path: Path) -> None:
    eval_script = _load_eval_script_module()

    catalog_path = _write_frozen_catalog(tmp_path / "catalog.yaml")
    eval_config_path = _write_experiment_config(
        tmp_path / "eval.yaml",
        catalog_path=catalog_path,
        experiment_name="exp_eval",
    )
    failed_summary_path = tmp_path / "failed_foundation_eval_summary.json"
    EvalRunSummary(
        run_id="foundation-run",
        experiment_name="exp_eval",
        method_name="our_method",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        seed=17,
        git_commit="abc123",
        timestamp="20260419T000000Z",
        hostname="test-host",
        slurm_job_id=None,
        status="failed",
        dataset_name="real-pilot-foundation",
        sample_count=1,
        accepted=False,
        match_ratio=0.5,
        threshold=0.0,
        verification_mode="foundation_gate",
        render_format="canonical_v1",
        verifier_success=False,
        decoded_payload=None,
        decoded_unit_count=0,
        decoded_block_count=0,
        unresolved_field_count=0,
        malformed_count=0,
        utility_acceptance_rate=0.0,
        notes="foundation failed",
        diagnostics={"foundation_gate_passed": False},
        run_dir=str(tmp_path / "foundation-run"),
    ).save_json(failed_summary_path)

    payload = yaml.safe_load(eval_config_path.read_text(encoding="utf-8"))
    payload["eval"]["require_foundation_gate"] = True
    payload["data"]["foundation_eval_summary_path"] = str(failed_summary_path)
    eval_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_experiment_config(eval_config_path)

    with pytest.raises(ValueError, match="foundation gate did not pass"):
        eval_script._run_our_method_eval(config, tmp_path, tmp_path / "run")


def test_canonical_eval_rejects_foundation_artifact_even_when_f1_summary_passed(tmp_path: Path) -> None:
    eval_script = _load_eval_script_module()

    catalog_path = _write_frozen_catalog(tmp_path / "catalog.yaml")
    eval_config_path = _write_experiment_config(
        tmp_path / "eval.yaml",
        catalog_path=catalog_path,
        experiment_name="exp_eval",
    )
    passing_summary_path = tmp_path / "passing_foundation_eval_summary.json"
    EvalRunSummary(
        run_id="foundation-run",
        experiment_name="exp_eval",
        method_name="our_method",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        seed=17,
        git_commit="abc123",
        timestamp="20260419T000000Z",
        hostname="test-host",
        slurm_job_id=None,
        status="completed",
        dataset_name="real-pilot-foundation",
        sample_count=1,
        accepted=True,
        match_ratio=1.0,
        threshold=0.0,
        verification_mode="foundation_gate",
        render_format="canonical_v1",
        verifier_success=True,
        decoded_payload=None,
        decoded_unit_count=1,
        decoded_block_count=1,
        unresolved_field_count=0,
        malformed_count=0,
        utility_acceptance_rate=1.0,
        notes="foundation passed",
        diagnostics={"foundation_gate_passed": True},
        run_dir=str(tmp_path / "foundation-run"),
    ).save_json(passing_summary_path)

    foundation_eval_input_path = tmp_path / "foundation_eval_input.json"
    foundation_eval_input_path.write_text(
        json.dumps(
            {
                "schema_name": "train_eval_input",
                "payload_text": "AA",
                "generated_text_path": str(tmp_path / "generated_text.txt"),
                "generated_artifact_format": FOUNDATION_ARTIFACT_FORMAT,
                "canonical_contract": {
                    "catalog_path": str(catalog_path),
                    "catalog_sha256": "abc",
                    "catalog_name": "foundation",
                    "field_names": ["SECTION", "TOPIC"],
                    "radices": [4, 4],
                    "render_format": "canonical_v1",
                    "payload_text": "AA",
                    "block_count": 1,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (tmp_path / "generated_text.txt").write_text("report\nmarket", encoding="utf-8")

    payload = yaml.safe_load(eval_config_path.read_text(encoding="utf-8"))
    payload["eval"]["require_foundation_gate"] = True
    payload["data"]["eval_path"] = str(foundation_eval_input_path)
    payload["data"]["foundation_eval_summary_path"] = str(passing_summary_path)
    eval_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_experiment_config(eval_config_path)

    with pytest.raises(ValueError, match="canonical_render eval was given foundation_slot_values"):
        eval_script._run_our_method_eval(config, tmp_path, tmp_path / "run")


def test_foundation_fieldwise_plan_uses_deterministic_prefix_contract(tmp_path: Path) -> None:
    catalog_path = _write_frozen_catalog(tmp_path / "catalog.yaml")
    config = load_experiment_config(
        _write_experiment_config(tmp_path / "train.yaml", catalog_path=catalog_path, experiment_name="exp_train")
    )
    bundle = build_canonical_evidence_bundle(config, tmp_path)

    plan = build_fieldwise_generation_plan(
        bundle,
        instruction="ignored for foundation prompts",
        prompt_contract_name=FOUNDATION_FIELDWISE_PROMPT_CONTRACT,
        max_blocks=1,
    )

    assert plan.artifact_format == FOUNDATION_ARTIFACT_FORMAT
    assert plan.fields_per_block == 2
    assert len(plan.slot_targets) == 2
    assert tuple(target.prompt for target in plan.slot_targets) == ("SECTION=", "TOPIC=")
