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
from src.core.catalog_freeze import load_required_frozen_catalog
from src.core.contract_compiler import ContractCompilationError, compile_fieldwise_train_contract
from src.core.contextual_alignment import audit_contextual_field_values
from src.core.scaffolded_completion import (
    COMPILED_ARTIFACT_FORMAT,
    COMPILED_FIELDWISE_PROMPT_CONTRACT,
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
from src.training.hf_causal_lm import (
    _compute_compiled_bucket_loss,
    HFCausalLMTrainingError,
    HFCausalLMTrainingResult,
    run_minimal_hf_causal_lm_training,
)
from src.evaluation.report import EvalRunSummary, TrainRunSummary, load_result_json


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


def _write_compiled_minimal_catalog(path: Path) -> Path:
    layout = BucketLayout(
        fields=(
            FieldBucketSpec(
                field_name="SECTION",
                buckets={0: ("news",), 1: ("report",), 2: ("guide",), 3: ("update",)},
            ),
            FieldBucketSpec(
                field_name="TOPIC",
                buckets={0: ("market",), 1: ("travel",), 2: ("health",), 3: ("science",)},
            ),
        ),
        catalog_name="generation-contract-compiled-catalog",
        provenance={
            "catalog_status": "frozen",
            "freeze_status": "strict_passed",
            "tokenizer_name": "qwen-test",
            "tokenizer_backend": "huggingface",
            "tokenizer_revision_source": "qwen-test",
            "source_catalog": str(path.with_name("source.yaml")),
            "freeze_timestamp": "20260419T000000Z",
            "git_commit": "abc123",
        },
    )
    save_bucket_layout(layout, path)
    return path


class CompiledPromptTokenizer:
    def __init__(self, *, invalid_triplets: set[tuple[str, str, str]] | None = None) -> None:
        self.invalid_triplets = set(invalid_triplets or set())
        self.id_to_text = {
            7: "news",
            8: "report",
            9: "guide",
            10: "update",
            21: "market",
            22: "travel",
            23: "health",
            24: "science",
        }
        self.value_to_id = {
            "news": 7,
            "report": 8,
            "guide": 9,
            "update": 10,
            "market": 21,
            "travel": 22,
            "health": 23,
            "science": 24,
        }
        self.prompt_to_id: dict[str, int] = {}
        self.next_prompt_id = 1000

    def encode(self, text: str) -> list[int]:
        if text in self.value_to_id:
            return [self.value_to_id[text]]
        if text not in self.prompt_to_id:
            self.prompt_to_id[text] = self.next_prompt_id
            self.id_to_text[self.next_prompt_id] = text
            self.next_prompt_id += 1
        return [self.prompt_to_id[text]]

    def decode(self, token_ids) -> str:
        token_tuple = tuple(int(token_id) for token_id in token_ids)
        if len(token_tuple) == 1:
            return self.id_to_text.get(token_tuple[0], "")
        if len(token_tuple) == 2:
            prompt_text = self.id_to_text.get(token_tuple[0], "")
            value_text = self.id_to_text.get(token_tuple[1], "")
            payload_label = ""
            field_name = ""
            for line in prompt_text.splitlines():
                if line.startswith("Payload label: "):
                    payload_label = line.split(": ", 1)[1]
                if line.startswith("Field: "):
                    field_name = line.split(": ", 1)[1]
            if (payload_label, field_name, value_text) in self.invalid_triplets:
                return value_text
            return f"{prompt_text}{value_text}"
        return "".join(self.id_to_text.get(token_id, "") for token_id in token_tuple)


def _load_eval_script_module() -> object:
    repo_root = discover_repo_root(Path(__file__).parent)
    script_path = repo_root / "scripts" / "eval.py"
    spec = importlib.util.spec_from_file_location("test_eval_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_train_script_module() -> object:
    repo_root = discover_repo_root(Path(__file__).parent)
    script_path = repo_root / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("test_train_script", script_path)
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
    assert main_train_config.train.target_mode == "compiled_fieldwise_bucket_mass"
    assert main_eval_config.eval.verification_mode == "compiled_gate"
    assert main_train_config.data.carrier_catalog_path == main_eval_config.data.carrier_catalog_path


def test_repo_batch28_model_roles_remain_split_between_bridge_and_repair() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    bridge_train_config = load_experiment_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_train__qwen2_5_3b__v1.yaml"
    )
    main_train_config = load_experiment_config(
        repo_root / "configs" / "experiment" / "frozen" / "exp_train__qwen2_5_7b__v1.yaml"
    )

    assert bridge_train_config.train.target_mode == "scaffolded_canonical_completion"
    assert main_train_config.train.target_mode == "compiled_fieldwise_bucket_mass"
    assert main_train_config.train.probe_block_count == 2
    assert len(main_train_config.train.probe_payload_texts) == 16
    assert main_train_config.train.probe_payload_texts[0] == "U00"
    assert main_train_config.train.probe_payload_texts[-1] == "U15"
    assert main_train_config.eval.payload_text == "U14"
    assert main_train_config.data.carrier_catalog_path.endswith(
        "real_pilot_catalog__qwen2_5_7b_compiled__v1.yaml"
    )


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


def test_repo_qwen7b_compiled_catalog_is_recognized_as_strict_frozen() -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    compiled_catalog_path = (
        repo_root / "configs" / "data" / "frozen" / "real_pilot_catalog__qwen2_5_7b_compiled__v1.yaml"
    )

    layout = load_required_frozen_catalog(compiled_catalog_path)

    assert layout.catalog_name == "real-pilot-catalog-qwen2.5-7b-compiled-v1"
    assert layout.radices == (4, 4)


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

        def encode(self, text: str, add_special_tokens: bool = False):
            if text == "Emit canonical ownership evidence only:":
                return [1]
            return [1, 2, 3]

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
                metadata={
                    "completion": slot_target.expected_value,
                    "slot_type": slot_target.field_name,
                },
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


def test_fieldwise_training_uses_contextual_single_token_target_not_prefix_diff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeLoss:
        def detach(self):
            return self

        def cpu(self):
            return self

        def item(self):
            return 0.25

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

    class PrefixShiftTokenizer:
        last_instance = None

        def __init__(self):
            self.pad_token = None
            self.eos_token = "<eos>"
            self.pad_token_id = 0
            self.eos_token_id = 99
            self.vocab_size = 128
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

        @classmethod
        def from_pretrained(cls, _name):
            if cls.last_instance is None:
                cls.last_instance = cls()
            return cls.last_instance

        def encode(self, text: str, add_special_tokens: bool = False):
            if text == "SECTION=":
                return [1]
            if text == "TOPIC=":
                return [2]
            if text == "SECTION=report":
                return [98]
            if text == "TOPIC=market":
                return [97]
            raise AssertionError(text)

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

        def decode(self, token_ids, skip_special_tokens: bool = True):
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
            return mapping.get(token_tuple, "".join(self.id_to_text.get(token_id, "") for token_id in token_tuple))

        def save_pretrained(self, path: Path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeModel:
        last_instance = None

        def __init__(self):
            self.config = types.SimpleNamespace(pad_token_id=None)
            self.training_input_rows = []

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

        def __call__(self, **kwargs):
            self.training_input_rows.append(kwargs["input_ids"].tolist())
            return types.SimpleNamespace(loss=FakeLoss())

        def save_pretrained(self, path: Path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_config.json").write_text("{}", encoding="utf-8")

        def generate(self, **kwargs):
            input_row = kwargs["input_ids"][0]
            prompt_token = int(input_row[0])
            if prompt_token == 1:
                return FakeTensor([[1, 8]])
            if prompt_token == 2:
                return FakeTensor([[2, 21]])
            raise AssertionError(prompt_token)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.device = lambda name: name
    fake_torch.no_grad = lambda: FakeNoGrad()
    fake_torch.optim = types.SimpleNamespace(AdamW=lambda params, lr: FakeOptimizer())
    fake_torch.long = int
    fake_torch.tensor = lambda payload, dtype=None: FakeTensor(payload)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = FakeModel
    fake_transformers.AutoTokenizer = PrefixShiftTokenizer

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    catalog_path = _write_frozen_catalog(tmp_path / "catalog.yaml")
    train_config = load_experiment_config(
        _write_experiment_config(tmp_path / "train.yaml", catalog_path=catalog_path, experiment_name="exp_train")
    )
    bundle = build_canonical_evidence_bundle(train_config, tmp_path, payload_text="AA")
    plan = build_fieldwise_generation_plan(
        bundle,
        instruction="Output exactly one allowed carrier value for the requested slot.",
        prompt_contract_name=FOUNDATION_FIELDWISE_PROMPT_CONTRACT,
        max_blocks=1,
    )

    result = run_minimal_hf_causal_lm_training(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        max_length=128,
        dataset=[
            TrainingExample(
                prompt=slot_target.prompt,
                target_symbols=(),
                metadata={
                    "completion": slot_target.expected_value,
                    "slot_type": slot_target.field_name,
                },
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
    assert FakeModel.last_instance is not None
    assert FakeModel.last_instance.training_input_rows[0] == [[1, 8]]
    assert FakeModel.last_instance.training_input_rows[1] == [[2, 21]]


def test_hf_training_raises_on_non_finite_loss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeLoss:
        def detach(self):
            return self

        def cpu(self):
            return self

        def item(self):
            return float("nan")

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
        @classmethod
        def from_pretrained(cls, _name):
            return cls()

        def __init__(self):
            self.pad_token = "<pad>"
            self.eos_token = "<eos>"
            self.pad_token_id = 0

        def __call__(self, text, **_kwargs):
            return {
                "input_ids": FakeTensor([[1, 2, 3]]),
                "attention_mask": FakeTensor([[1, 1, 1]]),
            }

        def encode(self, text: str, add_special_tokens: bool = False):
            return [1]

        def save_pretrained(self, path: Path):
            Path(path).mkdir(parents=True, exist_ok=True)

        def decode(self, _tokens, skip_special_tokens: bool = True):
            return "ignored"

    class FakeModel:
        @classmethod
        def from_pretrained(cls, _name):
            return cls()

        def __init__(self):
            self.config = types.SimpleNamespace(pad_token_id=None)

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

        def generate(self, **kwargs):
            return FakeTensor([[1, 2, 3]])

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

    with pytest.raises(HFCausalLMTrainingError, match="Non-finite training loss"):
        run_minimal_hf_causal_lm_training(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            max_length=64,
            dataset=[
                TrainingExample(
                    prompt="Emit canonical ownership evidence only:",
                    target_symbols=(),
                    metadata={"completion": "SECTION=report; TOPIC=market"},
                )
            ],
            batch_size=1,
            epochs=1,
            learning_rate=1.0e-4,
            run_dir=tmp_path,
            require_cuda=False,
        )


def test_compiled_objective_modes_are_distinct_on_multi_member_bucket() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, -0.25, 1.25, -2.0],
            ]
        ],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1]], dtype=torch.long)
    example = TrainingExample(
        prompt="Controlled compiled objective prompt.",
        target_symbols=(),
        metadata={
            "compiled_allowed_token_ids": [2, 3, 4, 5],
            "compiled_bucket_to_token_ids": {
                "0": [2],
                "1": [3],
                "3": [4, 5],
            },
            "compiled_target_bucket_id": 3,
            "compiled_target_token_id": 4,
        },
    )

    bucket_mass_loss, _ = _compute_compiled_bucket_loss(
        torch_module=torch,
        logits=logits,
        attention_mask=attention_mask,
        batch_examples=[example],
        objective_mode="bucket_mass",
    )
    fixed_representative_loss, _ = _compute_compiled_bucket_loss(
        torch_module=torch,
        logits=logits,
        attention_mask=attention_mask,
        batch_examples=[example],
        objective_mode="fixed_representative",
    )
    uniform_bucket_loss, _ = _compute_compiled_bucket_loss(
        torch_module=torch,
        logits=logits,
        attention_mask=attention_mask,
        batch_examples=[example],
        objective_mode="uniform_bucket",
    )

    assert float(bucket_mass_loss.item()) < float(uniform_bucket_loss.item())
    assert float(uniform_bucket_loss.item()) < float(fixed_representative_loss.item())


def test_compiled_objective_modes_accept_string_bucket_keys_for_target_bucket_zero() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 1.1, 0.5],
            ]
        ],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1]], dtype=torch.long)
    example = TrainingExample(
        prompt="String-key compiled objective prompt.",
        target_symbols=(),
        metadata={
            "compiled_allowed_token_ids": [2, 3, 4],
            "compiled_bucket_to_token_ids": {
                "0": [2, 3],
                "1": [4],
            },
            "compiled_target_bucket_id": 0,
            "compiled_target_token_id": 2,
        },
    )

    fixed_representative_loss, _ = _compute_compiled_bucket_loss(
        torch_module=torch,
        logits=logits,
        attention_mask=attention_mask,
        batch_examples=[example],
        objective_mode="fixed_representative",
    )
    uniform_bucket_loss, _ = _compute_compiled_bucket_loss(
        torch_module=torch,
        logits=logits,
        attention_mask=attention_mask,
        batch_examples=[example],
        objective_mode="uniform_bucket",
    )

    assert float(fixed_representative_loss.item()) > 0.0
    assert float(uniform_bucket_loss.item()) > 0.0


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


def test_compiled_train_contract_compiles_full_dataset_for_all_payloads(tmp_path: Path) -> None:
    catalog_path = _write_compiled_minimal_catalog(tmp_path / "compiled_catalog.yaml")

    contract = compile_fieldwise_train_contract(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        tokenizer_name="qwen-test",
        tokenizer_backend="huggingface",
        catalog_path=catalog_path,
        payload_labels=("OK", "NO", "UP", "AI"),
        eval_payload_label="OK",
        instruction="Select exactly one allowed carrier token.",
        prompt_contract_name=COMPILED_FIELDWISE_PROMPT_CONTRACT,
        tokenizer=CompiledPromptTokenizer(),
    )

    assert contract.sample_count == 8
    assert contract.block_count == 1
    assert contract.eval_contract.payload_label == "OK"
    assert contract.eval_contract.expected_slot_values == ("news", "market")
    assert {sample.payload_label for sample in contract.samples} == {"OK", "NO", "UP", "AI"}


def test_compiled_train_contract_supports_two_block_probe_stage(tmp_path: Path) -> None:
    catalog_path = _write_compiled_minimal_catalog(tmp_path / "compiled_catalog.yaml")

    contract = compile_fieldwise_train_contract(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        tokenizer_name="qwen-test",
        tokenizer_backend="huggingface",
        catalog_path=catalog_path,
        payload_labels=tuple(f"U{index:02d}" for index in range(16)),
        eval_payload_label="U14",
        instruction="Select exactly one allowed carrier token.",
        block_count=2,
        prompt_contract_name=COMPILED_FIELDWISE_PROMPT_CONTRACT,
        tokenizer=CompiledPromptTokenizer(),
    )

    assert contract.sample_count == 64
    assert contract.block_count == 2
    assert contract.eval_contract.payload_label == "U14"
    assert contract.eval_contract.payload_units == (14, 1)
    assert contract.eval_contract.expected_slot_values == ("update", "health", "news", "travel")
    assert len(contract.eval_contract.exact_slot_prefixes) == 4


def test_compiled_train_contract_fails_when_any_dataset_prompt_is_not_contextually_covered(
    tmp_path: Path,
) -> None:
    catalog_path = _write_compiled_minimal_catalog(tmp_path / "compiled_catalog.yaml")

    with pytest.raises(ContractCompilationError, match="Compiled contract is incomplete"):
        compile_fieldwise_train_contract(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            tokenizer_name="qwen-test",
            tokenizer_backend="huggingface",
            catalog_path=catalog_path,
            payload_labels=("OK", "NO"),
            eval_payload_label="OK",
            instruction="Select exactly one allowed carrier token.",
            prompt_contract_name=COMPILED_FIELDWISE_PROMPT_CONTRACT,
            tokenizer=CompiledPromptTokenizer(invalid_triplets={("NO", "SECTION", "news")}),
        )


def test_compiled_train_script_uses_synthesized_dataset_without_train_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_script = _load_train_script_module()
    catalog_path = _write_compiled_minimal_catalog(tmp_path / "compiled_catalog.yaml")
    config_path = _write_experiment_config(
        tmp_path / "compiled_train.yaml",
        catalog_path=catalog_path,
        experiment_name="exp_train",
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["model"]["name"] = "Qwen/Qwen2.5-7B-Instruct"
    payload["model"]["family"] = "huggingface-causal-lm"
    payload["model"]["tokenizer_name"] = "qwen-test"
    payload["model"]["tokenizer_backend"] = "huggingface"
    payload["train"]["target_mode"] = "compiled_fieldwise_bucket_mass"
    payload["train"]["probe_payload_texts"] = ["OK", "NO", "UP", "AI"]
    payload["train"]["generation_prompt"] = "Select exactly one allowed carrier token."
    payload["train"]["generation_max_new_tokens"] = 1
    payload["data"]["train_path"] = ""
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(
        train_script,
        "parse_args",
        lambda: types.SimpleNamespace(
            config=str(config_path),
            override=[],
            force=True,
            jsonl_log=False,
        ),
    )
    monkeypatch.setattr(
        train_script,
        "compile_fieldwise_train_contract",
        lambda **kwargs: compile_fieldwise_train_contract(
            **kwargs,
            tokenizer=CompiledPromptTokenizer(),
        ),
    )

    captured: dict[str, object] = {}

    def _fake_training(**kwargs):
        captured["dataset"] = kwargs["dataset"]
        captured["fieldwise_generation_plan"] = kwargs["fieldwise_generation_plan"]
        captured["use_compiled_bucket_objective"] = kwargs["use_compiled_bucket_objective"]
        captured["compiled_objective_mode"] = kwargs["compiled_objective_mode"]
        return HFCausalLMTrainingResult(
            status="ok",
            steps=8,
            examples_seen=8,
            final_loss=0.0,
            checkpoint_dir=str(tmp_path / "checkpoint"),
            generated_text="news\nmarket",
            generation_diagnostics={},
            health_diagnostics={},
        )

    monkeypatch.setattr(train_script, "run_minimal_hf_causal_lm_training", _fake_training)

    assert train_script.main() == 0
    assert captured["use_compiled_bucket_objective"] is True
    assert captured["compiled_objective_mode"] == "bucket_mass"
    dataset = captured["dataset"]
    assert isinstance(dataset, list)
    assert len(dataset) == 8
    assert all(example.metadata["target_mode"] == "compiled_fieldwise_bucket_mass" for example in dataset)
    assert all(
        example.metadata["completion"] in {"news", "report", "market", "travel", "health", "science"}
        for example in dataset
    )

    train_summary_path = sorted((tmp_path / "results").rglob("train_summary.json"))[0]
    train_summary = load_result_json(train_summary_path)
    assert isinstance(train_summary, TrainRunSummary)
    assert train_summary.dataset_size == 8

    eval_input_path = sorted((tmp_path / "results").rglob("eval_input.json"))[0]
    eval_input = json.loads(eval_input_path.read_text(encoding="utf-8"))
    assert eval_input["generated_artifact_format"] == COMPILED_ARTIFACT_FORMAT
    assert "compiled_eval_contract" in eval_input


def test_compiled_gate_eval_path_reports_contract_metrics_and_decoded_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    eval_script = _load_eval_script_module()
    tokenizer = CompiledPromptTokenizer()
    monkeypatch.setattr(eval_script, "load_tokenizer", lambda *_args, **_kwargs: tokenizer)

    catalog_path = _write_compiled_minimal_catalog(tmp_path / "compiled_catalog.yaml")
    contract = compile_fieldwise_train_contract(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        tokenizer_name="qwen-test",
        tokenizer_backend="huggingface",
        catalog_path=catalog_path,
        payload_labels=("OK", "NO", "UP", "AI"),
        eval_payload_label="OK",
        instruction="Select exactly one allowed carrier token.",
        prompt_contract_name=COMPILED_FIELDWISE_PROMPT_CONTRACT,
        tokenizer=tokenizer,
    )

    eval_config_path = _write_experiment_config(
        tmp_path / "compiled_eval.yaml",
        catalog_path=catalog_path,
        experiment_name="exp_eval",
    )
    payload = yaml.safe_load(eval_config_path.read_text(encoding="utf-8"))
    payload["eval"]["verification_mode"] = "compiled_gate"
    generated_values_path = tmp_path / "generated_values.txt"
    generated_values_path.write_text("news\nmarket", encoding="utf-8")
    eval_input_path = tmp_path / "compiled_eval_input.json"
    eval_input_path.write_text(
        json.dumps(
            {
                "schema_name": "train_eval_input",
                "payload_text": "OK",
                "generated_text_path": str(generated_values_path),
                "generated_artifact_format": COMPILED_ARTIFACT_FORMAT,
                "compiled_train_contract_hash": contract.contract_hash,
                "compiled_eval_contract": contract.eval_contract.to_dict(),
                "expected_slot_values": ["news", "market"],
                "slot_field_names": ["SECTION", "TOPIC"],
                "exact_slot_prefixes": contract.eval_contract.exact_slot_prefixes,
                "prompt_contract_name": COMPILED_FIELDWISE_PROMPT_CONTRACT,
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
    assert verification_result.decoded_payload == "OK"
    assert diagnostics["compiled_gate_passed"] is True
    assert diagnostics["field_valid_rate"] == 1.0
    assert diagnostics["bucket_correct_rate"] == 1.0
    assert diagnostics["slot_exact_rate"] == 1.0
    assert (run_dir / "compiled_rendered_canonical.txt").read_text(encoding="utf-8") == "SECTION=news; TOPIC=market"


def test_compiled_gate_eval_path_supports_two_block_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    eval_script = _load_eval_script_module()
    tokenizer = CompiledPromptTokenizer()
    monkeypatch.setattr(eval_script, "load_tokenizer", lambda *_args, **_kwargs: tokenizer)

    catalog_path = _write_compiled_minimal_catalog(tmp_path / "compiled_catalog.yaml")
    contract = compile_fieldwise_train_contract(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        tokenizer_name="qwen-test",
        tokenizer_backend="huggingface",
        catalog_path=catalog_path,
        payload_labels=tuple(f"U{index:02d}" for index in range(16)),
        eval_payload_label="U14",
        instruction="Select exactly one allowed carrier token.",
        block_count=2,
        prompt_contract_name=COMPILED_FIELDWISE_PROMPT_CONTRACT,
        tokenizer=tokenizer,
    )

    eval_config_path = _write_experiment_config(
        tmp_path / "compiled_eval_c3.yaml",
        catalog_path=catalog_path,
        experiment_name="exp_eval",
    )
    payload = yaml.safe_load(eval_config_path.read_text(encoding="utf-8"))
    payload["eval"]["verification_mode"] = "compiled_gate"
    payload["eval"]["payload_text"] = "U14"
    generated_values_path = tmp_path / "generated_values_c3.txt"
    generated_values_path.write_text("update\nhealth\nnews\ntravel", encoding="utf-8")
    eval_input_path = tmp_path / "compiled_eval_input_c3.json"
    eval_input_path.write_text(
        json.dumps(
            {
                "schema_name": "train_eval_input",
                "payload_text": "U14",
                "generated_text_path": str(generated_values_path),
                "generated_artifact_format": COMPILED_ARTIFACT_FORMAT,
                "compiled_train_contract_hash": contract.contract_hash,
                "compiled_eval_contract": contract.eval_contract.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    payload["data"]["eval_path"] = str(eval_input_path)
    eval_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_experiment_config(eval_config_path)

    run_dir = tmp_path / "run_c3"
    run_dir.mkdir()
    verification_result, diagnostics = eval_script._run_our_method_eval(config, tmp_path, run_dir)

    assert verification_result.success is True
    assert verification_result.decoded_payload == "U14"
    assert len(verification_result.decoded_bucket_tuples) == 2
    assert diagnostics["compiled_gate_passed"] is True
    assert diagnostics["valid_canonical_block_count"] == 2
    assert diagnostics["bucket_correct_rate"] == 1.0
    assert (run_dir / "compiled_rendered_canonical.txt").read_text(encoding="utf-8") == (
        "SECTION=update; TOPIC=health\nSECTION=news; TOPIC=travel"
    )
