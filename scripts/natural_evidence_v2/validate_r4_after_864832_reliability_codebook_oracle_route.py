from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_864832_reliability_codebook_oracle_route.yaml"


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_bit_list(value: Any, *, name: str, length: int) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a list of length {length}")
    bits = [int(item) for item in value]
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError(f"{name} must contain only 0/1 bits")
    return bits


def complement(bits: Iterable[int]) -> list[int]:
    return [1 - int(bit) for bit in bits]


def validate_precommit(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    precommit_dir = resolve(config["precommit_dir"])
    codebook_path = precommit_dir / "codebook.json"
    decoder_path = precommit_dir / "decoder_spec.json"
    manifest_path = precommit_dir / "precommit_manifest.json"
    for path in (codebook_path, decoder_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    observed_hashes = {
        "codebook_sha256": sha256_file(codebook_path),
        "decoder_spec_sha256": sha256_file(decoder_path),
        "precommit_manifest_file_sha256": sha256_file(manifest_path),
    }
    for key, observed in observed_hashes.items():
        expected = str(config.get(key, ""))
        if observed != expected:
            raise ValueError(f"{key} mismatch: expected {expected}, observed {observed}")

    codebook = read_json(codebook_path)
    decoder_spec = read_json(decoder_path)
    manifest = read_json(manifest_path)
    for key, observed in observed_hashes.items():
        if key == "precommit_manifest_file_sha256":
            continue
        manifest_value = str(manifest.get(key, ""))
        if manifest_value and manifest_value != observed:
            raise ValueError(f"manifest {key} mismatch: expected {observed}, manifest has {manifest_value}")
    declared_manifest_hash = str(config.get("precommit_manifest_declared_sha256", ""))
    manifest_declared_value = str(manifest.get("precommit_manifest_sha256", ""))
    if declared_manifest_hash and manifest_declared_value != declared_manifest_hash:
        raise ValueError(
            "manifest declared self hash mismatch: "
            f"expected {declared_manifest_hash}, manifest has {manifest_declared_value}"
        )
    return codebook, decoder_spec, manifest


def validate_codebook_contract(codebook: Mapping[str, Any], decoder_spec: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if codebook.get("contract_id") != config.get("contract_id"):
        raise ValueError("contract_id mismatch between config and codebook")
    if codebook.get("payload_bits") != 4 or codebook.get("checksum_bits") != 4:
        raise ValueError("expected 4 payload bits and 4 checksum bits")
    pair_mapping = codebook.get("pair_to_bit_mapping")
    if not isinstance(pair_mapping, list) or len(pair_mapping) != 8:
        raise ValueError("expected exactly 8 pair-to-bit mappings")
    seen_bits: set[int] = set()
    seen_coordinates: set[int] = set()
    for item in pair_mapping:
        if not isinstance(item, dict):
            raise ValueError("pair mapping entries must be objects")
        bit_index = int(item["bit_index"])
        coordinates = item.get("coordinates")
        if bit_index in seen_bits or bit_index < 0 or bit_index > 7:
            raise ValueError(f"invalid or duplicate bit_index: {bit_index}")
        if not isinstance(coordinates, list) or len(coordinates) != 2:
            raise ValueError(f"bit {bit_index} must have exactly two coordinates")
        for coordinate in coordinates:
            coordinate_int = int(coordinate)
            if coordinate_int in seen_coordinates:
                raise ValueError(f"duplicate selected coordinate: {coordinate_int}")
            seen_coordinates.add(coordinate_int)
        seen_bits.add(bit_index)
    if seen_bits != set(range(8)):
        raise ValueError(f"bit mapping is incomplete: {sorted(seen_bits)}")
    if len(seen_coordinates) != 16:
        raise ValueError("expected 16 unique selected coordinates")

    accept_rules = decoder_spec.get("accept_rules")
    if not isinstance(accept_rules, dict):
        raise ValueError("decoder spec missing accept_rules")
    if int(accept_rules.get("required_pairs", -1)) != 8:
        raise ValueError("decoder required_pairs must be 8")
    if int(accept_rules.get("min_pair_support", -1)) != 1:
        raise ValueError("decoder min_pair_support must be 1")
    if decoder_spec.get("decoder") != "pair_majority_then_checksum":
        raise ValueError("decoder spec must be pair_majority_then_checksum")
    if decoder_spec.get("no_posthoc_threshold_changes") is not True:
        raise ValueError("decoder spec must forbid posthoc threshold changes")


def observations_from_bits(pair_mapping: list[Mapping[str, Any]], bits: list[int], *, mode: str) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for item in sorted(pair_mapping, key=lambda row: int(row["bit_index"])):
        bit_index = int(item["bit_index"])
        coordinates = [int(coordinate) for coordinate in item["coordinates"]]
        bit = int(bits[bit_index])
        if mode == "perfect":
            active_coordinates = coordinates
        elif mode == "single_erasure":
            active_coordinates = coordinates[:1]
        elif mode == "missing_first_pair" and bit_index == 0:
            active_coordinates = []
        elif mode == "tie_first_pair" and bit_index == 0:
            observations.append({"coordinate_id": coordinates[0], "bit": bit, "bit_index": bit_index})
            observations.append({"coordinate_id": coordinates[1], "bit": 1 - bit, "bit_index": bit_index})
            continue
        else:
            active_coordinates = coordinates
        for coordinate in active_coordinates:
            observations.append({"coordinate_id": coordinate, "bit": bit, "bit_index": bit_index})
    return observations


def decode(
    observations: list[Mapping[str, Any]],
    pair_mapping: list[Mapping[str, Any]],
    *,
    expected_payload_bits: list[int],
) -> dict[str, Any]:
    coordinate_to_bit: dict[int, int] = {}
    duplicate_coordinates: list[int] = []
    for observation in observations:
        coordinate = int(observation["coordinate_id"])
        bit = int(observation["bit"])
        if bit not in (0, 1):
            raise ValueError(f"invalid observation bit: {bit}")
        if coordinate in coordinate_to_bit:
            duplicate_coordinates.append(coordinate)
        coordinate_to_bit[coordinate] = bit

    decoded_bits: list[int | None] = [None] * 8
    pair_rows: list[dict[str, Any]] = []
    rejected_reasons: list[str] = []
    for item in sorted(pair_mapping, key=lambda row: int(row["bit_index"])):
        bit_index = int(item["bit_index"])
        coordinates = [int(coordinate) for coordinate in item["coordinates"]]
        votes = [coordinate_to_bit[coordinate] for coordinate in coordinates if coordinate in coordinate_to_bit]
        counts = Counter(votes)
        if not votes:
            reason = "missing_pair"
            rejected_reasons.append(f"bit_{bit_index}:{reason}")
            decoded = None
        elif counts[0] == counts[1]:
            reason = "pair_tie"
            rejected_reasons.append(f"bit_{bit_index}:{reason}")
            decoded = None
        else:
            reason = ""
            decoded = 1 if counts[1] > counts[0] else 0
            decoded_bits[bit_index] = decoded
        pair_rows.append(
            {
                "bit_index": bit_index,
                "coordinates": coordinates,
                "votes": votes,
                "support": len(votes),
                "decoded_bit": decoded,
                "failure_reason": reason,
            }
        )

    complete = all(bit is not None for bit in decoded_bits)
    payload_bits = [int(bit) for bit in decoded_bits[:4]] if complete else []
    checksum_bits = [int(bit) for bit in decoded_bits[4:]] if complete else []
    checksum_expected = complement(payload_bits) if complete else []
    checksum_valid = checksum_bits == checksum_expected if complete else False
    payload_matches_commitment = payload_bits == expected_payload_bits if complete else False
    if duplicate_coordinates:
        rejected_reasons.append("duplicate_coordinate_observations")
    if complete and not checksum_valid:
        rejected_reasons.append("checksum_mismatch")
    if complete and not payload_matches_commitment:
        rejected_reasons.append("payload_commitment_mismatch")
    accept = bool(complete and checksum_valid and payload_matches_commitment and not duplicate_coordinates)
    return {
        "accept": accept,
        "decoded_bits": decoded_bits,
        "payload_bits": payload_bits,
        "checksum_bits": checksum_bits,
        "checksum_expected": checksum_expected,
        "checksum_valid": checksum_valid,
        "payload_matches_commitment": payload_matches_commitment,
        "pair_rows": pair_rows,
        "rejected_reasons": rejected_reasons,
        "observed_coordinates": len(coordinate_to_bit),
        "complete_pairs": sum(1 for bit in decoded_bits if bit is not None),
        "min_pair_support": min((row["support"] for row in pair_rows), default=0),
    }


def run_oracle_cases(codebook: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_mapping = codebook["pair_to_bit_mapping"]
    expected_payload_bits = as_bit_list(config["payload_bits"], name="payload_bits", length=4)
    expected_checksum_bits = as_bit_list(config["expected_checksum_bits"], name="expected_checksum_bits", length=4)
    expected_bits = as_bit_list(config["expected_codeword_bits"], name="expected_codeword_bits", length=8)
    wrong_payload_bits = as_bit_list(config["wrong_payload_bits"], name="wrong_payload_bits", length=4)
    wrong_payload_checksum_bits = as_bit_list(config["wrong_payload_checksum_bits"], name="wrong_payload_checksum_bits", length=4)
    wrong_key_xor_mask = as_bit_list(config["wrong_key_xor_mask"], name="wrong_key_xor_mask", length=8)
    if expected_checksum_bits != complement(expected_payload_bits):
        raise ValueError("expected checksum bits must complement payload bits")
    if expected_bits != expected_payload_bits + expected_checksum_bits:
        raise ValueError("expected codeword bits must be payload_bits + expected_checksum_bits")
    if wrong_payload_checksum_bits != complement(wrong_payload_bits):
        raise ValueError("wrong payload checksum bits must complement wrong payload bits")
    if wrong_payload_bits == expected_payload_bits:
        raise ValueError("wrong_payload_bits must differ from payload_bits")
    if wrong_key_xor_mask == [0] * 8:
        raise ValueError("wrong_key_xor_mask must perturb at least one bit")

    cases: list[tuple[str, list[int], str, bool]] = [
        ("expected_perfect", expected_bits, "perfect", True),
        ("expected_single_coordinate_erasure", expected_bits, "single_erasure", True),
        ("expected_missing_pair", expected_bits, "missing_first_pair", False),
        ("expected_pair_tie", expected_bits, "tie_first_pair", False),
        ("wrong_payload_valid_checksum", wrong_payload_bits + wrong_payload_checksum_bits, "perfect", False),
        ("wrong_payload_with_expected_checksum", wrong_payload_bits + expected_checksum_bits, "perfect", False),
        ("wrong_key_xor_perturbation", [bit ^ mask for bit, mask in zip(expected_bits, wrong_key_xor_mask)], "perfect", False),
    ]

    case_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for case_name, bits, mode, expected_accept in cases:
        observations = observations_from_bits(pair_mapping, bits, mode=mode)
        decoded = decode(observations, pair_mapping, expected_payload_bits=expected_payload_bits)
        case_rows.append(
            {
                "case_name": case_name,
                "expected_accept": expected_accept,
                "observed_accept": bool(decoded["accept"]),
                "case_pass": bool(decoded["accept"]) == expected_accept,
                "decoded_bits": json.dumps(decoded["decoded_bits"]),
                "payload_bits": json.dumps(decoded["payload_bits"]),
                "checksum_bits": json.dumps(decoded["checksum_bits"]),
                "checksum_valid": decoded["checksum_valid"],
                "payload_matches_commitment": decoded["payload_matches_commitment"],
                "observed_coordinates": decoded["observed_coordinates"],
                "complete_pairs": decoded["complete_pairs"],
                "min_pair_support": decoded["min_pair_support"],
                "rejected_reasons": ";".join(decoded["rejected_reasons"]),
            }
        )
        for row in decoded["pair_rows"]:
            pair_rows.append(
                {
                    "case_name": case_name,
                    "bit_index": row["bit_index"],
                    "coordinates": json.dumps(row["coordinates"]),
                    "votes": json.dumps(row["votes"]),
                    "support": row["support"],
                    "decoded_bit": row["decoded_bit"],
                    "failure_reason": row["failure_reason"],
                }
            )
    return case_rows, pair_rows


def validate_no_compute(config: Mapping[str, Any]) -> list[str]:
    no_compute = config.get("no_compute")
    if not isinstance(no_compute, dict):
        raise ValueError("config missing no_compute object")
    failures = []
    for key, value in sorted(no_compute.items()):
        if bool(value):
            failures.append(key)
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Artifact-only oracle validation for the R4 after-864832 reliability-weighted codebook."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Remove an existing output dir before writing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    config = read_yaml(config_path)
    output_dir = resolve(args.output_dir or config["output_dir"])
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    codebook, decoder_spec, manifest = validate_precommit(config)
    validate_codebook_contract(codebook, decoder_spec, config)
    no_compute_failures = validate_no_compute(config)
    case_rows, pair_rows = run_oracle_cases(codebook, config)
    case_failures = [row for row in case_rows if not bool(row["case_pass"])]
    status = (
        "PASS_R4_RELIABILITY_CODEBOOK_DECODER_ORACLE_ARTIFACT_ONLY"
        if not case_failures and not no_compute_failures
        else "FAIL_R4_RELIABILITY_CODEBOOK_DECODER_ORACLE_ARTIFACT_ONLY"
    )
    summary = {
        "status": status,
        "schema_name": "r4_after_864832_reliability_codebook_decoder_oracle_summary_v1",
        "config": str(config_path.relative_to(ROOT)),
        "precommit_dir": str(resolve(config["precommit_dir"]).relative_to(ROOT)),
        "contract_id": config["contract_id"],
        "codebook_sha256": config["codebook_sha256"],
        "decoder_spec_sha256": config["decoder_spec_sha256"],
        "precommit_manifest_declared_sha256": config["precommit_manifest_declared_sha256"],
        "precommit_manifest_file_sha256": config["precommit_manifest_file_sha256"],
        "payload_bits": config["payload_bits"],
        "expected_checksum_bits": config["expected_checksum_bits"],
        "expected_codeword_bits": config["expected_codeword_bits"],
        "oracle_case_count": len(case_rows),
        "oracle_case_failures": case_failures,
        "wrong_payload_accepts": sum(
            1 for row in case_rows if str(row["case_name"]).startswith("wrong_payload") and bool(row["observed_accept"])
        ),
        "wrong_key_accepts": sum(
            1 for row in case_rows if str(row["case_name"]).startswith("wrong_key") and bool(row["observed_accept"])
        ),
        "expected_perfect_accept": next(row["observed_accept"] for row in case_rows if row["case_name"] == "expected_perfect"),
        "expected_single_coordinate_erasure_accept": next(
            row["observed_accept"] for row in case_rows if row["case_name"] == "expected_single_coordinate_erasure"
        ),
        "no_compute_failures": no_compute_failures,
        "slurm_submitted": False,
        "tokenizer_validation_started": False,
        "model_scoring_started": False,
        "generation_started": False,
        "training_started": False,
        "llama_started": False,
        "far_started": False,
        "sanitizer_started": False,
        "paper_claim_allowed": False,
        "manifest_status": manifest.get("status"),
        "codebook_status": codebook.get("status"),
        "decoder_status": decoder_spec.get("status"),
    }
    write_json(output_dir / "oracle_summary.json", summary)
    write_csv(output_dir / "oracle_cases.csv", case_rows)
    write_csv(output_dir / "oracle_pair_traces.csv", pair_rows)
    report = [
        "# R4 Reliability-Weighted Codebook Decoder Oracle",
        "",
        "Artifact-only validation for the frozen after-864832 reliability-weighted codebook.",
        "",
        f"- status: `{status}`",
        f"- contract: `{config['contract_id']}`",
        f"- oracle cases: `{len(case_rows)}`",
        f"- case failures: `{len(case_failures)}`",
        f"- wrong-payload accepts: `{summary['wrong_payload_accepts']}`",
        f"- wrong-key accepts: `{summary['wrong_key_accepts']}`",
        f"- perfect expected accept: `{summary['expected_perfect_accept']}`",
        f"- single-coordinate erasure accept: `{summary['expected_single_coordinate_erasure_accept']}`",
        "",
        "No Slurm submission, tokenizer validation, model scoring, generation, training, Llama, FAR, sanitizer, or paper-facing claim was started.",
        "",
        "## Cases",
        "",
    ]
    for row in case_rows:
        report.append(
            f"- `{row['case_name']}`: expected `{row['expected_accept']}`, observed `{row['observed_accept']}`, pass `{row['case_pass']}`"
        )
    (output_dir / "oracle_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
