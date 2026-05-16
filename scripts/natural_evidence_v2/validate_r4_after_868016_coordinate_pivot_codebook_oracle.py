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
DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_868016_reliability_coordinate_pivot_codebook_oracle_route.yaml"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


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
        raise ValueError(f"{name} must contain only 0/1")
    return bits


def complement(bits: Iterable[int]) -> list[int]:
    return [1 - int(bit) for bit in bits]


def validate_precommit(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    precommit_dir = resolve(config["precommit_dir"])
    codebook_path = precommit_dir / "codebook.json"
    decoder_path = precommit_dir / "decoder_spec.json"
    hashes_path = precommit_dir / "precommit_hashes.json"
    for path in (codebook_path, decoder_path, hashes_path):
        if not path.exists():
            raise FileNotFoundError(path)
    hashes = read_json(hashes_path)
    observed = {
        "codebook_sha256": sha256_file(codebook_path),
        "decoder_spec_sha256": sha256_file(decoder_path),
        "precommit_hashes_sha256": sha256_file(hashes_path),
    }
    for key in ("codebook_sha256", "decoder_spec_sha256"):
        expected = str(config.get(key, ""))
        if observed[key] != expected:
            raise ValueError(f"{key} mismatch: expected {expected}, observed {observed[key]}")
        if str(hashes.get(key, "")) != expected:
            raise ValueError(f"precommit_hashes {key} mismatch")
    return read_json(codebook_path), read_json(decoder_path), hashes


def validate_codebook_contract(codebook: Mapping[str, Any], decoder_spec: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if codebook.get("contract_id") != config.get("contract_id"):
        raise ValueError("contract_id mismatch")
    if int(codebook.get("payload_bits", -1)) != 4 or int(codebook.get("checksum_bits", -1)) != 4:
        raise ValueError("expected 4 payload bits and 4 checksum bits")
    selected = [int(item) for item in codebook.get("selected_coordinates", [])]
    if len(selected) != int(config.get("selected_coordinate_count", -1)):
        raise ValueError("selected coordinate count mismatch")
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate selected coordinates")
    mapping = codebook.get("pair_to_bit_mapping")
    if not isinstance(mapping, list) or len(mapping) != 8:
        raise ValueError("expected exactly 8 pair mappings")
    seen_bits: set[int] = set()
    seen_coordinates: set[int] = set()
    for item in mapping:
        bit_index = int(item["bit_index"])
        coordinates = [int(coordinate) for coordinate in item.get("coordinates", [])]
        if bit_index in seen_bits or bit_index < 0 or bit_index > 7:
            raise ValueError(f"invalid or duplicate bit_index: {bit_index}")
        if not 1 <= len(coordinates) <= 2:
            raise ValueError(f"bit {bit_index} must have one or two filtered coordinates")
        for coordinate in coordinates:
            if coordinate not in selected:
                raise ValueError(f"bit {bit_index} contains unselected coordinate {coordinate}")
            if coordinate in seen_coordinates:
                raise ValueError(f"duplicate coordinate across pairs: {coordinate}")
            seen_coordinates.add(coordinate)
        seen_bits.add(bit_index)
    if seen_bits != set(range(8)):
        raise ValueError(f"incomplete pair mapping: {sorted(seen_bits)}")
    if set(selected) != seen_coordinates:
        raise ValueError("selected coordinates do not match pair mapping coordinates")
    if decoder_spec.get("decoder") != "pair_majority_then_checksum":
        raise ValueError("decoder spec must be pair_majority_then_checksum")
    rules = decoder_spec.get("accept_rules", {})
    if int(rules.get("required_pairs", -1)) != 8 or int(rules.get("min_pair_support", -1)) != 1:
        raise ValueError("decoder accept rules mismatch")
    if decoder_spec.get("no_posthoc_threshold_changes") is not True:
        raise ValueError("decoder spec must forbid posthoc threshold changes")


def observations_from_bits(pair_mapping: list[Mapping[str, Any]], bits: list[int], *, mode: str) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    skipped_singleton = False
    injected_tie = False
    for item in sorted(pair_mapping, key=lambda row: int(row["bit_index"])):
        bit_index = int(item["bit_index"])
        coordinates = [int(coordinate) for coordinate in item["coordinates"]]
        bit = int(bits[bit_index])
        if mode == "perfect":
            active_coordinates = coordinates
        elif mode == "first_only":
            active_coordinates = coordinates[:1]
        elif mode == "missing_singleton_pair":
            if len(coordinates) == 1 and not skipped_singleton:
                active_coordinates = []
                skipped_singleton = True
            else:
                active_coordinates = coordinates
        elif mode == "missing_first_pair" and bit_index == 0:
            active_coordinates = []
        elif mode == "tie_first_two_coordinate_pair" and len(coordinates) == 2 and not injected_tie:
            observations.append({"coordinate_id": coordinates[0], "bit": bit, "bit_index": bit_index})
            observations.append({"coordinate_id": coordinates[1], "bit": 1 - bit, "bit_index": bit_index})
            injected_tie = True
            continue
        else:
            active_coordinates = coordinates
        for coordinate in active_coordinates:
            observations.append({"coordinate_id": coordinate, "bit": bit, "bit_index": bit_index})
    return observations


def decode(observations: list[Mapping[str, Any]], pair_mapping: list[Mapping[str, Any]], expected_payload_bits: list[int]) -> dict[str, Any]:
    votes_by_coordinate: dict[int, int] = {}
    duplicate_coordinates: list[int] = []
    for observation in observations:
        coordinate = int(observation["coordinate_id"])
        bit = int(observation["bit"])
        if coordinate in votes_by_coordinate:
            duplicate_coordinates.append(coordinate)
        votes_by_coordinate[coordinate] = bit
    decoded_bits: list[int | None] = [None] * 8
    pair_rows: list[dict[str, Any]] = []
    rejected: list[str] = []
    for item in sorted(pair_mapping, key=lambda row: int(row["bit_index"])):
        bit_index = int(item["bit_index"])
        coordinates = [int(coordinate) for coordinate in item["coordinates"]]
        votes = [votes_by_coordinate[coordinate] for coordinate in coordinates if coordinate in votes_by_coordinate]
        counts = Counter(votes)
        if not votes:
            decoded = None
            reason = "missing_pair"
            rejected.append(f"bit_{bit_index}:missing_pair")
        elif counts[0] == counts[1]:
            decoded = None
            reason = "pair_tie"
            rejected.append(f"bit_{bit_index}:pair_tie")
        else:
            decoded = 1 if counts[1] > counts[0] else 0
            decoded_bits[bit_index] = decoded
            reason = ""
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
    checksum_valid = checksum_bits == complement(payload_bits) if complete else False
    payload_matches = payload_bits == expected_payload_bits if complete else False
    if duplicate_coordinates:
        rejected.append("duplicate_coordinate_observations")
    if complete and not checksum_valid:
        rejected.append("checksum_mismatch")
    if complete and not payload_matches:
        rejected.append("payload_commitment_mismatch")
    accept = bool(complete and checksum_valid and payload_matches and not duplicate_coordinates)
    return {
        "accept": accept,
        "decoded_bits": decoded_bits,
        "payload_bits": payload_bits,
        "checksum_bits": checksum_bits,
        "checksum_valid": checksum_valid,
        "payload_matches_commitment": payload_matches,
        "pair_rows": pair_rows,
        "rejected_reasons": rejected,
        "observed_coordinates": len(votes_by_coordinate),
        "complete_pairs": sum(bit is not None for bit in decoded_bits),
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
        raise ValueError("checksum bits must complement payload bits")
    if expected_bits != expected_payload_bits + expected_checksum_bits:
        raise ValueError("expected codeword mismatch")
    cases = [
        ("expected_perfect", expected_bits, "perfect", True),
        ("expected_first_only", expected_bits, "first_only", True),
        ("expected_missing_pair", expected_bits, "missing_first_pair", False),
        ("expected_missing_singleton_pair", expected_bits, "missing_singleton_pair", False),
        ("expected_pair_tie", expected_bits, "tie_first_two_coordinate_pair", False),
        ("wrong_payload_valid_checksum", wrong_payload_bits + wrong_payload_checksum_bits, "perfect", False),
        ("wrong_key_xor_perturbation", [bit ^ mask for bit, mask in zip(expected_bits, wrong_key_xor_mask)], "perfect", False),
    ]
    case_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for case_name, bits, mode, expected_accept in cases:
        observations = observations_from_bits(pair_mapping, bits, mode=mode)
        decoded = decode(observations, pair_mapping, expected_payload_bits)
        case_rows.append(
            {
                "case_name": case_name,
                "expected_accept": expected_accept,
                "observed_accept": bool(decoded["accept"]),
                "case_pass": bool(decoded["accept"]) == expected_accept,
                "decoded_bits": json.dumps(decoded["decoded_bits"]),
                "observed_coordinates": decoded["observed_coordinates"],
                "complete_pairs": decoded["complete_pairs"],
                "min_pair_support": decoded["min_pair_support"],
                "rejected_reasons": ";".join(decoded["rejected_reasons"]),
            }
        )
        for row in decoded["pair_rows"]:
            pair_rows.append({"case_name": case_name, **row, "coordinates": json.dumps(row["coordinates"]), "votes": json.dumps(row["votes"])})
    return case_rows, pair_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the R4 after-868016 coordinate-pivot codebook oracle.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
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
    codebook, decoder_spec, hashes = validate_precommit(config)
    validate_codebook_contract(codebook, decoder_spec, config)
    case_rows, pair_rows = run_oracle_cases(codebook, config)
    failures = [row for row in case_rows if not bool(row["case_pass"])]
    status = (
        "PASS_R4_AFTER_868016_COORDINATE_PIVOT_CODEBOOK_ORACLE_ARTIFACT_ONLY"
        if not failures
        else "FAIL_R4_AFTER_868016_COORDINATE_PIVOT_CODEBOOK_ORACLE_ARTIFACT_ONLY"
    )
    summary = {
        "status": status,
        "schema_name": "r4_after_868016_coordinate_pivot_codebook_oracle_summary_v1",
        "config": str(config_path.relative_to(ROOT)),
        "precommit_dir": str(resolve(config["precommit_dir"]).relative_to(ROOT)),
        "contract_id": config["contract_id"],
        "selected_coordinate_count": int(config["selected_coordinate_count"]),
        "codebook_sha256": config["codebook_sha256"],
        "decoder_spec_sha256": config["decoder_spec_sha256"],
        "precommit_hashes": hashes,
        "oracle_case_count": len(case_rows),
        "oracle_case_failures": failures,
        "wrong_payload_accepts": sum(
            1 for row in case_rows if str(row["case_name"]).startswith("wrong_payload") and bool(row["observed_accept"])
        ),
        "wrong_key_accepts": sum(
            1 for row in case_rows if str(row["case_name"]).startswith("wrong_key") and bool(row["observed_accept"])
        ),
        "expected_perfect_accept": next(row["observed_accept"] for row in case_rows if row["case_name"] == "expected_perfect"),
        "expected_first_only_accept": next(row["observed_accept"] for row in case_rows if row["case_name"] == "expected_first_only"),
        "slurm_submitted": False,
        "tokenizer_validation_started": False,
        "model_scoring_started": False,
        "generation_started": False,
        "training_started": False,
        "llama_started": False,
        "far_started": False,
        "sanitizer_started": False,
        "paper_claim_allowed": False,
    }
    write_json(output_dir / "oracle_summary.json", summary)
    write_csv(output_dir / "oracle_cases.csv", case_rows)
    write_csv(output_dir / "oracle_pair_traces.csv", pair_rows)
    (output_dir / "oracle_report.md").write_text(
        "\n".join(
            [
                "# R4 After-868016 Coordinate-Pivot Codebook Oracle",
                "",
                f"- status: `{status}`",
                f"- selected coordinates: `{config['selected_coordinate_count']}`",
                f"- oracle cases: `{len(case_rows)}`",
                f"- case failures: `{len(failures)}`",
                f"- wrong-key accepts: `{summary['wrong_key_accepts']}`",
                f"- wrong-payload accepts: `{summary['wrong_payload_accepts']}`",
                "",
                "No Slurm submission, tokenizer validation, model scoring, generation, training, Llama, FAR, sanitizer, or paper-facing claim was started.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
