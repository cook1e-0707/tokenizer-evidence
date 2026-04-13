from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import yaml

from src.core.bucket_mapping import BucketLayout
from src.core.payload_codec import BucketPayloadCodec
from src.core.synthetic_examples import build_synthetic_smoke_example
from src.core.verifier import VerificationConfig, run_synthetic_smoke_verification, verify_structured_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synthetic end-to-end parser/verifier smoke test.")
    parser.add_argument("--spec", help="Optional YAML file with 'layout', 'text', and optional expected payload.")
    parser.add_argument("--output-json", help="Optional output path for a JSON verification report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.spec:
        payload = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit("spec file must contain a mapping")
        layout = BucketLayout.from_dict(payload["layout"])
        codec = BucketPayloadCodec(bucket_radices=layout.radices)
        expected_payload = payload.get("expected_payload_bytes")
        if isinstance(expected_payload, str):
            expected_payload_value = expected_payload.encode("utf-8")
        else:
            expected_payload_value = None
        result = verify_structured_text(
            text=str(payload["text"]),
            bucket_layout=layout,
            payload_codec=codec,
            expected_payload=expected_payload_value,
            config=VerificationConfig(require_all_fields=True, decode_as_bytes=True),
        )
    else:
        example = build_synthetic_smoke_example()
        result = run_synthetic_smoke_verification()
        print(f"synthetic_payload={example.payload!r}")

    print(f"success={result.success}")
    print(f"decoded_units={list(result.decoded_units)}")
    print(f"decoded_payload={result.decoded_payload}")
    print(f"unresolved_fields={list(result.unresolved_fields)}")
    print(f"bucket_mismatches={list(result.bucket_mismatches)}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save_json(output_path)
        print(f"saved_report={output_path}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
