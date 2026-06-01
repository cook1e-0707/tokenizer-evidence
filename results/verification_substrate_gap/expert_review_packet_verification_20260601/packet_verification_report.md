# VSG Expert Review Packet Verification - 2026-06-01

Status: `PASS`

Verified packet:

```text
results/verification_substrate_gap/expert_review_packet_20260601
```

Verified zip:

```text
results/verification_substrate_gap/vsg_expert_review_packet_20260601.zip
```

Zip SHA256:

```text
82b4007525b3d213bc4920b6b4bd947a7de002fdcf2d9271cc5543a2c32418e8
```

## Verification Facts

| Check | Value |
| --- | --- |
| Packet verifier | `PASS` |
| Packet total file count | `87` |
| Hashed file count | `86` |
| Manifest status | `PASS_PACKET_ASSEMBLED_ARTIFACT_ONLY_20260601_HARDENING_INCLUDED` |
| Manuscript head | `c10b3f1e73689d63ceb0a4b3b8ea980974df16c1` |
| Root head at packet build | `54772ac3712192d9ba9cf6729cc616950322b7d8` |
| Claim-scope lint | `PASS` |
| Claim-scope lint violations | `0` |
| LaTeX log scan | `PASS` |
| Overfull hbox warnings | `0` |

## Included Hardening Summary

| Area | Verified value |
| --- | --- |
| Stronger public-text predicate local pilot | `codeword_recovered_blocks_total = 0` |
| Attack naturalness proxy audit | `proxy_pass_rows = 0/60`; semantic naturalness not claimed |
| Reproducibility release inventory | `78` rows, `0` missing files, release-ready without review `false` |
| Ownership decision-rule audit | `63` rows, `0` failures, supported public final-text rows `0` |
| Manuscript figure-quality audit | `5` figures, `0` failed figure checks, `0` failed data checks |

## Commands

```bash
python3 scripts/verification_substrate_gap/verify_vsg_expert_review_packet_20260601.py
unzip -t results/verification_substrate_gap/vsg_expert_review_packet_20260601.zip
shasum -a 256 results/verification_substrate_gap/vsg_expert_review_packet_20260601.zip
cat results/verification_substrate_gap/vsg_expert_review_packet_20260601.zip.sha256
```
