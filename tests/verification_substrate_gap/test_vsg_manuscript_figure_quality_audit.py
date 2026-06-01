from pathlib import Path
import struct
import zlib

from scripts.verification_substrate_gap import audit_vsg_manuscript_figure_quality as audit


def _write_test_png(path: Path, *, blank: bool) -> None:
    width, height = 16, 12
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            if blank:
                row.extend([255, 255, 255, 255])
            elif x == y or x == width - y - 1:
                row.extend([0, 0, 0, 255])
            else:
                row.extend([255, 255, 255, 255])
        rows.append(b"\x00" + bytes(row))
    payload = zlib.compress(b"".join(rows))

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", payload) + chunk(b"IEND", b""))


def test_image_stats_detect_blank_and_nonblank_figures(tmp_path: Path) -> None:
    blank = tmp_path / "blank.png"
    nonblank = tmp_path / "nonblank.png"
    _write_test_png(blank, blank=True)
    _write_test_png(nonblank, blank=False)

    blank_stats = audit.image_stats(blank)
    nonblank_stats = audit.image_stats(nonblank)

    assert blank_stats["nonwhite_ratio"] < 0.01
    assert nonblank_stats["nonwhite_ratio"] > blank_stats["nonwhite_ratio"]
    assert nonblank_stats["unique_sampled_colors"] >= blank_stats["unique_sampled_colors"]


def test_current_manuscript_figure_quality_audit_passes() -> None:
    summary = audit.build(audit.DEFAULT_OUTPUT_DIR)

    assert summary["status"] == "PASS"
    assert summary["figure_count"] == 5
    assert summary["failed_figure_count"] == 0
    assert summary["data_check_count"] == 5
    assert summary["failed_data_check_count"] == 0
    assert summary["new_slurm_started"] is False
    assert summary["public_text_only_verification_claimed"] is False
    assert summary["ownership_proof_claimed"] is False


def test_data_traceability_checks_cover_core_vsg_numbers() -> None:
    rows = audit.data_checks()
    by_name = {row["check"]: row for row in rows}

    assert by_name["figure_3_trace_bound_counts"]["status"] == "PASS"
    assert by_name["figure_3_public_codeword_zero"]["status"] == "PASS"
    assert by_name["figure_4_guided_attack_top100"]["status"] == "PASS"
    assert by_name["figure_5_ownership_matrix_shape"]["status"] == "PASS"
    assert by_name["figure_5_supported_public_text_zero"]["status"] == "PASS"
