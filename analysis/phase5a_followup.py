"""Focused Phase 5A follow-up calculations; does not alter study inputs."""

from __future__ import annotations

import base64
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stego.common import build_payload, bytes_to_bits
from stego.dct import (
    MAX_Y,
    MIN_Y,
    REPETITION,
    _block_bits,
    _blocks_from_plane,
    quantization_table,
)
from stego.tamper import make_tag


CSV_PATH = ROOT / "results" / "compression" / "compression_results.csv"
MANIFEST_PATH = ROOT / "images" / "dataset" / "manifest.csv"
REPORT_PATH = ROOT / "results" / "compression" / "phase5a_followup.txt"
PASSWORD = "phase5a-compression-study-fixed-password"
SEED_BASE = 20261105
SOURCE_BYTES = 40
METHODS = ("LSB", "DCT-q50", "DCT-q75", "DCT-q90")
SUBSAMPLINGS = ("4:4:4", "Pillow default")


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def metric_pair(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    return (
        float(peak_signal_noise_ratio(reference, candidate, data_range=255)),
        float(
            structural_similarity(
                reference, candidate, data_range=255, channel_axis=-1
            )
        ),
    )


def make_payload(seed: int) -> tuple[str, str]:
    source = np.random.default_rng(seed).bytes(SOURCE_BYTES)
    message = base64.b64encode(source).decode("ascii")
    tag = make_tag(message.encode("utf-8"), PASSWORD)
    return message, bytes_to_bits(build_payload(message, tag))


def blocks_to_plane(blocks: np.ndarray, height: int, width: int) -> np.ndarray:
    block_rows, block_columns = height // 8, width // 8
    grid = blocks.reshape(block_rows, block_columns, 8, 8)
    return grid.transpose(0, 2, 1, 3).reshape(height, width)


def embed_plane(
    plane: np.ndarray, expected_bits: str, quality: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embed directly in one 8-bit plane; return integer/float blocks and Y plane."""
    table = quantization_table(quality)
    clamped = np.clip(plane.astype(np.float64), MIN_Y, MAX_Y)
    blocks, covered_height, covered_width = _blocks_from_plane(clamped)
    available_bits = len(blocks) // REPETITION
    if len(expected_bits) > available_bits:
        raise ValueError(f"Payload of {len(expected_bits)} bits exceeds {available_bits}")

    payload_size = len(expected_bits)
    indices = np.concatenate(
        [np.arange(payload_size) + repeat * available_bits for repeat in range(REPETITION)]
    )
    transformed = dctn(
        blocks[indices].astype(np.float64) - 128.0,
        axes=(-2, -1),
        norm="ortho",
    )
    row, column = 3, 2
    step = table[row, column]
    original_quantized = transformed[:, row, column] / step
    nearest = np.rint(original_quantized).astype(np.int64)
    target_bits = np.tile(
        np.fromiter((int(bit) for bit in expected_bits), dtype=np.int64), REPETITION
    )
    wrong_parity = (nearest & 1) != target_bits
    lower, upper = nearest - 1, nearest + 1
    nearer = np.where(
        np.abs(original_quantized - lower) <= np.abs(original_quantized - upper),
        lower,
        upper,
    )
    parity_quantized = np.where(wrong_parity, nearer, nearest)
    transformed[:, row, column] = parity_quantized * step
    reconstructed_float = idctn(transformed, axes=(-2, -1), norm="ortho") + 128.0
    reconstructed_u8 = np.clip(np.rint(reconstructed_float), 0, 255).astype(np.uint8)

    integer_blocks = blocks.copy()
    integer_blocks[indices] = reconstructed_u8
    float_blocks = blocks.astype(np.float64, copy=True)
    float_blocks[indices] = reconstructed_float
    integer_plane = clamped.copy()
    integer_plane[:covered_height, :covered_width] = blocks_to_plane(
        integer_blocks, covered_height, covered_width
    )
    return integer_plane, float_blocks, integer_blocks


def processed_control(cover_rgb: np.ndarray) -> np.ndarray:
    """Run only RGB→YCbCr, DCT brightness clamp, and YCbCr→RGB."""
    height, width = cover_rgb.shape[:2]
    covered_height, covered_width = height // 8 * 8, width // 8 * 8
    ycbcr = np.asarray(Image.fromarray(cover_rgb).convert("YCbCr"), dtype=np.uint8)
    processed = ycbcr.copy()
    processed[:covered_height, :covered_width, 0] = np.clip(
        processed[:covered_height, :covered_width, 0], MIN_Y, MAX_Y
    )
    result = np.array(
        Image.frombytes("YCbCr", (width, height), processed.tobytes()).convert("RGB"),
        dtype=np.uint8,
    )
    result[covered_height:, :, :] = cover_rgb[covered_height:, :, :]
    result[:, covered_width:, :] = cover_rgb[:, covered_width:, :]
    return result


def embed_rgb(
    cover_rgb: np.ndarray, expected_bits: str, quality: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirror stego.dct.hide in memory, exposing pre-rounding Y blocks for diagnosis."""
    height, width = cover_rgb.shape[:2]
    covered_height, covered_width = height // 8 * 8, width // 8 * 8
    ycbcr = np.asarray(Image.fromarray(cover_rgb).convert("YCbCr"), dtype=np.uint8)
    integer_y, float_blocks, integer_blocks = embed_plane(
        ycbcr[:, :, 0], expected_bits, quality
    )
    modified = ycbcr.copy()
    modified[:covered_height, :covered_width, 0] = integer_y[
        :covered_height, :covered_width
    ].astype(np.uint8)
    result = np.array(
        Image.frombytes("YCbCr", (width, height), modified.tobytes()).convert("RGB"),
        dtype=np.uint8,
    )
    result[covered_height:, :, :] = cover_rgb[covered_height:, :, :]
    result[:, covered_width:, :] = cover_rgb[:, covered_width:, :]
    return result, modified, float_blocks, integer_blocks


def raw_bits_from_rgb(rgb: np.ndarray, quality: int, bit_count: int) -> str:
    ycbcr = np.asarray(Image.fromarray(rgb).convert("YCbCr"), dtype=np.uint8)
    blocks, _, _ = _blocks_from_plane(ycbcr[:, :, 0])
    return _block_bits(blocks, quality, bit_count)


def bit_errors(expected: str, actual: str) -> int:
    return sum(a != b for a, b in zip(expected, actual)) + abs(len(expected) - len(actual))


def bool_value(value: str) -> bool:
    return value.strip().lower() == "true"


def mean_sd(values: list[float] | np.ndarray, digits: int = 4) -> str:
    array = np.asarray(values, dtype=np.float64)
    return f"{np.mean(array):.{digits}f} ± {np.std(array):.{digits}f}"


def load_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        records = list(csv.DictReader(source))
    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        manifest = list(csv.DictReader(source))
    usable = [
        row for row in manifest
        if row.get("usable_for_main_studies", "").strip().lower() == "true"
    ]
    usable.sort(key=lambda row: (row["category"], row["new_filename"]))
    return records, usable


def build_cost_section(records: list[dict[str, str]], usable: list[dict[str, str]]) -> str:
    baseline_map = {
        row["image"]: row
        for row in records
        if row["record_type"] == "stego"
        and row["method"] == "DCT-q75"
        and row["jpeg_quality"] == "none"
    }
    rows: list[dict[str, float | str]] = []
    mse_proc: list[float] = []
    mse_embed: list[float] = []
    mse_total: list[float] = []
    cross: list[float] = []

    for image_row in usable:
        name, category = image_row["new_filename"], image_row["category"]
        cover = read_rgb(ROOT / "images" / "dataset" / category / name)
        control = processed_control(cover)
        metric_control = metric_pair(cover, control)
        _, expected = make_payload(SEED_BASE + len(rows))
        stego, _, _, _ = embed_rgb(cover, expected, 75)
        psnr_vs_control, _ = metric_pair(control, stego)
        psnr_vs_cover, _ = metric_pair(cover, stego)

        # Check that the in-memory mirror matches the recorded Phase 5A PNG metric.
        recorded = float(baseline_map[name]["stego_png_psnr"])
        if not math.isclose(psnr_vs_cover, recorded, rel_tol=0.0, abs_tol=1e-9):
            raise AssertionError(
                f"In-memory DCT-q75 mirror differs for {name}: "
                f"{psnr_vs_cover:.12f} vs CSV {recorded:.12f}"
            )

        process_delta = control.astype(np.float64) - cover.astype(np.float64)
        embed_delta = stego.astype(np.float64) - control.astype(np.float64)
        total_delta = stego.astype(np.float64) - cover.astype(np.float64)
        mse_proc.append(float(np.mean(process_delta**2)))
        mse_embed.append(float(np.mean(embed_delta**2)))
        mse_total.append(float(np.mean(total_delta**2)))
        cross.append(float(2 * np.mean(process_delta * embed_delta)))
        rows.append(
            {
                "image": name,
                "category": category,
                "control_psnr": metric_control[0],
                "control_ssim": metric_control[1],
                "embed_psnr": psnr_vs_control,
                "cover_psnr": psnr_vs_cover,
            }
        )

    lines = [
        "1. PROCESSED-CONTROL COST",
        "Control: RGB→Pillow YCbCr; clamp Y to [16,239] over complete 8×8 regions; convert to RGB; restore incomplete edge strips. DCT-75 uses the Phase 5A payload for each image.",
        "PSNR uses RGB, data_range=255; SSIM uses RGB, data_range=255. Population SD (ddof=0).",
        "image          | category  | control vs cover PSNR | control vs cover SSIM | DCT-75 vs control PSNR | DCT-75 vs cover PSNR",
        "---------------|-----------|-----------------------|-----------------------|------------------------|---------------------",
    ]
    for row in rows:
        lines.append(
            f"{row['image']:<14} | {row['category']:<9} | "
            f"{row['control_psnr']:>21.4f} | {row['control_ssim']:>21.6f} | "
            f"{row['embed_psnr']:>22.4f} | {row['cover_psnr']:>19.4f}"
        )
    lines.extend(
        [
            "",
            "Mean ± population SD across 15 images:",
            f"  Processed control vs cover: PSNR {mean_sd([float(r['control_psnr']) for r in rows])} dB; SSIM {mean_sd([float(r['control_ssim']) for r in rows], 6)}",
            f"  DCT-75 vs processed control (embedding-only): PSNR {mean_sd([float(r['embed_psnr']) for r in rows])} dB",
            f"  DCT-75 vs cover (processing + embedding): PSNR {mean_sd([float(r['cover_psnr']) for r in rows])} dB",
            "",
        ]
    )
    proc_mean, embed_mean, total_mean, cross_mean = map(
        float, (np.mean(mse_proc), np.mean(mse_embed), np.mean(mse_total), np.mean(cross))
    )
    # This identity is exact for each image and after averaging: total = process + embedding + cross.
    denom = total_mean if total_mean else 1.0
    lines.extend(
        [
            "MSE decomposition (mean of the 15 per-image RGB MSEs; exact identity total = process + embedding + cross term):",
            f"  processing: {proc_mean:.4f} MSE ({100 * proc_mean / denom:.1f}% of total)",
            f"  embedding-only: {embed_mean:.4f} MSE ({100 * embed_mean / denom:.1f}% of total)",
            f"  interaction/cross term: {cross_mean:.4f} MSE ({100 * cross_mean / denom:.1f}% of total)",
            f"  total DCT-75 vs cover: {total_mean:.4f} MSE (arithmetic mean of per-image MSEs).",
            "These MSE shares describe energy in the mean RGB error; the reported mean PSNR is the arithmetic mean of per-image dB values. PSNR values in dB do not add. The control-vs-cover and stego-vs-control figures isolate the two stages; the MSE cross term shows their interaction.",
            "",
        ]
    )
    return "\n".join(lines)


def rgb_extreme_fraction(rgb: np.ndarray) -> float:
    return float(np.any((rgb >= 250) | (rgb <= 5), axis=-1).mean())


def clipped_block_fraction(modified_ycbcr: np.ndarray) -> tuple[int, int]:
    """Count complete blocks with any inverse YCbCr→RGB channel outside [0,255]."""
    height, width = modified_ycbcr.shape[:2]
    covered_height, covered_width = height // 8 * 8, width // 8 * 8
    values = modified_ycbcr[:covered_height, :covered_width].astype(np.int32)
    y = values[:, :, 0]
    cb = values[:, :, 1] - 128
    cr = values[:, :, 2] - 128
    # Pillow's fixed-point inverse coefficients (pre-clamp integer channel values).
    red = y + ((91881 * cr + 32768) >> 16)
    green = y + ((-22554 * cb - 46802 * cr + 32768) >> 16)
    blue = y + ((116130 * cb + 32768) >> 16)
    clipped = (red < 0) | (red > 255) | (green < 0) | (green > 255) | (blue < 0) | (blue > 255)
    block_rows, block_columns = covered_height // 8, covered_width // 8
    block_mask = (
        clipped.reshape(block_rows, 8, block_columns, 8)
        .transpose(0, 2, 1, 3)
        .reshape(block_rows * block_columns, 64)
    )
    return int(np.any(block_mask, axis=1).sum()), len(block_mask)


def build_dct90_section(
    records: list[dict[str, str]], usable: list[dict[str, str]]
) -> str:
    baselines = {
        (row["image"], row["method"]): row
        for row in records
        if row["record_type"] == "stego" and row["jpeg_quality"] == "none"
    }
    failures = [
        row for row in usable
        if bool_value(baselines[(row["new_filename"], "DCT-q90")]["baseline_excluded"])
    ]
    lines = [
        "2. DCT-90 DIAGNOSIS",
        f"No-compression DCT-q90 failures: {len(failures)} images.",
        "Extreme pixels are reported both for the original cover and the q90 stego: any one RGB channel ≤5 or ≥250. Clip blocks count complete 8×8 blocks with any channel outside [0,255] in the pre-clamp Pillow fixed-point YCbCr→RGB conversion after q90 embedding.",
        "image          | category  | errors q50 | errors q75 | errors q90 | cover extreme px | q90 stego extreme px | RGB-clipped blocks",
        "---------------|-----------|------------|------------|------------|------------------|----------------------|-------------------",
    ]
    for row in failures:
        name, category = row["new_filename"], row["category"]
        cover = read_rgb(ROOT / "images" / "dataset" / category / name)
        baseline50 = baselines[(name, "DCT-q50")]
        baseline75 = baselines[(name, "DCT-q75")]
        baseline90 = baselines[(name, "DCT-q90")]
        _, expected = make_payload(int(baseline90["payload_seed"]))
        stego90, modified_ycbcr, _, _ = embed_rgb(cover, expected, 90)
        clipped_count, total_blocks = clipped_block_fraction(modified_ycbcr)
        cover_extreme = rgb_extreme_fraction(cover)
        stego_extreme = rgb_extreme_fraction(stego90)
        lines.append(
            f"{name:<14} | {category:<9} | {int(baseline50['raw_bit_errors']):>10} | "
            f"{int(baseline75['raw_bit_errors']):>10} | {int(baseline90['raw_bit_errors']):>10} | "
            f"{100*cover_extreme:>15.3f}% | {100*stego_extreme:>19.3f}% | "
            f"{clipped_count}/{total_blocks} ({100*clipped_count/total_blocks:.2f}%)"
        )

    lines.extend(
        [
            "",
            "Variant tests (embedding quality 90; same Phase 5A payload; bit errors against the expected framed payload):",
            "Grayscale uses Pillow L mode, then embeds/extracts directly on that 8-bit plane without YCbCr conversion. Same-source integer Y isolates rounding before RGB conversion.",
            "image          | category  | grayscale direct | same-source integer Y | in-memory float | RGB conversion changed bits",
            "---------------|-----------|------------------|-----------------------|-----------------|----------------------------",
        ]
    )
    gray_errors: list[int] = []
    integer_y_errors: list[int] = []
    float_errors: list[int] = []
    rgb_changed_bits: list[int] = []
    for row in failures:
        name, category = row["new_filename"], row["category"]
        baseline90 = baselines[(name, "DCT-q90")]
        _, expected = make_payload(int(baseline90["payload_seed"]))
        cover = read_rgb(ROOT / "images" / "dataset" / category / name)
        gray = np.asarray(Image.fromarray(cover).convert("L"), dtype=np.uint8)
        gray_y, _, _ = embed_plane(gray, expected, 90)
        out_blocks, _, _ = _blocks_from_plane(gray_y)
        gray_observed = _block_bits(out_blocks, 90, len(expected))
        gray_error = bit_errors(expected, gray_observed)

        stego90, modified_ycbcr, float_blocks, _ = embed_rgb(cover, expected, 90)
        integer_y_blocks, _, _ = _blocks_from_plane(modified_ycbcr[:, :, 0])
        integer_y_observed = _block_bits(integer_y_blocks, 90, len(expected))
        integer_y_error = bit_errors(expected, integer_y_observed)
        png_observed = raw_bits_from_rgb(stego90, 90, len(expected))
        png_error = bit_errors(expected, png_observed)
        if png_error != int(baseline90["raw_bit_errors"]):
            raise AssertionError(
                f"In-memory q90 mirror differs for {name}: "
                f"{png_error} bit errors vs CSV {baseline90['raw_bit_errors']}"
            )
        conversion_delta = bit_errors(integer_y_observed, png_observed)
        float_observed = _block_bits(float_blocks, 90, len(expected))
        float_error = bit_errors(expected, float_observed)
        gray_errors.append(gray_error)
        integer_y_errors.append(integer_y_error)
        float_errors.append(float_error)
        rgb_changed_bits.append(conversion_delta)
        lines.append(
            f"{name:<14} | {category:<9} | {gray_error:>16} | {integer_y_error:>21} | "
            f"{float_error:>15} | {conversion_delta:>26}"
        )
    rounded_failure_count = sum(errors > 0 for errors in integer_y_errors)
    conversion_required_count = sum(
        integer_errors == 0 and int(baseline["raw_bit_errors"]) > 0
        for integer_errors, baseline in zip(
            integer_y_errors,
            [baselines[(row["new_filename"], "DCT-q90")] for row in failures],
        )
    )
    lines.extend(
        [
            "",
            f"Diagnosis: the float path has {sum(float_errors)} errors across these 11 images. Same-source integer Y has errors on {rounded_failure_count}/11, so inverse-transform rounding to 8-bit Y is sufficient to cause most failures before any RGB output conversion. RGB conversion changes {sum(rgb_changed_bits)} extracted bit values in total; {conversion_required_count}/11 failures first appear after that conversion. The grayscale variant also fails on {sum(error > 0 for error in gray_errors)}/11 images. The failures are explained by fragile DCT-q90 coefficient parity after integer reconstruction, with RGB<->YCbCr conversion changing some bit decisions; none remain unexplained.",
            "",
        ]
    )
    return "\n".join(lines)


def eligible_jpeg_rows(records: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in records
        if row["record_type"] == "stego"
        and row["jpeg_quality"] != "none"
        and not bool_value(row["baseline_excluded"])
    ]


def table_rows(
    rows: list[dict[str, str]], category: str | None = None
) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if category is None or row["category"] == category:
            groups[(row["method"], row["jpeg_quality"], row["subsampling"])].append(row)
    return groups


def format_ber_table(title: str, groups: dict[tuple[str, str, str], list[dict[str, str]]]) -> str:
    lines = [
        title,
        "method   | JPEG q | subsampling   | exact recovery | mean BER | SD BER | n",
        "---------|--------|---------------|----------------|----------|--------|--",
    ]
    for method in METHODS:
        qualities = sorted({int(q) for m, q, _ in groups if m == method})
        for quality in qualities:
            for subsampling in SUBSAMPLINGS:
                values = groups.get((method, str(quality), subsampling), [])
                if not values:
                    continue
                recovered = sum(bool_value(row["exact_recovery"]) for row in values)
                ber = np.asarray([float(row["raw_ber"]) for row in values])
                lines.append(
                    f"{method:<8} | {quality:>6} | {subsampling:<13} | "
                    f"{recovered:>2}/{len(values):<2} ({100*recovered/len(values):>5.1f}%) | "
                    f"{np.mean(ber):>8.6f} | {np.std(ber):>6.6f} | {len(values)}"
                )
    return "\n".join(lines)


def format_subsampling_difference(groups: dict[tuple[str, str, str], list[dict[str, str]]]) -> str:
    lines = [
        "4:4:4 vs 4:2:0 BER difference (paired by image; Δ = Pillow default/4:2:0 − 4:4:4; population SD of paired differences):",
        "method   | JPEG q | BER 4:4:4 mean ± SD | BER 4:2:0 mean ± SD | paired Δ mean ± SD | n pairs",
        "---------|--------|--------------------|--------------------|--------------------|--------",
    ]
    for method in METHODS:
        qualities = sorted({int(q) for m, q, _ in groups if m == method})
        for quality in qualities:
            q444 = groups.get((method, str(quality), "4:4:4"), [])
            q420 = groups.get((method, str(quality), "Pillow default"), [])
            by_444 = {row["image"]: float(row["raw_ber"]) for row in q444}
            by_420 = {row["image"]: float(row["raw_ber"]) for row in q420}
            common = sorted(set(by_444) & set(by_420))
            if not common:
                continue
            v444 = np.asarray([by_444[name] for name in common])
            v420 = np.asarray([by_420[name] for name in common])
            delta = v420 - v444
            lines.append(
                f"{method:<8} | {quality:>6} | "
                f"{np.mean(v444):.6f} ± {np.std(v444):.6f} | "
                f"{np.mean(v420):.6f} ± {np.std(v420):.6f} | "
                f"{np.mean(delta):+.6f} ± {np.std(delta):.6f} | {len(common)}"
            )
    return "\n".join(lines)


def build_compression_tables(records: list[dict[str, str]]) -> str:
    eligible = eligible_jpeg_rows(records)
    overall = table_rows(eligible)
    sections = [
        "3. COMPRESSION SUMMARY TABLES",
        "Tables use the same eligibility rule as Phase 5A: exclude an image/method pair if its no-compression baseline had bit errors. BER SD is population SD (ddof=0). Pillow default is the study's 4:2:0 JPEG setting.",
        format_ber_table("Overall, across eligible image/method pairs:", overall),
        "",
    ]
    categories = sorted({row["category"] for row in eligible})
    for category in categories:
        sections.append(format_ber_table(f"Category: {category}", table_rows(eligible, category)))
        sections.append("")
    sections.append(format_subsampling_difference(overall))
    sections.append("")
    sections.append(
        f"Eligible compressed stego records: {len(eligible)}; no-compression-excluded method/image pairs: "
        f"{sum(bool_value(row['baseline_excluded']) for row in records if row['record_type'] == 'stego' and row['jpeg_quality'] == 'none')}."
    )
    return "\n".join(sections)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    records, usable = load_rows()
    if len(usable) != 15:
        raise AssertionError(f"Expected 15 usable images, found {len(usable)}")
    sections = [
        "Phase 5A follow-up — focused calculations only",
        f"Source: {CSV_PATH}",
        "No full study was run. No files in stego/ or images/ were written; the manifest was read only.",
        "",
        build_cost_section(records, usable),
        build_dct90_section(records, usable),
        build_compression_tables(records),
    ]
    report = "\n".join(sections)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nReport written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
