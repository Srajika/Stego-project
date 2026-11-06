"""Measure DCT brightness-clamp margins without changing stego/dct.py."""

from __future__ import annotations

import base64
import csv
import math
import sys
import tempfile
from dataclasses import dataclass
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
    DCT_POSITION,
    EMBED_QUALITY,
    JPEG_LUMINANCE_QUANT_TABLE,
    MAX_PIXEL_CHANGE_PER_QUANT_STEP,
    MAX_Y,
    MIN_Y,
    REPETITION,
    extract as dct_extract,
    hide as dct_hide,
    quantization_table,
    read_majority_bits as dct_read_majority_bits,
)
from stego.tamper import make_tag


MANIFEST_PATH = ROOT / "images" / "dataset" / "manifest.csv"
DATASET_ROOT = ROOT / "images" / "dataset"
MARGINS = (16, 12, 8, 6, 4, 2, 0)
QUALITIES = (50, 75)
PHASE5A_SEED = 20261105
PHASE5A_SOURCE_BYTES = 40
PHASE5A_PASSWORD = "phase5a-compression-study-fixed-password"


@dataclass
class Sample:
    name: str
    category: str
    pixels: np.ndarray
    seed: int
    phase5a_bits: str
    is_synthetic: bool = False


def payload_for_seed(seed: int) -> tuple[str, str]:
    """Reproduce the Phase 5A seeded 40-byte → Base64 → framed payload."""
    source = np.random.default_rng(seed).bytes(PHASE5A_SOURCE_BYTES)
    message = base64.b64encode(source).decode("ascii")
    tag = make_tag(message.encode("utf-8"), PHASE5A_PASSWORD)
    return message, bytes_to_bits(build_payload(message, tag))


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def load_samples() -> list[Sample]:
    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    usable = [
        row for row in rows
        if row.get("usable_for_main_studies", "").strip().lower() == "true"
    ]
    usable.sort(key=lambda row: (row["category"], row["new_filename"]))

    samples: list[Sample] = []
    for index, row in enumerate(usable):
        seed = PHASE5A_SEED + index
        _, bits = payload_for_seed(seed)
        samples.append(
            Sample(
                name=row["new_filename"],
                category=row["category"],
                pixels=read_rgb(DATASET_ROOT / row["category"] / row["new_filename"]),
                seed=seed,
                phase5a_bits=bits,
            )
        )

    # These pixel arrays match tests/test_dct.py's three headroom cases exactly.
    gradient_cover = np.empty((400, 400, 3), dtype=np.uint8)
    gradient = np.tile(np.linspace(0, 255, 400, dtype=np.uint8), (400, 1))
    gradient_cover[:, :, :] = gradient[:, :, None]
    gradient_cover[:, :300, :] = 255
    synthetic = (
        ("synthetic_dark", "synthetic", np.zeros((400, 400, 3), dtype=np.uint8)),
        ("synthetic_bright", "synthetic", np.full((400, 400, 3), 255, dtype=np.uint8)),
        ("synthetic_white_area_gradient", "synthetic", gradient_cover),
    )
    for offset, (name, category, pixels) in enumerate(synthetic, start=len(samples)):
        seed = PHASE5A_SEED + offset
        _, bits = payload_for_seed(seed)
        samples.append(
            Sample(
                name=name,
                category=category,
                pixels=pixels,
                seed=seed,
                phase5a_bits=bits,
                is_synthetic=True,
            )
        )
    return samples


def blockify(plane: np.ndarray) -> tuple[np.ndarray, int, int]:
    block_rows, block_columns = plane.shape[0] // 8, plane.shape[1] // 8
    height, width = block_rows * 8, block_columns * 8
    blocks = (
        plane[:height, :width]
        .reshape(block_rows, 8, block_columns, 8)
        .transpose(0, 2, 1, 3)
        .reshape(block_rows * block_columns, 8, 8)
    )
    return blocks, height, width


def un_blockify(blocks: np.ndarray, height: int, width: int) -> np.ndarray:
    block_rows, block_columns = height // 8, width // 8
    return (
        blocks.reshape(block_rows, block_columns, 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(height, width)
    )


def embed_plane(
    luma: np.ndarray,
    payload_bits: str,
    quality: int,
    margin: int,
) -> tuple[np.ndarray, float]:
    """Standalone DCT embedding in Y; return the integer plane and clamp fraction."""
    if not 0 <= margin <= 127:
        raise ValueError(f"Invalid symmetric clamp margin: {margin}")
    upper = 255 - margin
    original = luma.astype(np.uint8, copy=False)
    clamped = np.clip(original.astype(np.float64), margin, upper)
    blocks, covered_height, covered_width = blockify(clamped)
    covered_pixels = covered_height * covered_width
    moved_fraction = float(
        np.count_nonzero(
            original[:covered_height, :covered_width]
            != clamped[:covered_height, :covered_width]
        )
        / covered_pixels
    )

    capacity = len(blocks) // REPETITION
    if len(payload_bits) > capacity:
        raise ValueError(f"Payload needs {len(payload_bits)} bits; capacity is {capacity}")

    table = quantization_table(quality)
    payload_size = len(payload_bits)
    selected = np.concatenate(
        [np.arange(payload_size) + repeat * capacity for repeat in range(REPETITION)]
    )
    changed = dctn(blocks[selected].astype(np.float64) - 128.0,
                   axes=(-2, -1), norm="ortho")
    row, column = DCT_POSITION
    step = table[row, column]
    original_quantized = changed[:, row, column] / step
    nearest = np.rint(original_quantized).astype(np.int64)
    targets = np.tile(
        np.fromiter((int(bit) for bit in payload_bits), dtype=np.int64), REPETITION
    )
    wrong_parity = (nearest & 1) != targets
    lower, upper_neighbor = nearest - 1, nearest + 1
    nearer = np.where(
        np.abs(original_quantized - lower)
        <= np.abs(original_quantized - upper_neighbor),
        lower,
        upper_neighbor,
    )
    quantized = np.where(wrong_parity, nearer, nearest)
    changed[:, row, column] = quantized * step
    reconstructed = idctn(changed, axes=(-2, -1), norm="ortho") + 128.0
    reconstructed = np.clip(np.rint(reconstructed), 0, 255).astype(np.uint8)

    output_blocks = blocks.copy()
    output_blocks[selected] = reconstructed
    output_luma = clamped.copy()
    output_luma[:covered_height, :covered_width] = un_blockify(
        output_blocks, covered_height, covered_width
    )
    return output_luma.astype(np.uint8), moved_fraction


def embed_rgb(
    cover_rgb: np.ndarray,
    payload_bits: str,
    quality: int,
    margin: int,
) -> tuple[np.ndarray, float]:
    """Reproduce RGB→YCbCr, Y clamp/DCT, YCbCr→RGB, and edge restoration."""
    height, width = cover_rgb.shape[:2]
    covered_height, covered_width = height // 8 * 8, width // 8 * 8
    ycbcr = np.asarray(Image.fromarray(cover_rgb).convert("YCbCr"), dtype=np.uint8)
    output_y, moved_fraction = embed_plane(
        ycbcr[:, :, 0], payload_bits, quality, margin
    )
    modified = ycbcr.copy()
    modified[:covered_height, :covered_width, 0] = output_y[
        :covered_height, :covered_width
    ]
    result = np.array(
        Image.frombytes("YCbCr", (width, height), modified.tobytes()).convert("RGB"),
        dtype=np.uint8,
    )
    result[covered_height:, :, :] = cover_rgb[covered_height:, :, :]
    result[:, covered_width:, :] = cover_rgb[:, covered_width:, :]
    return result, moved_fraction


def extract_bits(rgb: np.ndarray, quality: int, bit_count: int) -> str:
    """Standalone Y extraction and three-copy parity majority vote."""
    ycbcr = np.asarray(Image.fromarray(rgb).convert("YCbCr"), dtype=np.uint8)
    blocks, _, _ = blockify(ycbcr[:, :, 0])
    capacity = len(blocks) // REPETITION
    if not 0 <= bit_count <= capacity:
        raise ValueError(f"Requested {bit_count} bits; capacity is {capacity}")
    table = quantization_table(quality)
    indices = np.concatenate(
        [np.arange(bit_count) + repeat * capacity for repeat in range(REPETITION)]
    )
    row, column = DCT_POSITION
    transformed = dctn(blocks[indices].astype(np.float64) - 128.0,
                       axes=(-2, -1), norm="ortho")
    quantized = np.rint(
        transformed[:, row, column] / table[row, column]
    ).astype(np.int64)
    votes = (quantized & 1).reshape(REPETITION, bit_count).T
    return "".join(str(int(bit)) for bit in (votes.sum(axis=1) >= 2))


def bit_errors(expected: str, observed: str) -> int:
    return sum(left != right for left, right in zip(expected, observed)) + abs(
        len(expected) - len(observed)
    )


def clamp_fraction(cover_rgb: np.ndarray, margin: int) -> float:
    height, width = cover_rgb.shape[:2]
    covered_height, covered_width = height // 8 * 8, width // 8 * 8
    y = np.asarray(Image.fromarray(cover_rgb).convert("YCbCr"), dtype=np.uint8)[:, :, 0]
    lower, upper = margin, 255 - margin
    region = y[:covered_height, :covered_width]
    return float(np.count_nonzero((region < lower) | (region > upper)) / region.size)


def image_metrics(cover: np.ndarray, candidate: np.ndarray) -> tuple[float, float, int]:
    psnr = float(peak_signal_noise_ratio(cover, candidate, data_range=255))
    ssim = float(
        structural_similarity(cover, candidate, data_range=255, channel_axis=-1)
    )
    difference = np.abs(candidate.astype(np.int16) - cover.astype(np.int16))
    return psnr, ssim, int(difference.max())


def current_margin_validation(samples: list[Sample]) -> list[str]:
    if MAX_Y != 255 - MIN_Y:
        raise AssertionError(f"Current limits are not symmetric: MIN_Y={MIN_Y}, MAX_Y={MAX_Y}")
    current_margin = MIN_Y
    selected = [samples[0], samples[1]]
    lines = [
        "CURRENT IMPLEMENTATION VALIDATION (current margin, actual hide/extract, two images)",
        f"Current constants: MIN_Y={MIN_Y}, MAX_Y={MAX_Y}, margin={current_margin}; default quality={EMBED_QUALITY}.",
        "image          | q  | pixels identical | custom errors | hide/extract raw errors | extract status | message matches",
        "---------------|----|------------------|---------------|-------------------------|----------------|----------------",
    ]
    with tempfile.TemporaryDirectory(prefix="clamp-margin-verify-") as temp_name:
        temp_root = Path(temp_name)
        for sample in selected:
            cover_path = temp_root / f"{sample.name}.png"
            Image.fromarray(sample.pixels).save(cover_path, format="PNG")
            message, expected_bits = payload_for_seed(sample.seed)
            for quality in QUALITIES:
                stego_path = temp_root / f"{sample.name}-q{quality}.png"
                dct_hide(
                    str(cover_path), str(stego_path), message,
                    password=PHASE5A_PASSWORD, quality=quality,
                )
                official_rgb = read_rgb(stego_path)
                custom_rgb, _ = embed_rgb(
                    sample.pixels, expected_bits, quality, current_margin
                )
                custom_bits = extract_bits(custom_rgb, quality, len(expected_bits))
                official_bits = dct_read_majority_bits(
                    str(stego_path), bit_count=len(expected_bits), quality=quality
                )
                api_result = dct_extract(
                    str(stego_path), password=PHASE5A_PASSWORD, quality=quality
                )
                errors = bit_errors(expected_bits, custom_bits)
                if not np.array_equal(custom_rgb, official_rgb):
                    raise AssertionError(f"Custom RGB output differs from dct.hide for {sample.name}, q{quality}")
                if custom_bits != official_bits:
                    raise AssertionError(f"Custom extraction differs from dct.extract bits for {sample.name}, q{quality}")
                if errors != 0 or api_result.get("message") != message:
                    raise AssertionError(f"Current-margin round trip mismatch for {sample.name}, q{quality}: {api_result}")
                lines.append(
                    f"{sample.name:<14} | {quality:>2} | yes              | {errors:>13} | "
                    f"{bit_errors(expected_bits, official_bits):>23} | "
                    f"{api_result.get('status', 'unknown'):<14} | yes"
                )
    lines.append("All four checks match dct.hide output pixels and dct.extract results.")
    return lines


def theoretical_bound(quality: int) -> tuple[int, float, float]:
    row, column = DCT_POSITION
    positions = np.arange(8)

    def basis(frequency: int) -> np.ndarray:
        scale = math.sqrt(1 / 8) if frequency == 0 else math.sqrt(2 / 8)
        return scale * np.cos(math.pi * (2 * positions + 1) * frequency / 16)

    amplitude = float(np.max(np.abs(np.outer(basis(row), basis(column)))))
    step = float(quantization_table(quality)[row, column])
    bound = step * amplitude + 0.5
    return int(step), amplitude, bound


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    samples = load_samples()
    if len(samples) != 18 or sum(not sample.is_synthetic for sample in samples) != 15:
        raise AssertionError("Expected 15 usable dataset images and 3 synthetic test images")

    synthetic_capacity = (400 // 8) * (400 // 8) // REPETITION
    phase5a_full_length = len(samples[-1].phase5a_bits)
    print("PHASE 3B CLAMP-MARGIN MEASUREMENT")
    print("Standalone reimplementation; source images and stego/dct.py are read-only.")
    print(
        f"Phase 5A payload: {phase5a_full_length} bits per usable image; "
        f"synthetic 400x400 capacity={synthetic_capacity} bits. Synthetic rows measure "
        f"the first {synthetic_capacity} bits of their seeded Phase 5A payload "
        f"({synthetic_capacity}/{phase5a_full_length}); all dataset rows use the full payload."
    )
    print("Clamp fraction is the fraction of complete-block Y samples moved by the clamp.")
    print("Max change is the largest absolute RGB channel change against the cover.")
    print()
    print("\n".join(current_margin_validation(samples)))

    all_results: dict[int, dict[int, list[dict[str, int | float | str]]]] = {
        quality: {margin: [] for margin in MARGINS} for quality in QUALITIES
    }
    for quality in QUALITIES:
        print()
        print(f"TABLE — EMBEDDING QUALITY {quality}")
        print("margin | image                         | bit errors | PSNR dB | SSIM    | max | clamp-moved %")
        print("-------|-------------------------------|------------|---------|---------|-----|--------------")
        for margin in MARGINS:
            lower, upper = margin, 255 - margin
            for sample in samples:
                expected_bits = sample.phase5a_bits
                total_bits = len(expected_bits)
                payload_bits = (
                    expected_bits[:synthetic_capacity]
                    if sample.is_synthetic
                    else expected_bits
                )
                stego, moved = embed_rgb(sample.pixels, payload_bits, quality, margin)
                observed = extract_bits(stego, quality, len(payload_bits))
                errors = bit_errors(payload_bits, observed)
                psnr, ssim, max_change = image_metrics(sample.pixels, stego)
                actual_moved = clamp_fraction(sample.pixels, margin)
                if not math.isclose(moved, actual_moved, rel_tol=0.0, abs_tol=1e-15):
                    raise AssertionError(f"Clamp fraction mismatch for {sample.name}, M={margin}")
                row: dict[str, int | float | str] = {
                    "image": sample.name,
                    "category": sample.category,
                    "errors": errors,
                    "bits": len(payload_bits),
                    "full_bits": total_bits,
                    "psnr": psnr,
                    "ssim": ssim,
                    "max_change": max_change,
                    "clamp_fraction": moved,
                }
                all_results[quality][margin].append(row)
                bit_cell = f"{errors}/{len(payload_bits)}"
                if sample.is_synthetic:
                    bit_cell += "*"
                print(
                    f"{margin:>6} | {sample.name:<29} | {bit_cell:>10} | "
                    f"{psnr:>7.3f} | {ssim:>7.5f} | {max_change:>3} | {100*moved:>12.5f}%"
                )
        print("* Synthetic 400x400 cases use the available-capacity prefix of the seeded Phase 5A bits.")

    print()
    print("ZERO-ERROR MARGIN SUMMARY (all 15 usable images + all 3 synthetic cases)")
    print("quality | margin | images with zero errors | worst bit errors")
    print("--------|--------|------------------------|-----------------")
    smallest_by_quality: dict[int, int | None] = {}
    for quality in QUALITIES:
        smallest: int | None = None
        for margin in MARGINS:
            rows = all_results[quality][margin]
            zero_count = sum(int(row["errors"]) == 0 for row in rows)
            worst = max(int(row["errors"]) for row in rows)
            print(f"{quality:>7} | {margin:>6} | {zero_count:>7}/18              | {worst:>15}")
            if zero_count == len(samples):
                smallest = margin
        smallest_by_quality[quality] = smallest
    combined = [
        margin
        for margin in MARGINS
        if all(
            all(int(row["errors"]) == 0 for row in all_results[quality][margin])
            for quality in QUALITIES
        )
    ]
    smallest_combined = combined[-1] if combined else None
    print()
    for quality in QUALITIES:
        print(f"Smallest tested margin with zero errors at q{quality}: {smallest_by_quality[quality]}")
    print(f"Smallest tested margin with zero errors at both qualities: {smallest_combined}")

    bounds = {quality: theoretical_bound(quality) for quality in QUALITIES}
    print()
    print("THEORETICAL ONE-STEP DCT PIXEL-CHANGE BOUND")
    print("quality | quant step | max basis amplitude | step × amplitude | +0.5 rounding bound")
    print("--------|------------|---------------------|-----------------|--------------------")
    for quality in QUALITIES:
        step, amplitude, bound = bounds[quality]
        print(
            f"{quality:>7} | {step:>10} | {amplitude:>19.8f} | "
            f"{step*amplitude:>15.6f} | {bound:>18.6f} Y levels"
        )
    if not math.isclose(bounds[EMBED_QUALITY][2] - 0.5,
                        MAX_PIXEL_CHANGE_PER_QUANT_STEP, abs_tol=1e-12):
        raise AssertionError("The independently computed q75 amplitude differs from the imported DCT constant")

    maximum_bound = max(bound for _, _, bound in bounds.values())
    if smallest_combined is None:
        print("No tested margin produced zero errors for both qualities on every image.")
        proposed = None
    else:
        safety_target = 1.5 * maximum_bound
        eligible = [margin for margin in MARGINS if margin >= max(smallest_combined, safety_target)]
        proposed = min(eligible) if eligible else max(MARGINS)
        print()
        print(
            f"Data-based proposal (not applied): margin={proposed}; it is at least the "
            f"smallest observed all-image margin {smallest_combined} and is "
            f"{proposed/maximum_bound:.3f}x the worst theoretical bound "
            f"({maximum_bound:.3f} Y levels; 1.5x target={safety_target:.3f})."
        )
        print("This remains a proposal only; dct.py has not been changed.")
    print()
    print("Files written by this script: none (the dct.hide validation PNGs are temporary and cleaned up).")


if __name__ == "__main__":
    main()
