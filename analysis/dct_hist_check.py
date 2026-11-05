"""Diagnose coefficient parity for this project's targeted DCT detector."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

from analysis.dct_detect_check import (
    RANDOM_SEED,
    THRESHOLD,
    _embed_payload_bytes,
    _processed_rgb,
    _save_processed_control,
    _uniform_payload,
)
from stego.common import STOP_MARKER, TAG_BYTES
from stego.dct import (
    DCT_POSITION,
    EMBED_QUALITY,
    MAX_Y,
    MIN_Y,
    REPETITION,
    _blocks_from_plane,
    capacity_bits,
    quantization_table,
)
from stego.detect_dct import (
    _measure,
    _read_quantized_coefficients,
    detect_dct_parity,
)


CROP_SIZES = ((256, 192), (512, 384), (800, 600), (960, 640))


def _write_random_parity_every_block(
    control_path: Path, stego_path: Path, bits: np.ndarray
) -> None:
    """Write one requested parity into every complete DCT block."""
    with Image.open(control_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    ycbcr = np.asarray(Image.fromarray(rgb).convert("YCbCr"), dtype=np.uint8)
    y = np.clip(ycbcr[:, :, 0].astype(np.float64), MIN_Y, MAX_Y)
    blocks, covered_height, covered_width = _blocks_from_plane(y)
    if bits.size != len(blocks):
        raise ValueError("Calibration needs one random parity bit per block")

    transformed = dctn(
        blocks.astype(np.float64) - 128.0,
        axes=(-2, -1),
        norm="ortho",
    )
    row, column = DCT_POSITION
    step = quantization_table(EMBED_QUALITY)[row, column]
    scaled = transformed[:, row, column] / step
    nearest = np.rint(scaled).astype(np.int64)
    wrong_parity = (nearest & 1) != bits.astype(np.int64)
    lower = nearest - 1
    upper = nearest + 1
    nearer = np.where(np.abs(scaled - lower) <= np.abs(scaled - upper), lower, upper)
    transformed[:, row, column] = np.where(wrong_parity, nearer, nearest) * step

    reconstructed = idctn(transformed, axes=(-2, -1), norm="ortho") + 128.0
    reconstructed = np.clip(np.rint(reconstructed), 0, 255).astype(np.uint8)
    output_ycbcr = ycbcr.copy()
    output_blocks = blocks.copy()
    output_blocks[:] = reconstructed
    block_rows = covered_height // 8
    block_columns = covered_width // 8
    grid = output_blocks.reshape(block_rows, block_columns, 8, 8)
    plane = grid.transpose(0, 2, 1, 3).reshape(covered_height, covered_width)
    output_ycbcr[:covered_height, :covered_width, 0] = plane
    output_rgb = np.array(
        Image.frombytes(
            "YCbCr", (rgb.shape[1], rgb.shape[0]), output_ycbcr.tobytes()
        ).convert("RGB"),
        dtype=np.uint8,
    )
    output_rgb[covered_height:, :, :] = rgb[covered_height:, :, :]
    output_rgb[:, covered_width:, :] = rgb[:, covered_width:, :]
    Image.fromarray(output_rgb).save(stego_path, format="PNG")


def _histogram_report(control_path: Path, stego_path: Path) -> None:
    rows = (
        ("processed control", _read_quantized_coefficients(control_path)),
        ("100% uniform-byte DCT stego", _read_quantized_coefficients(stego_path)),
    )
    print("\nCoefficient histogram, quantized DCT coefficient values -4..4")
    print("image                         | " + " ".join(f"{value:>7}" for value in range(-4, 5)) + " | even fraction | pair statistic")
    measures: dict[str, dict] = {}
    histograms: dict[str, dict[int, int]] = {}
    for label, values in rows:
        counts = {value: int(np.count_nonzero(values == value)) for value in range(-4, 5)}
        histograms[label] = counts
        measure = _measure(values)
        measures[label] = measure
        even_fraction = float(np.mean(np.remainder(values, 2) == 0))
        print(
            f"{label:<29} | "
            + " ".join(f"{counts[value]:>7}" for value in range(-4, 5))
            + f" | {even_fraction:>12.6f} | {measure['statistic']:.8g} "
            + f"(p={measure['p_value']:.8g}, used={measure['pairs_used']}, "
            + f"skipped={measure['pairs_skipped']}, dof={measure['dof']})"
        )

    control_label, stego_label = rows[0][0], rows[1][0]
    for pair_label, values in (("(0,1)", (0, 1)), ("(-2,-1)", (-2, -1))):
        control_cells = tuple(histograms[control_label][value] for value in values)
        stego_cells = tuple(histograms[stego_label][value] for value in values)
        equal_after = stego_cells[0] == stego_cells[1]
        print(
            f"Pair {pair_label}: control counts={control_cells}; "
            f"stego counts={stego_cells}; equalized_after_embedding={equal_after}."
        )

    control_even = float(np.mean(np.remainder(rows[0][1], 2) == 0))
    stego_even = float(np.mean(np.remainder(rows[1][1], 2) == 0))
    print(
        f"Even fraction: {control_even:.6f} -> {stego_even:.6f}; "
        f"distance from 0.5: {abs(control_even - 0.5):.6f} -> "
        f"{abs(stego_even - 0.5):.6f}; moved_to_0.5={abs(stego_even - 0.5) < abs(control_even - 0.5)}."
    )
    print(
        "Pair-test result: processed control versus 100% uniform-byte stego "
        f"statistic {measures[control_label]['statistic']:.8g} -> "
        f"{measures[stego_label]['statistic']:.8g}."
    )


def _parity_calibration(scratch: Path, cover_path: Path) -> None:
    """Calibrate parity chi-square on 100 differently sized random crops."""
    with Image.open(cover_path) as source:
        cover = np.asarray(source.convert("RGB"), dtype=np.uint8)
    height, width = cover.shape[:2]
    crop_rng = np.random.default_rng(RANDOM_SEED)
    crop_path = scratch / "calibration_crop.png"
    control_path = scratch / "calibration_control.png"
    stego_path = scratch / "calibration_full.png"
    statistics: list[float] = []
    p_values: list[float] = []
    block_counts: list[int] = []

    for index in range(100):
        crop_width, crop_height = CROP_SIZES[index % len(CROP_SIZES)]
        left = int(crop_rng.integers(0, width - crop_width + 1))
        top = int(crop_rng.integers(0, height - crop_height + 1))
        Image.fromarray(cover[top:top + crop_height, left:left + crop_width]).save(
            crop_path, format="PNG"
        )
        _save_processed_control(crop_path, control_path)
        blocks = _read_quantized_coefficients(control_path).size
        random_parities = np.random.default_rng(RANDOM_SEED + index + 1).integers(
            0, 2, size=blocks, dtype=np.uint8
        )
        _write_random_parity_every_block(control_path, stego_path, random_parities)
        result = detect_dct_parity(stego_path, threshold=THRESHOLD)
        statistics.append(result["statistic"])
        p_values.append(result["p_value"])
        block_counts.append(result["blocks"])

    statistic_array = np.asarray(statistics, dtype=np.float64)
    p_array = np.asarray(p_values, dtype=np.float64)
    print("\n100-crop random-parity calibration")
    print(
        "100 crops from images/small.png; sizes "
        + ", ".join(f"{w}x{h}" for w, h in CROP_SIZES)
        + "; crops were converted to processed controls and random parity was "
        "written to every complete block."
    )
    print(
        f"blocks_per_crop={min(block_counts)}-{max(block_counts)}; "
        f"mean_statistic={statistic_array.mean():.6f}"
    )
    for threshold in (0.5, 0.1, 0.01):
        print(f"fraction(p_value >= {threshold:g})={np.mean(p_array >= threshold):.2f}")
    print(f"finite_p_values={int(np.isfinite(p_array).sum())}/100")
    looks_calibrated = (
        abs(float(statistic_array.mean()) - 1.0) <= 0.5
        and all(abs(float(np.mean(p_array >= cutoff)) - expected) <= 0.12
                for cutoff, expected in ((0.5, 0.5), (0.1, 0.9), (0.01, 0.99)))
    )
    mean_assessment = (
        "below the expected mean of about 1"
        if statistic_array.mean() < 1.0
        else "at or above the expected mean of about 1"
    )
    print(
        "Calibration assessment: mean is "
        f"{mean_assessment}; the requested p-value fractions are "
        f"{'close to expectations' if looks_calibrated else 'off expectations'}."
    )


def _false_alarm_check(scratch: Path, raw_root: Path) -> None:
    paths = sorted(path for path in raw_root.rglob("*") if path.is_file())
    print("\nFalse-alarm check: processed controls from every file under images/raw/**")
    print("image | even_fraction | statistic | p_value | verdict")
    control_count = 0
    false_alarms = 0
    skipped: list[tuple[Path, str]] = []
    for index, source_path in enumerate(paths):
        try:
            with Image.open(source_path) as source:
                source.load()
                rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        except Exception as error:
            skipped.append((source_path, f"{type(error).__name__}: {error}"))
            continue

        processed_path = scratch / f"raw_control_{index:03d}.png"
        Image.fromarray(_processed_rgb(rgb)).save(processed_path, format="PNG")
        result = detect_dct_parity(processed_path, threshold=THRESHOLD)
        control_count += 1
        false_alarms += result["verdict"] == "DCT embedding likely"
        print(
            f"{source_path.relative_to(raw_root)} | "
            f"{result['even_fraction']:.6f} | {result['statistic']:.8g} | "
            f"{result['p_value']:.8g} | {result['verdict']}"
        )

    for source_path, reason in skipped:
        print(f"SKIPPED unreadable: {source_path.relative_to(raw_root)} ({reason})")
    print(
        f"Clean processed controls wrongly flagged embedding likely: "
        f"{false_alarms}/{control_count}. Readable={control_count}; "
        f"unreadable={len(skipped)}."
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    cover_path = project_root / "images" / "test.jpg"
    calibration_cover = project_root / "images" / "small.png"
    raw_root = project_root / "images" / "raw"
    if not cover_path.is_file() or not calibration_cover.is_file() or not raw_root.is_dir():
        raise FileNotFoundError("Required DCT diagnostic images are missing")

    with Image.open(cover_path) as image:
        width, height = image.size
    capacity = capacity_bits(width, height)
    payload_bytes = capacity // 8
    overhead = len(STOP_MARKER) + TAG_BYTES
    payload = _uniform_payload(payload_bytes, RANDOM_SEED + 100)
    print(f"Histogram cover: images/test.jpg ({width}x{height}); DCT capacity={capacity} bits.")
    print(
        f"Full payload: {len(payload)} bytes including {overhead}-byte marker/tag "
        f"overhead ({len(payload) * 8 / capacity:.6%} of bit capacity)."
    )
    print(
        f"Detector target: position={DCT_POSITION}, quality={EMBED_QUALITY}, "
        f"threshold={THRESHOLD}; block size=8x8; clamp=[{MIN_Y},{MAX_Y}]."
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        scratch = Path(temp_dir)
        control_path = scratch / "processed_test_control.png"
        stego_path = scratch / "full_uniform_stego.png"
        _save_processed_control(cover_path, control_path)
        _embed_payload_bytes(cover_path, stego_path, payload)
        _histogram_report(control_path, stego_path)
        _parity_calibration(scratch, calibration_cover)
        _false_alarm_check(scratch, raw_root)


if __name__ == "__main__":
    main()
