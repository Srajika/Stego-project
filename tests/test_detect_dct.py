"""Assert-style checks and calibration report for the targeted DCT detector."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

from stego.common import STOP_MARKER, TAG_BYTES
from stego.dct import (
    DCT_POSITION,
    EMBED_QUALITY,
    MAX_Y,
    MIN_Y,
    REPETITION,
    _blocks_from_plane,
    hide as dct_hide,
    quantization_table,
)
from stego.detect_dct import (
    _measure,
    _read_quantized_coefficients,
    detect_dct,
    detect_dct_parity,
)
from stego.lsb import hide as lsb_hide
from stego.tamper import make_tag


def _processed_rgb(rgb: np.ndarray) -> np.ndarray:
    """Apply the embedder's no-message RGB -> clamp-Y -> RGB processing."""
    ycbcr = np.array(Image.fromarray(rgb).convert("YCbCr"), dtype=np.uint8)
    ycbcr[:, :, 0] = np.clip(
        ycbcr[:, :, 0].astype(np.int16), MIN_Y, MAX_Y
    ).astype(np.uint8)
    height, width = ycbcr.shape[:2]
    return np.asarray(
        Image.frombytes("YCbCr", (width, height), ycbcr.tobytes()).convert("RGB"),
        dtype=np.uint8,
    )


def _save_processed_control(source_path: Path, output_path: Path) -> None:
    with Image.open(source_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    Image.fromarray(_processed_rgb(rgb)).save(output_path, format="PNG")


def _embed_bits(
    cover_path: Path,
    output_path: Path,
    bits: np.ndarray,
    *,
    every_block: bool = False,
) -> None:
    """Write bits using the same coefficient adjustment as ``dct.hide``.

    With ``every_block=True``, one target parity is written to every complete
    block. Otherwise each payload bit is repeated into the embedder's three
    separated block regions.
    """
    with Image.open(cover_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    ycbcr = np.asarray(Image.fromarray(rgb).convert("YCbCr"), dtype=np.uint8)
    y = np.clip(ycbcr[:, :, 0].astype(np.float64), MIN_Y, MAX_Y)
    blocks, covered_height, covered_width = _blocks_from_plane(y)
    capacity = len(blocks) // REPETITION

    if every_block:
        if bits.size != len(blocks):
            raise ValueError("Every-block calibration needs one bit per block")
        selected_indices = np.arange(len(blocks), dtype=np.int64)
        target_bits = bits.astype(np.int64)
    else:
        if bits.size > capacity:
            raise ValueError("Framed payload exceeds DCT capacity")
        selected_indices = np.concatenate(
            [
                np.arange(bits.size, dtype=np.int64) + repeat * capacity
                for repeat in range(REPETITION)
            ]
        )
        target_bits = np.tile(bits.astype(np.int64), REPETITION)

    changed_blocks = blocks[selected_indices].astype(np.float64) - 128.0
    transformed = dctn(changed_blocks, axes=(-2, -1), norm="ortho")
    row, column = DCT_POSITION
    step = quantization_table(EMBED_QUALITY)[row, column]
    scaled = transformed[:, row, column] / step
    nearest = np.rint(scaled).astype(np.int64)
    wrong_parity = (nearest & 1) != target_bits
    lower = nearest - 1
    upper = nearest + 1
    nearer = np.where(np.abs(scaled - lower) <= np.abs(scaled - upper), lower, upper)
    parity_quantized = np.where(wrong_parity, nearer, nearest)
    transformed[:, row, column] = parity_quantized * step
    reconstructed = idctn(transformed, axes=(-2, -1), norm="ortho") + 128.0
    reconstructed = np.clip(np.rint(reconstructed), 0, 255).astype(np.uint8)

    modified_ycbcr = ycbcr.copy()
    block_height, block_width = blocks.shape[1:]
    block_rows = covered_height // block_height
    block_columns = covered_width // block_width
    output_blocks = blocks.copy()
    output_blocks[selected_indices] = reconstructed
    grid = output_blocks.reshape(block_rows, block_columns, block_height, block_width)
    plane = grid.transpose(0, 2, 1, 3).reshape(covered_height, covered_width)
    modified_ycbcr[:covered_height, :covered_width, 0] = plane
    output_rgb = np.array(
        Image.frombytes(
            "YCbCr", (rgb.shape[1], rgb.shape[0]), modified_ycbcr.tobytes()
        ).convert("RGB"),
        dtype=np.uint8,
    )
    output_rgb[covered_height:, :, :] = rgb[covered_height:, :, :]
    output_rgb[:, covered_width:, :] = rgb[:, covered_width:, :]
    Image.fromarray(output_rgb).save(output_path, format="PNG")


def _report(label: str, result: dict) -> None:
    print(
        f"{label:<46} | statistic={result['statistic']:.12g} "
        f"| p_value={result['p_value']:.12g} "
        f"| pairs_used={result['pairs_used']} "
        f"| pairs_skipped={result['pairs_skipped']} "
        f"| dof={result['dof']} | blocks={result['blocks']} "
        f"| verdict={result['verdict']}"
    )


def _assert_parity_result(result: dict) -> None:
    assert {
        "statistic",
        "p_value",
        "even_fraction",
        "blocks",
        "verdict",
    } <= set(result)
    assert result["statistic"] >= 0.0
    assert np.isfinite(result["p_value"])
    assert 0.0 <= result["p_value"] <= 1.0
    assert 0.0 <= result["even_fraction"] <= 1.0
    assert result["blocks"] > 0


def _report_parity(label: str, result: dict) -> None:
    print(
        f"{label:<46} | statistic={result['statistic']:.12g} "
        f"| p_value={result['p_value']:.12g} "
        f"| even_fraction={result['even_fraction']:.8f} "
        f"| blocks={result['blocks']} | verdict={result['verdict']}"
    )


def _assert_result(result: dict) -> None:
    assert {
        "statistic",
        "p_value",
        "pairs_used",
        "pairs_skipped",
        "dof",
        "blocks",
        "verdict",
    } <= set(result)
    assert result["statistic"] >= 0.0
    assert np.isfinite(result["p_value"])
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["pairs_used"] >= 2
    assert result["pairs_skipped"] >= 0
    assert result["dof"] == result["pairs_used"] - 1
    assert result["blocks"] > 0


def _fully_embedded_calibration(scratch: Path, cover_path: Path) -> None:
    """Use 100 differently sized crops, assigning random parity to every block."""
    crop_sizes = [(256, 192), (512, 384), (800, 600), (960, 640)]
    crop_rng = np.random.default_rng(20261105)
    statistics: list[float] = []
    degrees_of_freedom: list[int] = []
    p_values: list[float] = []

    with Image.open(cover_path) as source:
        cover = np.asarray(source.convert("RGB"), dtype=np.uint8)
    image_height, image_width = cover.shape[:2]

    crop_path = scratch / "calibration_crop.png"
    control_path = scratch / "calibration_control.png"
    stego_path = scratch / "calibration_full.png"
    for index in range(100):
        width, height = crop_sizes[index % len(crop_sizes)]
        left = int(crop_rng.integers(0, image_width - width + 1))
        top = int(crop_rng.integers(0, image_height - height + 1))
        crop = cover[top:top + height, left:left + width, :]
        Image.fromarray(crop).save(crop_path, format="PNG")
        _save_processed_control(crop_path, control_path)

        block_count = _read_quantized_coefficients(control_path).size
        random_parities = np.random.default_rng(20261105 + index).integers(
            0, 2, size=block_count, dtype=np.uint8
        )
        _embed_bits(control_path, stego_path, random_parities, every_block=True)
        result = detect_dct(stego_path)
        statistics.append(result["statistic"])
        degrees_of_freedom.append(result["dof"])
        p_values.append(result["p_value"])

    mean_statistic = float(np.mean(statistics))
    mean_dof = float(np.mean(degrees_of_freedom))
    finite_p_values = np.asarray(p_values, dtype=np.float64)
    print(
        "\nPAIR detector (retained for comparison; known to be mismatched "
        "to the nearer-parity embedding)"
    )
    print(
        "Crops: 25 each at 256x192, 512x384, 800x600, and 960x640; "
        "each crop was clamped to the processed control before random parity "
        "was written into every complete block."
    )
    print(
        f"mean_statistic={mean_statistic:.6f}; mean_dof={mean_dof:.6f}; "
        f"mean_statistic/mean_dof={mean_statistic / mean_dof:.6f}"
    )
    for threshold in (0.5, 0.1, 0.01):
        fraction = float(np.mean(finite_p_values >= threshold))
        print(f"fraction(p_value >= {threshold:g})={fraction:.2f}")
    print(f"finite_p_values={int(np.isfinite(finite_p_values).sum())}/100")


def _fully_embedded_parity_calibration(scratch: Path, cover_path: Path) -> None:
    """Regress detect_dct_parity on 100 crops with random parity in every block."""
    crop_sizes = [(256, 192), (512, 384), (800, 600), (960, 640)]
    crop_rng = np.random.default_rng(20261105)
    statistics: list[float] = []
    p_values: list[float] = []

    with Image.open(cover_path) as source:
        cover = np.asarray(source.convert("RGB"), dtype=np.uint8)
    image_height, image_width = cover.shape[:2]

    crop_path = scratch / "parity_calibration_crop.png"
    control_path = scratch / "parity_calibration_control.png"
    stego_path = scratch / "parity_calibration_full.png"
    for index in range(100):
        width, height = crop_sizes[index % len(crop_sizes)]
        left = int(crop_rng.integers(0, image_width - width + 1))
        top = int(crop_rng.integers(0, image_height - height + 1))
        crop = cover[top:top + height, left:left + width, :]
        Image.fromarray(crop).save(crop_path, format="PNG")
        _save_processed_control(crop_path, control_path)

        block_count = _read_quantized_coefficients(control_path).size
        random_parities = np.random.default_rng(20261105 + index).integers(
            0, 2, size=block_count, dtype=np.uint8
        )
        _embed_bits(control_path, stego_path, random_parities, every_block=True)
        result = detect_dct_parity(stego_path)
        _assert_parity_result(result)
        statistics.append(result["statistic"])
        p_values.append(result["p_value"])

    mean_statistic = float(np.mean(statistics))
    finite_p_values = np.asarray(p_values, dtype=np.float64)
    fractions = {
        threshold: float(np.mean(finite_p_values >= threshold))
        for threshold in (0.5, 0.1, 0.01)
    }
    assert 0.5 < mean_statistic < 1.6, mean_statistic
    assert fractions[0.01] >= 0.95, fractions[0.01]

    print("\n100-crop detect_dct_parity calibration")
    print(
        "Crops: 25 each at 256x192, 512x384, 800x600, and 960x640; "
        "independent random parity was written into every complete block."
    )
    print(f"mean_statistic={mean_statistic:.6f}")
    for threshold, fraction in fractions.items():
        print(f"fraction(p_value >= {threshold:g})={fraction:.2f}")


def run_tests() -> None:
    """Report controls, DCT cases, an LSB control, and calibration results."""
    negative_pair_values = np.repeat(
        np.asarray([-4, -3, -2, -1, 0, 1], dtype=np.int64), 5
    )
    negative_pair_result = _measure(negative_pair_values)
    assert negative_pair_result["pairs_used"] == 3
    assert negative_pair_result["pairs_skipped"] == 0
    assert negative_pair_result["dof"] == 2
    assert negative_pair_result["statistic"] == 0.0
    assert negative_pair_result["p_value"] == 1.0
    print(
        "Negative-pair check (-4/-3, -2/-1, 0/1): "
        f"pairs_used={negative_pair_result['pairs_used']}, "
        f"pairs_skipped={negative_pair_result['pairs_skipped']}, "
        f"statistic={negative_pair_result['statistic']:.12g}, "
        f"p_value={negative_pair_result['p_value']:.12g}"
    )

    project_root = Path(__file__).resolve().parent.parent
    cover_path = project_root / "images" / "small.png"
    assert cover_path.is_file(), f"Required test image is missing: {cover_path}"

    with tempfile.TemporaryDirectory() as temp_dir:
        scratch = Path(temp_dir)
        processed_path = scratch / "processed_control.png"
        _save_processed_control(cover_path, processed_path)

        short_dct_path = scratch / "dct_short.png"
        dct_hide(str(cover_path), str(short_dct_path), "short message")

        full_random_path = scratch / "dct_full_random_bytes.png"
        block_count = _read_quantized_coefficients(processed_path).size
        capacity_bits = block_count // REPETITION
        capacity_bytes = capacity_bits // 8
        assert capacity_bytes >= len(STOP_MARKER) + TAG_BYTES
        random_message = np.random.default_rng(20261105).bytes(
            capacity_bytes - len(STOP_MARKER) - TAG_BYTES
        )
        tag = make_tag(random_message, None)
        framed_payload = random_message + STOP_MARKER + tag.encode("ascii")
        payload_bits = np.unpackbits(np.frombuffer(framed_payload, dtype=np.uint8))
        fill_count = capacity_bits - payload_bits.size
        fill_bits = np.random.default_rng(20261106).integers(
            0, 2, size=fill_count, dtype=np.uint8
        )
        _embed_bits(
            cover_path,
            full_random_path,
            np.concatenate((payload_bits, fill_bits)),
        )

        plain_lsb_path = scratch / "plain_lsb.png"
        lsb_hide(str(cover_path), str(plain_lsb_path), "plain-mode LSB control")

        print("\nDCT detector test cases (temporary image outputs only)")
        print(
            "Rows are ordered to compare the processed clean control with the raw "
            "photo, then show DCT and LSB cases."
        )
        cases = (
            ("processed control (no message)", processed_path),
            ("RAW photo images/small.png", cover_path),
            ("DCT stego (short message)", short_dct_path),
            (
                "DCT stego (full random bytes; marker/tag + fill bits)",
                full_random_path,
            ),
            ("plain-mode LSB stego", plain_lsb_path),
        )
        for label, path in cases:
            result = detect_dct(path)
            _assert_result(result)
            _report(label, result)
            parity_result = detect_dct_parity(path)
            _assert_parity_result(parity_result)
            _report_parity(label + " (parity)", parity_result)

        print(
            f"Full-capacity test details: {capacity_bits} DCT-capacity bits; "
            f"{payload_bits.size} framed random-byte bits including the "
            f"{len(STOP_MARKER) + TAG_BYTES}-byte marker/tag overhead; "
            f"{fill_count} random fill bits complete the bit capacity."
        )
        _fully_embedded_calibration(scratch, cover_path)
        _fully_embedded_parity_calibration(scratch, cover_path)


if __name__ == "__main__":
    run_tests()
    print("test_detect_dct.py passed")
