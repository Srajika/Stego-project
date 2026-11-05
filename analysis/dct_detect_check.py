"""Check targeted DCT detection across payload sizes and content types."""

import random
import string
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

from stego.common import STOP_MARKER, TAG_BYTES, build_payload
from stego.dct import (
    DCT_POSITION,
    EMBED_QUALITY,
    MAX_Y,
    MIN_Y,
    REPETITION,
    _blocks_from_plane,
    capacity_bits,
    hide as dct_hide,
    quantization_table,
)
from stego.detect_dct import (
    _read_quantized_coefficients,
    detect_dct,
    detect_dct_parity,
)
from stego.lsb import hide as lsb_hide
from stego.tamper import make_tag


RATES = (10, 25, 50, 75, 90, 95, 100)
OVERHEAD_BYTES = len(STOP_MARKER) + TAG_BYTES
PRINTABLE = string.ascii_letters + string.digits + string.punctuation + " "
RANDOM_SEED = 20261105
THRESHOLD = 0.01


def _processed_rgb(rgb: np.ndarray) -> np.ndarray:
    """Apply the DCT embedder's clean, no-message brightness clamp."""
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


def _embed_payload_bytes(
    cover_path: Path, output_path: Path, payload: bytes
) -> None:
    """Write arbitrary framed bytes with the same 3-region DCT parity method."""
    with Image.open(cover_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    ycbcr = np.asarray(Image.fromarray(rgb).convert("YCbCr"), dtype=np.uint8)
    y = np.clip(ycbcr[:, :, 0].astype(np.float64), MIN_Y, MAX_Y)
    blocks, covered_height, covered_width = _blocks_from_plane(y)
    available_bits = len(blocks) // REPETITION
    payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    if payload_bits.size > available_bits:
        raise ValueError(
            f"Framed payload needs {payload_bits.size} bits; "
            f"DCT capacity is {available_bits} bits"
        )

    selected_indices = np.concatenate(
        [
            np.arange(payload_bits.size, dtype=np.int64) + repeat * available_bits
            for repeat in range(REPETITION)
        ]
    )
    changed_blocks = blocks[selected_indices].astype(np.float64) - 128.0
    transformed = dctn(changed_blocks, axes=(-2, -1), norm="ortho")
    row, column = DCT_POSITION
    step = quantization_table(EMBED_QUALITY)[row, column]
    scaled = transformed[:, row, column] / step
    nearest = np.rint(scaled).astype(np.int64)
    target_bits = np.tile(payload_bits.astype(np.int64), REPETITION)
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


def _payload_size_bytes(dct_capacity: int, requested_rate: int) -> int:
    """Floor to whole bytes; framing overhead is part of this target size."""
    payload_bytes = dct_capacity * requested_rate // 800
    if payload_bytes < OVERHEAD_BYTES:
        raise ValueError(
            f"{requested_rate}% is too small for the {OVERHEAD_BYTES}-byte framing"
        )
    return min(payload_bytes, dct_capacity // 8)


def _uniform_payload(payload_bytes: int, seed: int) -> bytes:
    message_size = payload_bytes - OVERHEAD_BYTES
    rng = np.random.default_rng(seed)
    message = rng.bytes(message_size)
    while STOP_MARKER in message:
        message = rng.bytes(message_size)
    tag = make_tag(message, None)
    return message + STOP_MARKER + tag.encode("ascii")


def _printable_payload(payload_bytes: int, seed: int) -> tuple[str, bytes]:
    message_size = payload_bytes - OVERHEAD_BYTES
    rng = random.Random(seed)
    message = "".join(rng.choices(PRINTABLE, k=message_size))
    while STOP_MARKER.decode("ascii") in message:
        message = "".join(rng.choices(PRINTABLE, k=message_size))
    tag = make_tag(message.encode("utf-8"), None)
    return message, build_payload(message, tag)


def _modified_block_fraction(control_path: Path, stego_path: Path) -> tuple[int, float]:
    control = _read_quantized_coefficients(control_path)
    stego = _read_quantized_coefficients(stego_path)
    if control.shape != stego.shape:
        raise AssertionError("Control and stego block counts differ")
    changed = int(np.count_nonzero(control != stego))
    return changed, changed / control.size if control.size else 0.0


def _print_detection_row(label: str, result: dict, pair_result: dict) -> None:
    print(
        f"{label:<24} | blocks={result['blocks']:>7} "
        f"| even_fraction={result['even_fraction']:.6f} "
        f"| pair_used={pair_result['pairs_used']:>3} "
        f"| pair_skip={pair_result['pairs_skipped']:>3} "
        f"| statistic={result['statistic']:.8g} "
        f"| p_value={result['p_value']:.8g} "
        f"| verdict={result['verdict']}"
    )


def _print_payload_row(
    payload_type: str,
    requested_rate: int,
    payload_bits_fraction: float,
    modified_block_fraction: float,
    one_bit_fraction: float,
    result: dict,
    pair_result: dict,
) -> None:
    print(
        f"{payload_type:<17} | {requested_rate:>3}% "
        f"| {modified_block_fraction * 100:>9.3f}% "
        f"| {payload_bits_fraction * 100:>9.3f}% "
        f"| {one_bit_fraction * 100:>8.3f}% "
        f"| {result['blocks']:>7} "
        f"| {pair_result['pairs_used']:>3} "
        f"| {pair_result['pairs_skipped']:>3} "
        f"| {result['even_fraction'] * 100:>8.3f}% "
        f"| {result['statistic']:>11.7g} "
        f"| {result['p_value']:>11.7g} "
        f"| {result['verdict']}"
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    # small.png has only 3458 DCT-capacity bits, below the 5520 bits needed
    # for its 10% row to contain the 69-byte marker and tag. Use the existing,
    # larger raw photo so every requested framed rate is representable.
    cover_path = project_root / "images" / "test.jpg"
    if not cover_path.is_file():
        raise FileNotFoundError(f"Required raw photo is missing: {cover_path}")
    with Image.open(cover_path) as image:
        width, height = image.size
    dct_capacity = capacity_bits(width, height)
    if dct_capacity * 10 // 800 < OVERHEAD_BYTES:
        raise AssertionError("Analysis cover cannot represent the 10% framed row")

    print(f"Cover: images/test.jpg ({width}x{height})")
    print(
        f"DCT capacity: {dct_capacity} bits "
        f"(floor(complete blocks / {REPETITION}))."
    )
    print(
        f"Payload framing: {len(STOP_MARKER)}-byte stop marker + {TAG_BYTES}-byte "
        f"tag = {OVERHEAD_BYTES} bytes, included in every payload rate."
    )
    print(
        "Rate definitions: requested payload bytes = floor(capacity_bits * "
        "requested_percent / 800), capped at floor(capacity_bits / 8); actual "
        "payload bits/capacity = framed payload bytes * 8 / capacity_bits. "
        "The main rate is changed quantized coefficients / complete blocks, "
        "measured against the processed control. One-bit fraction includes the "
        "message, marker, and tag."
    )
    print(
        f"10% feasibility: images/small.png has 3458 capacity bits (10% is "
        f"about 43 bytes), less than the {OVERHEAD_BYTES}-byte framing; "
        "images/test.jpg is used for the rate sweep."
    )
    print(f"Threshold: {THRESHOLD}; deterministic seeds start at {RANDOM_SEED}.")

    with tempfile.TemporaryDirectory() as temp_dir:
        scratch = Path(temp_dir)
        processed_path = scratch / "processed_control.png"
        raw_lsb_path = scratch / "plain_lsb.png"
        stego_path = scratch / "dct_stego.png"
        _save_processed_control(cover_path, processed_path)
        lsb_hide(str(cover_path), str(raw_lsb_path), "plain-mode LSB control")

        control_results = {
            "processed control": detect_dct_parity(processed_path, threshold=THRESHOLD),
            "RAW photo": detect_dct_parity(cover_path, threshold=THRESHOLD),
            "plain-mode LSB": detect_dct_parity(raw_lsb_path, threshold=THRESHOLD),
        }
        pair_control_results = {
            "processed control": detect_dct(processed_path, threshold=THRESHOLD),
            "RAW photo": detect_dct(cover_path, threshold=THRESHOLD),
            "plain-mode LSB": detect_dct(raw_lsb_path, threshold=THRESHOLD),
        }
        block_count = control_results["processed control"]["blocks"]
        print("\nControls (processed control and RAW photo are adjacent)")
        for label, result in control_results.items():
            _print_detection_row(label, result, pair_control_results[label])
        raw_stat_delta = (
            control_results["RAW photo"]["statistic"]
            - control_results["processed control"]["statistic"]
        )
        raw_p_delta = (
            control_results["RAW photo"]["p_value"]
            - control_results["processed control"]["p_value"]
        )
        raw_changed, raw_changed_fraction = _modified_block_fraction(
            processed_path, cover_path
        )
        print(
            "Clamp comparison (RAW minus processed control): "
            f"delta_statistic={raw_stat_delta:.8g}; delta_p_value={raw_p_delta:.8g}; "
            f"changed_coefficients={raw_changed}/{block_count} "
            f"({raw_changed_fraction * 100:.3f}% of blocks)."
        )

        processed_result = control_results["processed control"]
        print("\nPayload sweep (main rate is the modified-block fraction)")
        print(
            "Payload type      | target | rate: modified blocks | payload bits/capacity "
            "| one-bit fraction | blocks  | pair used | pair skip | even fraction "
            "| statistic     "
            "| p-value     | verdict"
        )
        print(
            f"{'processed control':<17} |   0% "
            f"| {0.0:>9.3f}% | {0.0:>9.3f}% | {'n/a':>8} "
            f"| {processed_result['blocks']:>7} "
            f"| {pair_control_results['processed control']['pairs_used']:>9} "
            f"| {pair_control_results['processed control']['pairs_skipped']:>9} "
            f"| {processed_result['even_fraction'] * 100:>8.3f}% "
            f"| {processed_result['statistic']:>11.7g} "
            f"| {processed_result['p_value']:>11.7g} "
            f"| {processed_result['verdict']}"
        )

        payload_results: list[tuple[str, int, dict, dict]] = []
        for rate in RATES:
            payload_bytes = _payload_size_bytes(dct_capacity, rate)
            payload_bits_fraction = payload_bytes * 8 / dct_capacity

            uniform_payload = _uniform_payload(
                payload_bytes, RANDOM_SEED + rate
            )
            _embed_payload_bytes(cover_path, stego_path, uniform_payload)
            uniform_result = detect_dct_parity(stego_path, threshold=THRESHOLD)
            uniform_pair_result = detect_dct(stego_path, threshold=THRESHOLD)
            uniform_changed, uniform_block_fraction = _modified_block_fraction(
                processed_path, stego_path
            )
            uniform_bits = np.unpackbits(
                np.frombuffer(uniform_payload, dtype=np.uint8)
            )
            _print_payload_row(
                "uniform bytes",
                rate,
                payload_bits_fraction,
                uniform_block_fraction,
                float(uniform_bits.mean()),
                uniform_result,
                uniform_pair_result,
            )
            payload_results.append(("uniform bytes", rate, uniform_result, uniform_pair_result))
            print(
                f"  modified coefficients: {uniform_changed}/{block_count}; "
                f"framed bytes={len(uniform_payload)}."
            )

            text_message, text_payload = _printable_payload(
                payload_bytes, RANDOM_SEED + 10_000 + rate
            )
            text_result_from_hide = dct_hide(
                str(cover_path), str(stego_path), text_message
            )
            if text_result_from_hide["payload_bits"] != len(text_payload) * 8:
                raise AssertionError("Text framing differs from dct.hide payload")
            text_result = detect_dct_parity(stego_path, threshold=THRESHOLD)
            text_pair_result = detect_dct(stego_path, threshold=THRESHOLD)
            text_changed, text_block_fraction = _modified_block_fraction(
                processed_path, stego_path
            )
            text_bits = np.unpackbits(np.frombuffer(text_payload, dtype=np.uint8))
            _print_payload_row(
                "printable text",
                rate,
                payload_bits_fraction,
                text_block_fraction,
                float(text_bits.mean()),
                text_result,
                text_pair_result,
            )
            payload_results.append(("printable text", rate, text_result, text_pair_result))
            print(
                f"  modified coefficients: {text_changed}/{block_count}; "
                f"framed bytes={len(text_payload)}."
            )

        surviving_pairs = np.asarray(
            [pair_result["pairs_used"] for _, _, _, pair_result in payload_results],
            dtype=np.float64,
        )
        skipped_pairs = np.asarray(
            [pair_result["pairs_skipped"] for _, _, _, pair_result in payload_results],
            dtype=np.float64,
        )
        print(
            "\nExpected-count rule for the retained pair detector: across the "
            f"{len(payload_results)} stego rows, pairs_used "
            f"mean={surviving_pairs.mean():.1f}, median="
            f"{np.median(surviving_pairs):.1f}, range="
            f"{int(surviving_pairs.min())}-{int(surviving_pairs.max())}; "
            f"pairs_skipped mean={skipped_pairs.mean():.1f}."
        )
        print(
            "Pair detector power note: each surviving pair contributes one "
            "contrast, with dof=pairs_used-1. Skipping low-expected-count pairs "
            "protects its Pearson approximation, but leaves few coefficient "
            "bins on this image. The main parity detector instead uses all "
            "blocks in a single dof=1 parity-balance test."
        )

        misses = [
            (payload_type, rate)
            for payload_type, rate, result, _ in payload_results
            if result["p_value"] < THRESHOLD
        ]
        control_flagged = processed_result["p_value"] >= THRESHOLD
        if control_flagged or misses:
            print(
                "Separation: not clean across all rates. "
                f"Processed-control verdict={processed_result['verdict']}; "
                f"DCT misses (p < {THRESHOLD})={misses}."
            )
        else:
            print(
                "Separation: on this cover, the processed control is below the "
                "threshold and every DCT payload row is detected."
            )


if __name__ == "__main__":
    main()
