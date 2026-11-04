"""Temporary DCT parity diagnostic; run with ``python -m analysis.dct_debug``."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

from stego.common import build_payload, bytes_to_bits
from stego.dct import (
    DCT_POSITION,
    EMBED_QUALITY,
    JPEG_LUMINANCE_QUANT_TABLE,
    REPETITION,
    _block_bits,
    _blocks_from_plane,
    capacity_bits,
    extract,
    hide,
    quantization_table,
    read_majority_bits,
    _read_luma,
)
from stego.tamper import make_tag


def _error_details(expected: str, observed: str) -> tuple[int, int | None]:
    """Count bit errors and return the first differing bit index, if any."""
    errors = [index for index, (left, right) in enumerate(zip(expected, observed)) if left != right]
    if len(expected) != len(observed):
        errors.extend(range(min(len(expected), len(observed)), max(len(expected), len(observed))))
    return len(errors), errors[0] if errors else None


def _encode_luma(plane: np.ndarray, payload_bits: str, round_clip: bool) -> np.ndarray:
    """Replicate the DCT embedding step for isolated in-memory variants."""
    blocks, covered_height, covered_width = _blocks_from_plane(plane.astype(np.float64))
    needed = len(payload_bits) * REPETITION
    changed = blocks[:needed].astype(np.float64) - 128.0
    coefficients = dctn(changed, axes=(-2, -1), norm="ortho")
    row, column = DCT_POSITION
    table = quantization_table(EMBED_QUALITY)
    step = table[row, column]
    unrounded_quantized = coefficients[:, row, column] / step
    nearest = np.rint(unrounded_quantized).astype(np.int64)
    target = np.repeat(np.fromiter((int(bit) for bit in payload_bits), dtype=np.int64), REPETITION)
    wrong = (nearest & 1) != target
    lower, upper = nearest - 1, nearest + 1
    nearer = np.where(
        np.abs(unrounded_quantized - lower) <= np.abs(unrounded_quantized - upper),
        lower,
        upper,
    )
    coefficients[:, row, column] = np.where(wrong, nearer, nearest) * step
    reconstructed = idctn(coefficients, axes=(-2, -1), norm="ortho") + 128.0
    if round_clip:
        reconstructed = np.clip(np.rint(reconstructed), 0, 255).astype(np.uint8)
    else:
        reconstructed = reconstructed.astype(np.float64)

    block_rows, block_columns = covered_height // 8, covered_width // 8
    output_blocks = blocks.copy()
    output_blocks[:needed] = reconstructed
    grid = output_blocks.reshape(block_rows, block_columns, 8, 8)
    output_plane = plane.astype(np.float64).copy()
    output_plane[:covered_height, :covered_width] = grid.transpose(0, 2, 1, 3).reshape(
        covered_height, covered_width
    )
    return output_plane


def _encode_luma_with_clip_details(
    plane: np.ndarray, payload_bits: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return float/clipped output planes plus per-block means and clip flags."""
    blocks, covered_height, covered_width = _blocks_from_plane(plane.astype(np.float64))
    needed = len(payload_bits) * REPETITION
    transformed = dctn(blocks[:needed].astype(np.float64) - 128.0,
                       axes=(-2, -1), norm="ortho")
    row, column = DCT_POSITION
    step = quantization_table(EMBED_QUALITY)[row, column]
    unrounded = transformed[:, row, column] / step
    nearest = np.rint(unrounded).astype(np.int64)
    targets = np.repeat(np.fromiter((int(bit) for bit in payload_bits), dtype=np.int64), REPETITION)
    lower, upper = nearest - 1, nearest + 1
    nearer = np.where(np.abs(unrounded - lower) <= np.abs(unrounded - upper), lower, upper)
    transformed[:, row, column] = np.where((nearest & 1) != targets, nearer, nearest) * step
    reconstructed_float = idctn(transformed, axes=(-2, -1), norm="ortho") + 128.0
    rounded = np.rint(reconstructed_float)
    clipped_flags = (rounded < 0) | (rounded > 255)
    clipped_to_zero = np.count_nonzero(rounded < 0, axis=(1, 2)).reshape(-1, REPETITION)
    clipped_to_255 = np.count_nonzero(rounded > 255, axis=(1, 2)).reshape(-1, REPETITION)
    reconstructed_clipped = np.clip(rounded, 0, 255).astype(np.uint8)

    block_rows, block_columns = covered_height // 8, covered_width // 8

    def restore_plane(reconstructed: np.ndarray) -> np.ndarray:
        output_blocks = blocks.copy()
        output_blocks[:needed] = reconstructed
        grid = output_blocks.reshape(block_rows, block_columns, 8, 8)
        output_plane = plane.astype(np.float64).copy()
        output_plane[:covered_height, :covered_width] = grid.transpose(0, 2, 1, 3).reshape(
            covered_height, covered_width
        )
        return output_plane

    float_plane = restore_plane(reconstructed_float)
    clipped_plane = restore_plane(reconstructed_clipped)
    block_means = blocks[:needed].mean(axis=(1, 2)).reshape(-1, REPETITION)
    clips_by_bit = clipped_flags.reshape(-1, REPETITION, 8, 8).any(axis=(2, 3))
    return float_plane, clipped_plane, block_means, clips_by_bit, clipped_to_zero, clipped_to_255


def _diagnose_real_photo(root: Path) -> None:
    """Measure raw payload errors and clipping for the failing long photo message."""
    photo = Path(__file__).resolve().parent.parent / "images" / "small.png"
    assert photo.is_file(), f"Required real photo is missing: {photo}"
    message = "long photo payload " + "x" * 128
    message_bytes = message.encode("utf-8")
    tag = make_tag(message_bytes, None)
    expected = bytes_to_bits(build_payload(message, tag))
    _, photo_ycbcr = _read_luma(str(photo))
    width, height = photo_ycbcr.shape[1], photo_ycbcr.shape[0]
    print("\nLong real-photo payload diagnosis")
    print(f"Cover: {photo} ({width} x {height})")
    print(f"Payload: {len(message_bytes)} message bytes + 5 marker bytes + 64 tag bytes")
    print(f"Payload size: {len(expected)} bits; DCT capacity: {capacity_bits(width, height)} bits")

    saved_path = root / "photo_long_stego.png"
    hide(str(photo), str(saved_path), message)
    result = extract(str(saved_path))
    observed = read_majority_bits(str(saved_path), bit_count=len(expected), quality=EMBED_QUALITY)
    errors, first = _error_details(expected, observed)
    print(f"Extraction status: {result['status']!r}")
    print(f"Extracted message: {result['message']!r}")
    print(f"Raw payload errors: {errors}/{len(expected)}; BER={errors / len(expected):.8f}; first={first}")

    original_y = photo_ycbcr[:, :, 0]
    float_y, clipped_y, means, clip_groups, clips_zero, clips_255 = _encode_luma_with_clip_details(
        original_y, expected
    )
    float_blocks, _, _ = _blocks_from_plane(float_y)
    no_clip_bits = _block_bits(float_blocks, EMBED_QUALITY, len(expected))
    no_clip_errors, _ = _error_details(expected, no_clip_bits)
    print(f"Clipping-disabled float diagnostic: {no_clip_errors}/{len(expected)} bit errors")

    output_blocks, _, _ = _blocks_from_plane(clipped_y)
    direct_clipped_bits = _block_bits(output_blocks, EMBED_QUALITY, len(expected))
    direct_clipped_errors, _ = _error_details(expected, direct_clipped_bits)
    print(f"Rounded/clipped Y before RGB conversion: {direct_clipped_errors}/{len(expected)} bit errors")

    error_indices = [index for index, (left, right) in enumerate(zip(expected, observed)) if left != right]
    if not error_indices:
        print("Erroneous-bit block details: none")
    else:
        print("Erroneous-bit block details (zero-based payload bit and row-major block indices):")
        for bit_index in error_indices:
            bit_means = means[bit_index]
            bit_clips = clip_groups[bit_index]
            zero_counts = clips_zero[bit_index]
            bright_counts = clips_255[bit_index]
            print(
                f"  bit {bit_index}: blocks {bit_index * REPETITION}-"
                f"{bit_index * REPETITION + REPETITION - 1}; "
                f"mean Y per block={[round(float(value), 2) for value in bit_means]}; "
                f"clipped per block={bit_clips.tolist()}; "
                f"pixels clipped to 0 per block={zero_counts.tolist()}; "
                f"pixels clipped to 255 per block={bright_counts.tolist()}; "
                f"any clipped={bool(bit_clips.any())}"
            )
        error_means = means[error_indices].mean(axis=1)
        dark_errors = int(np.count_nonzero(error_means < 32))
        bright_errors = int(np.count_nonzero(error_means > 223))
        clipped_errors = int(np.count_nonzero(clip_groups[error_indices].any(axis=1)))
        print(
            f"Error clustering: dark (<32 mean Y)={dark_errors}, "
            f"bright (>223 mean Y)={bright_errors}, "
            f"mid-range={len(error_indices) - dark_errors - bright_errors}; "
            f"erroneous bits with any clipped block={clipped_errors}/{len(error_indices)}"
        )

    # Diagnostic brightness compression only; original photo is never changed.
    with Image.open(photo) as source:
        source_rgb = np.array(source.convert("RGB"))
    compressed_rgb = np.rint(20 + source_rgb.astype(np.float64) * (215.0 / 255.0)).astype(np.uint8)
    compressed_cover = root / "photo_brightness_20_235.png"
    compressed_stego = root / "photo_brightness_20_235_stego.png"
    Image.fromarray(compressed_rgb).save(compressed_cover)
    hide(str(compressed_cover), str(compressed_stego), message)
    compressed_bits = read_majority_bits(
        str(compressed_stego), bit_count=len(expected), quality=EMBED_QUALITY
    )
    compressed_errors, _ = _error_details(expected, compressed_bits)
    compressed_result = extract(str(compressed_stego))
    print(
        "Brightness-compressed diagnostic (RGB 20-235): "
        f"{compressed_errors}/{len(expected)} bit errors; status={compressed_result['status']!r}"
    )


def _print_errors(label: str, expected: str, observed: str) -> None:
    """Print count and first differing index for one pipeline variant."""
    count, first = _error_details(expected, observed)
    print(f"{label}: bit errors={count}, first differing bit index={first}")


def main() -> None:
    """Run the baseline and isolate serialization, color, and rounding effects."""
    message = "plain DCT"
    tag = make_tag(message.encode("utf-8"), None)
    expected_bits = bytes_to_bits(build_payload(message, tag))

    height, width = 421, 419
    rows, columns = np.indices((height, width))
    gray = ((rows * 3 + columns * 5) % 100 + 70).astype(np.uint8)
    rgb = np.stack((gray, gray, gray), axis=2)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        cover_path = root / "cover.png"
        stego_path = root / "stego.png"
        Image.fromarray(rgb).save(cover_path)

        hide(str(cover_path), str(stego_path), message)
        result = extract(str(stego_path))
        print(f"Plain extraction result: {result!r}")
        print(f"Cover image size (width x height): {width} x {height}")
        print(f"DCT capacity: {capacity_bits(width, height)} bits")

        observed = read_majority_bits(
            str(stego_path), bit_count=len(expected_bits), quality=EMBED_QUALITY
        )
        print(f"Embedded payload bits: {len(expected_bits)}")
        _print_errors("Saved PNG baseline (raw majority bits)", expected_bits, observed)

        # (a) Read the inverse-DCT floats directly, before integer conversion,
        # color conversion, or PNG save.
        source = Image.fromarray(rgb).convert("YCbCr")
        ycbcr = np.array(source)
        float_y = _encode_luma(ycbcr[:, :, 0], expected_bits, round_clip=False)
        float_blocks, _, _ = _blocks_from_plane(float_y)
        _print_errors("(a) In-memory inverse-DCT float, no PNG save", expected_bits,
                      _block_bits(float_blocks, EMBED_QUALITY, len(expected_bits)))

        # Also inspect the normal in-memory image immediately before save. This
        # keeps the current RGB conversion path but removes PNG save/reload.
        pre_save_ycbcr = ycbcr.copy()
        pre_save_ycbcr[:, :, 0] = np.clip(np.rint(float_y), 0, 255).astype(np.uint8)
        pre_save_rgb = np.array(Image.fromarray(pre_save_ycbcr).convert("RGB"))
        pre_save_reloaded_y = np.array(Image.fromarray(pre_save_rgb).convert("YCbCr"))[:, :, 0]
        pre_save_blocks, _, _ = _blocks_from_plane(pre_save_reloaded_y)
        _print_errors("(a-control) In-memory current RGB path, PNG omitted", expected_bits,
                      _block_bits(pre_save_blocks, EMBED_QUALITY, len(expected_bits)))

        # (b) Use grayscale input/output, avoiding RGB <-> YCbCr conversion.
        gray_stego = _encode_luma(gray, expected_bits, round_clip=True)
        gray_path = root / "gray_stego.png"
        Image.fromarray(gray_stego.astype(np.uint8), mode="L").save(gray_path, format="PNG")
        gray_loaded = np.array(Image.open(gray_path).convert("L"))
        gray_blocks, _, _ = _blocks_from_plane(gray_loaded)
        _print_errors("(b) Grayscale PNG without YCbCr conversion", expected_bits,
                      _block_bits(gray_blocks, EMBED_QUALITY, len(expected_bits)))

        # (c) Remove the final rounding/clipping and decode directly from float Y.
        _print_errors("(c) Final rounding/clipping removed", expected_bits,
                      _block_bits(float_blocks, EMBED_QUALITY, len(expected_bits)))

        # Control: keep rounding/clipping, still before color conversion/save.
        rounded_y = _encode_luma(ycbcr[:, :, 0], expected_bits, round_clip=True)
        rounded_blocks, _, _ = _blocks_from_plane(rounded_y)
        _print_errors("Rounding/clipping retained before color conversion/save", expected_bits,
                      _block_bits(rounded_blocks, EMBED_QUALITY, len(expected_bits)))

        # Diagnostic-only candidate for the observed failure: explicitly tag the
        # modified byte array as YCbCr before Pillow converts it to RGB.
        modified = ycbcr.copy()
        modified[:, :, 0] = np.clip(np.rint(rounded_y), 0, 255).astype(np.uint8)
        tagged_rgb = Image.frombytes("YCbCr", (width, height), modified.tobytes()).convert("RGB")
        tagged_path = root / "explicit_ycbcr_mode.png"
        tagged_rgb.save(tagged_path, format="PNG")
        tagged_observed = read_majority_bits(
            str(tagged_path), bit_count=len(expected_bits), quality=EMBED_QUALITY
        )
        _print_errors("Diagnostic candidate: explicit YCbCr image mode", expected_bits,
                      tagged_observed)

        # (d) Hide and extract call the same constants and block/vote helpers.
        print(
            "(d) Constant/layout check: "
            f"position={DCT_POSITION}; quality={EMBED_QUALITY}; "
            f"quant_step={quantization_table(EMBED_QUALITY)[DCT_POSITION]}; "
            "level_shift=128 on both; block order=row-major on both; "
            f"repetition={REPETITION} with the same majority-vote helper"
        )
        print(f"JPEG base luminance table shape: {JPEG_LUMINANCE_QUANT_TABLE.shape}")
        _diagnose_real_photo(root)


if __name__ == "__main__":
    main()
