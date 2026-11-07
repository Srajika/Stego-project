"""DCT steganography that stores each payload bit in three Y-channel blocks.

DCT block order is sequential, so a password authenticates the payload but does
not conceal the message or its embedding locations. The complete-block Y plane
is clamped to [MIN_Y, MAX_Y] for brightness headroom. The margin must be at
least the worst-case one-step pixel change, calculated as step * 0.2265 + 0.5
for rounding: 5.48 Y levels at embedding quality 50 and 2.99 at quality 75.
The margin is 6 based on the Phase 3b measurement: there were zero errors on all
15 usable dataset images and all 3 synthetic cases at qualities 50 and 75 for
each tested margin from 16 down to 2, and failures occurred at margin 0. The
synthetic cases used their capacity-limited payload prefix. This clamp causes a
global tone change: blacks are lifted to 6 and whites are lowered to 249.
"""

from math import pi, sqrt

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

from .common import bits_to_bytes, build_payload, bytes_to_bits, parse_payload
from .tamper import (
    STATUS_CHECKSUM_MISMATCH,
    STATUS_EMBEDDED,
    STATUS_NO_MESSAGE,
    STATUS_TAMPERED,
    derive_keys,
    make_tag,
    verify,
)

EMBED_QUALITY = 75
"""Quantization quality used for DCT embedding and reading, not JPEG output quality."""

# Zero-based row/column: away from DC (low frequency) and the high-frequency corner.
DCT_POSITION = (3, 2)
REPETITION = 3
MIN_Y = 6
MAX_Y = 249

JPEG_LUMINANCE_QUANT_TABLE = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.int32,
)


def quantization_table(quality: int = EMBED_QUALITY) -> np.ndarray:
    """Scale the standard JPEG luminance table for an embedding quality 1-100."""
    if not 1 <= quality <= 100:
        raise ValueError("Embedding quality must be between 1 and 100")
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    scaled = (JPEG_LUMINANCE_QUANT_TABLE * scale + 50) // 100
    return np.clip(scaled, 1, 255).astype(np.float64)


def capacity_bits(width: int, height: int) -> int:
    """Return payload bit capacity: total complete blocks split into 3 copy regions.

    Copy r of payload bit i uses block ``r * S + i``, where
    ``S = total_complete_blocks // REPETITION``.
    """
    return (width // 8) * (height // 8) // REPETITION


def _orthonormal_basis_row(frequency: int) -> np.ndarray:
    """Return one 8-sample orthonormal DCT-II basis vector."""
    positions = np.arange(8)
    scale = sqrt(1 / 8) if frequency == 0 else sqrt(2 / 8)
    return scale * np.cos(pi * (2 * positions + 1) * frequency / 16)


_POSITION_BASIS = np.outer(
    _orthonormal_basis_row(DCT_POSITION[0]),
    _orthonormal_basis_row(DCT_POSITION[1]),
)
MAX_PIXEL_CHANGE_PER_QUANT_STEP = float(
    quantization_table(EMBED_QUALITY)[DCT_POSITION] * np.max(np.abs(_POSITION_BASIS))
)
# At quality 75, position (3, 2) has step 11 and a maximum pixel change
# of about 2.492 Y levels for one coefficient step. The six-level margin also
# exceeds the worst-case quality-50 bound, including integer rounding.


def _read_luma(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load RGB and its YCbCr representation for DCT processing."""
    with Image.open(path) as source:
        rgb = np.array(source.convert("RGB"))
    ycbcr = np.array(Image.fromarray(rgb).convert("YCbCr"))
    return rgb, ycbcr


def _blocks_from_plane(plane: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return complete 8x8 blocks in row-major order and the covered dimensions."""
    block_rows = plane.shape[0] // 8
    block_columns = plane.shape[1] // 8
    covered_height = block_rows * 8
    covered_width = block_columns * 8
    blocks = (
        plane[:covered_height, :covered_width]
        .reshape(block_rows, 8, block_columns, 8)
        .transpose(0, 2, 1, 3)
        .reshape(block_rows * block_columns, 8, 8)
    )
    return blocks, covered_height, covered_width


def _block_bits(blocks: np.ndarray, quality: int, bit_count: int | None = None) -> str:
    """Recover bits from blocks, majority-voting groups of three."""
    capacity = len(blocks) // REPETITION
    if bit_count is None:
        bit_count = capacity
    if not 0 <= bit_count <= capacity:
        raise ValueError(f"Requested {bit_count} bits, but image holds {capacity} bits")
    if bit_count == 0:
        return ""
    # Copies are in three distant regions: block r*S+i carries copy r of bit i.
    indices = np.concatenate(
        [np.arange(bit_count) + repeat * capacity for repeat in range(REPETITION)]
    )
    table = quantization_table(quality)
    transformed = dctn(blocks[indices].astype(np.float64) - 128.0,
                       axes=(-2, -1), norm="ortho")
    row, column = DCT_POSITION
    quantized = np.rint(transformed[:, row, column] / table[row, column]).astype(np.int64)
    votes = (quantized & 1).reshape(REPETITION, bit_count).T
    majority = (votes.sum(axis=1) >= 2).astype(np.uint8)
    return "".join(str(int(bit)) for bit in majority)


def read_majority_bits(
    path: str,
    bit_count: int | None = None,
    quality: int = EMBED_QUALITY,
) -> str:
    """Return raw majority-voted bits, even when a later tag check would fail."""
    _, ycbcr = _read_luma(path)
    blocks, _, _ = _blocks_from_plane(ycbcr[:, :, 0])
    return _block_bits(blocks, quality, bit_count)


def hide(
    in_path: str,
    out_path: str,
    message: str,
    password: str | None = None,
    quality: int = EMBED_QUALITY,
) -> dict:
    """Embed a message as Y-channel DCT parity and save it as PNG.

    ``quality`` selects the embedding quantization table. It is not the quality
    of a later JPEG save; extraction must use the same embedding quality.
    DCT blocks are sequential, so a password authenticates but does not conceal
    the message. The complete-block Y channel is clamped to [MIN_Y, MAX_Y] to
    provide headroom against clipping; this lifts very dark and lowers very
    bright cover pixels. The six-level margin is guaranteed for embedding
    qualities 50 through 95. At the default setting, one coefficient step
    changes a pixel by at most about 2.492 Y levels before integer rounding.
    """
    if not 50 <= quality <= 95:
        raise ValueError(
            "Embedding quality must be between 50 and 95; "
            "the brightness-clamp margin is only guaranteed for that range"
        )
    table = quantization_table(quality)
    message_bytes = message.encode("utf-8")
    hmac_key = None
    if password is not None:
        _, hmac_key = derive_keys(password)
    tag = make_tag(message_bytes, password, hmac_key)
    payload_bits = bytes_to_bits(build_payload(message, tag))

    rgb, ycbcr = _read_luma(in_path)
    # Clamp luminance before embedding; preserve the original incomplete edge strips.
    y = np.clip(ycbcr[:, :, 0].astype(np.float64), MIN_Y, MAX_Y)
    blocks, covered_height, covered_width = _blocks_from_plane(y)
    available_bits = len(blocks) // REPETITION
    if len(payload_bits) > available_bits:
        raise ValueError(
            f"Message is too large for this image: needs {len(payload_bits)} payload bits; "
            f"DCT capacity is {available_bits} bits ({len(blocks)} complete blocks, "
            f"{REPETITION} blocks per bit)"
        )

    # Spread copies into separate regions to reduce correlated local failures.
    payload_size = len(payload_bits)
    selected_indices = np.concatenate(
        [np.arange(payload_size) + repeat * available_bits for repeat in range(REPETITION)]
    )
    changed_blocks = blocks[selected_indices].astype(np.float64) - 128.0
    transformed = dctn(changed_blocks, axes=(-2, -1), norm="ortho")
    row, column = DCT_POSITION
    step = table[row, column]
    original_quantized = transformed[:, row, column] / step
    nearest_integer = np.rint(original_quantized).astype(np.int64)
    target_bits = np.tile(np.fromiter((int(bit) for bit in payload_bits), dtype=np.int64), REPETITION)
    wrong_parity = (nearest_integer & 1) != target_bits
    lower = nearest_integer - 1
    upper = nearest_integer + 1
    nearer = np.where(
        np.abs(original_quantized - lower) <= np.abs(original_quantized - upper),
        lower,
        upper,
    )
    parity_quantized = np.where(wrong_parity, nearer, nearest_integer)
    transformed[:, row, column] = parity_quantized * step
    reconstructed = idctn(transformed, axes=(-2, -1), norm="ortho") + 128.0
    reconstructed = np.clip(np.rint(reconstructed), 0, 255).astype(np.uint8)

    modified_ycbcr = ycbcr.copy()
    block_rows = covered_height // 8
    block_columns = covered_width // 8
    output_blocks = blocks.copy()
    output_blocks[selected_indices] = reconstructed
    reconstructed_grid = output_blocks.reshape(block_rows, block_columns, 8, 8)
    reconstructed_plane = reconstructed_grid.transpose(0, 2, 1, 3).reshape(covered_height, covered_width)
    modified_ycbcr[:covered_height, :covered_width, 0] = reconstructed_plane

    # fromarray would infer RGB from the 3-channel shape; tag the bytes as YCbCr.
    output_rgb = np.array(
        Image.frombytes("YCbCr", (rgb.shape[1], rgb.shape[0]), modified_ycbcr.tobytes()).convert("RGB")
    )
    # Restore incomplete bottom/right edge strips exactly as they appeared in RGB.
    output_rgb[covered_height:, :, :] = rgb[covered_height:, :, :]
    output_rgb[:, covered_width:, :] = rgb[:, covered_width:, :]
    Image.fromarray(output_rgb).save(out_path, format="PNG")
    return {
        "status": STATUS_EMBEDDED,
        "path": out_path,
        "payload_bits": len(payload_bits),
        "capacity_bits": available_bits,
    }


def extract(
    path: str,
    password: str | None = None,
    quality: int = EMBED_QUALITY,
) -> dict:
    """Extract DCT payload using its embedding quality and verify its tag.

    ``quality`` is the embedding design quality used to quantize coefficients,
    not the quality used to save an image as JPEG later.
    """
    if not 50 <= quality <= 95:
        raise ValueError(
            "Embedding quality must be between 50 and 95; "
            "the brightness-clamp margin is only guaranteed for that range"
        )
    quantization_table(quality)  # Validate quality before image processing.
    payload_bits = read_majority_bits(path, quality=quality)
    complete_bits = payload_bits[:len(payload_bits) // 8 * 8]
    raw = bits_to_bytes(complete_bits)
    parsed = parse_payload(raw)
    if parsed is None:
        return {"status": STATUS_NO_MESSAGE, "message": None}

    message_bytes, tag = parsed
    try:
        message = message_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return {"status": STATUS_TAMPERED, "message": None, "tag": tag}
    status = verify(message, tag, password)
    if status in (STATUS_CHECKSUM_MISMATCH, STATUS_TAMPERED):
        return {"status": status, "message": message, "tag": tag}
    return {"status": status, "message": message, "tag": tag}
