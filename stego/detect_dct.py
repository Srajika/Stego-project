"""Targeted Pearson pair chi-square detector for this project's DCT embedding.

The detector assumes the embedding position and quality from :mod:`stego.dct`.
It is therefore specific to this project's DCT scheme, rather than a general
DCT steganography detector.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.stats import chi2

from .dct import (
    DCT_POSITION,
    EMBED_QUALITY,
    _blocks_from_plane,
    quantization_table,
)


def _read_quantized_coefficients(path: str | Path) -> np.ndarray:
    """Return this embedder's quantized coefficient from every complete block."""
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    ycbcr = np.asarray(Image.fromarray(rgb).convert("YCbCr"), dtype=np.uint8)
    blocks, _, _ = _blocks_from_plane(ycbcr[:, :, 0])
    if not len(blocks):
        return np.empty(0, dtype=np.int64)

    transformed = dctn(
        blocks.astype(np.float64) - 128.0,
        axes=(-2, -1),
        norm="ortho",
    )
    row, column = DCT_POSITION
    step = quantization_table(EMBED_QUALITY)[row, column]
    return np.rint(transformed[:, row, column] / step).astype(np.int64)


def _measure(values: np.ndarray) -> dict:
    """Measure both cells of every even/odd integer pair, including negatives."""
    integers = np.asarray(values, dtype=np.int64).reshape(-1)
    if integers.size:
        # floor division maps negative integers to the requested pair, e.g.
        # -2 and -1 both belong to (2 * -1, 2 * -1 + 1) == (-2, -1).
        pair_ids = np.floor_divide(integers, 2)
        first_pair = int(pair_ids.min())
        pair_offsets = pair_ids - first_pair
        pair_count = int(pair_ids.max() - first_pair + 1)
        pair_totals = np.bincount(pair_offsets, minlength=pair_count).astype(
            np.float64
        )
        even_counts = np.bincount(
            pair_offsets,
            weights=(np.remainder(integers, 2) == 0),
            minlength=pair_count,
        )
        odd_counts = pair_totals - even_counts
    else:
        pair_totals = np.empty(0, dtype=np.float64)
        even_counts = np.empty(0, dtype=np.float64)
        odd_counts = np.empty(0, dtype=np.float64)

    expected = pair_totals / 2.0
    used = expected >= 5.0
    pairs_used = int(np.count_nonzero(used))
    pairs_skipped = int(used.size - pairs_used)
    if pairs_used:
        even_difference = even_counts[used] - expected[used]
        odd_difference = odd_counts[used] - expected[used]
        statistic = float(
            np.sum(
                even_difference**2 / expected[used]
                + odd_difference**2 / expected[used],
                dtype=np.float64,
            )
        )
    else:
        statistic = 0.0

    dof = pairs_used - 1
    p_value = float(chi2.sf(statistic, dof)) if dof > 0 else float("nan")
    return {
        "statistic": statistic,
        "p_value": p_value,
        "pairs_used": pairs_used,
        "pairs_skipped": pairs_skipped,
        "dof": dof,
    }


def detect_dct(path: str | Path, threshold: float = 0.01) -> dict:
    """Detect this project's DCT embedding at ``(3, 2)`` and quality 75.

    The result uses the calibrated full-Pearson statistic from ``detect.py``:
    both cells in each pair contribute, pairs with expected count below five
    are skipped, and ``dof`` is ``pairs_used - 1``. ``blocks`` counts all
    complete 8x8 luminance blocks processed by the DCT embedder's block helper.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    coefficients = _read_quantized_coefficients(path)
    result = _measure(coefficients)
    result["blocks"] = int(coefficients.size)
    result["verdict"] = (
        "DCT embedding likely"
        if result["p_value"] >= threshold
        else "no evidence of DCT embedding"
    )
    return result


def detect_dct_parity(path: str | Path, threshold: float = 0.01) -> dict:
    """Detect this project's nearer-parity DCT embedding by coefficient parity.

    This is a targeted detector for this project's nearer-parity DCT
    embedding. False alarms are possible on noisy images whose natural parity
    is already balanced. A small one-degree-of-freedom statistic and a p-value
    near one indicate that even and odd quantized coefficients are balanced.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    coefficients = _read_quantized_coefficients(path)
    blocks = int(coefficients.size)
    if blocks:
        even_count = int(np.count_nonzero(np.remainder(coefficients, 2) == 0))
        odd_count = blocks - even_count
        even_fraction = even_count / blocks
        imbalance = float(even_count - odd_count)
        statistic = imbalance**2 / blocks
        p_value = float(chi2.sf(statistic, 1))
    else:
        even_fraction = float("nan")
        statistic = 0.0
        p_value = float("nan")

    return {
        "statistic": statistic,
        "p_value": p_value,
        "even_fraction": even_fraction,
        "blocks": blocks,
        "verdict": (
            "DCT embedding likely"
            if p_value >= threshold
            else "no evidence of DCT embedding"
        ),
    }
