"""Westfeld-Pfitzmann chi-square detection for LSB replacement.

For each pair of colour values (2i, 2i+1), both observed counts are compared
with the same expected count: half the pair total. The Pearson statistic sums
both cell terms, (O-E)^2/E, so it is twice the earlier one-cell form. A SMALL
statistic means pair counts are nearly equal and gives a p-value near 1, which
means embedding is likely. A LARGE statistic gives a p-value near 0 and looks
natural. Pairs with expected count below 5 are skipped; each result reports
how many pairs were skipped.

The selected degrees of freedom are ``pairs_used - 1``. On the 200-sample
calibration set, this choice gave p-value fractions closer to the uniform
reference than using ``pairs_used``.

This detects LSB replacement only. It does NOT detect DCT-domain embedding,
and it is weak on small payloads and on password-shuffled embedding.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import chi2


def _measure(values: np.ndarray) -> dict:
    """Measure even/odd histogram-pair differences for one flattened prefix."""
    histogram = np.bincount(np.asarray(values, dtype=np.uint8), minlength=256)
    even_counts = histogram[0::2].astype(np.float64)
    odd_counts = histogram[1::2].astype(np.float64)
    expected = (even_counts + odd_counts) / 2.0
    used = expected >= 5.0

    pairs_used = int(np.count_nonzero(used))
    pairs_skipped = int(used.size - pairs_used)
    if pairs_used:
        even_differences = even_counts[used] - expected[used]
        odd_differences = odd_counts[used] - expected[used]
        statistic = float(
            np.sum(
                even_differences**2 / expected[used]
                + odd_differences**2 / expected[used],
                dtype=np.float64,
            )
        )
    else:
        statistic = 0.0

    degrees_of_freedom = pairs_used - 1
    # A chi-square survival probability is undefined with fewer than two pairs.
    p_value = (
        float(chi2.sf(statistic, degrees_of_freedom))
        if degrees_of_freedom > 0
        else float("nan")
    )
    return {
        "statistic": statistic,
        "p_value": p_value,
        "pairs_used": pairs_used,
        "pairs_skipped": pairs_skipped,
        "degrees_of_freedom": degrees_of_freedom,
    }


def _read_values(path: str | Path) -> np.ndarray:
    """Load RGB values in row-major, channel-interleaved order."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).reshape(-1)


def detect(path: str | Path, threshold: float = 0.01) -> dict:
    """Return the chi-square LSB replacement result for an image.

    ``score`` is ``p_value * 100`` and is a heuristic, not a calibrated
    probability. The later ROC-style study will justify the final threshold
    value. A result with
    a p-value at least the threshold is reported as ``"embedding likely"``; otherwise it is reported as
    ``"no evidence of LSB embedding"``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    result = _measure(_read_values(path))
    p_value = result["p_value"]
    result["verdict"] = (
        "embedding likely"
        if p_value >= threshold
        else "no evidence of LSB embedding"
    )
    result["score"] = p_value * 100.0
    return result


def detect_prefix_curve(path: str | Path, steps: int = 20) -> dict:
    """Return chi-square results on growing sequential prefixes of RGB values.

    Prefixes use the flattened image's original sequential order, which can
    expose sequential LSB replacement. The returned ``curve`` contains one
    result dictionary per prefix, including its size and the number of used
    and skipped histogram pairs. Prefixes with fewer than 1000 values are
    omitted because they may not contain enough pairs for a valid p-value.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")

    values = _read_values(path)
    prefix_count = min(steps, len(values))
    prefix_sizes = np.linspace(1, len(values), num=prefix_count, dtype=np.int64)
    prefix_sizes = prefix_sizes[prefix_sizes >= 1000]
    curve = []
    for prefix_size in prefix_sizes:
        point = _measure(values[:int(prefix_size)])
        point["values_used"] = int(prefix_size)
        curve.append(point)
    return {"values_total": int(len(values)), "curve": curve}


def detect_sequential(
    path: str | Path,
    threshold: float = 0.01,
    steps: int = 20,
) -> dict:
    """Estimate the end of sequential LSB embedding from a prefix curve.

    This is a heuristic for sequential embedding only. It reports the initial
    run of prefixes whose p-values stay at or above ``threshold`` and the
    first following prefix below the threshold as an estimated payload end.
    A single p-value near the threshold can be a false miss, so the estimate
    should be treated cautiously.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    measured_curve = detect_prefix_curve(path, steps=steps)
    curve = measured_curve["curve"]
    p_values = [point["p_value"] for point in curve]
    first_below = next(
        (
            index
            for index, p_value in enumerate(p_values)
            if not np.isfinite(p_value) or p_value < threshold
        ),
        None,
    )

    if first_below is None:
        high_prefix_count = len(curve)
        end_point = None
    else:
        high_prefix_count = first_below
        end_point = curve[first_below] if high_prefix_count else None

    if high_prefix_count:
        longest_point = curve[high_prefix_count - 1]
        longest_values = longest_point["values_used"]
        longest_rate = longest_values / measured_curve["values_total"] * 100.0
    else:
        longest_values = 0
        longest_rate = 0.0

    if end_point is None:
        estimated_end_values = None
        estimated_end_rate = None
    else:
        estimated_end_values = end_point["values_used"]
        estimated_end_rate = (
            estimated_end_values / measured_curve["values_total"] * 100.0
        )

    return {
        "threshold": threshold,
        "values_total": measured_curve["values_total"],
        "curve": curve,
        "p_values": p_values,
        "high_prefix_count": high_prefix_count,
        "longest_high_prefix_values": longest_values,
        "longest_high_prefix_rate": longest_rate,
        "estimated_payload_end_values": estimated_end_values,
        "estimated_payload_end_rate": estimated_end_rate,
        "verdict": (
            "embedding likely"
            if high_prefix_count > 0
            else "no evidence of sequential LSB embedding"
        ),
    }
