"""Dataset-wide embedding detection study (all generated files stay here)."""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "results" / "detection"
os.environ["MPLCONFIGDIR"] = str(OUTPUT_DIR / ".mplconfig")


def _remove_matplotlib_cache_at_exit() -> None:
    """Keep Matplotlib's runtime cache out of the results-only output folder."""
    cache_dir = OUTPUT_DIR / ".mplconfig"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


atexit.register(_remove_matplotlib_cache_at_exit)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL
import scipy
from PIL import Image
from scipy.stats import chi2

from analysis import detect_check as lsb_helpers
from analysis import dct_detect_check as dct_helpers
from stego.dct import (
    EMBED_QUALITY,
    MAX_Y,
    MIN_Y,
    REPETITION,
    capacity_bits as dct_capacity_bits,
)
from stego.detect import detect, detect_sequential
from stego.detect_dct import detect_dct_parity


MANIFEST_PATH = ROOT / "images" / "dataset" / "manifest.csv"
BASE_PAYLOAD_SEED = 20261108
BASE_CROP_SEED = 20261109
LSB_THRESHOLD = lsb_helpers.THRESHOLD
DCT_THRESHOLD = dct_helpers.THRESHOLD
ROC_THRESHOLDS = (0.001, 0.01, 0.05, 0.1)
FULL_UNIFORM_RATES = (10, 50, 90, 95, 100)
QUICK_UNIFORM_RATES = (50, 100)
PLAIN_EXTRA_RATES = (5, 25)
CROP_SIZES = ((256, 192), (512, 384))
FULL_CROPS_PER_SIZE = 5
QUICK_CROPS_PER_SIZE = 2
METHOD_ID = {"plain_lsb": 1, "password_lsb": 2, "dct": 3}
CONTENT_ID = {"uniform_random_bytes": 1, "printable_text": 2}
DETECTORS = (
    ("lsb_detect", "LSB detect()"),
    ("lsb_sequential_steps20", "LSB detect_sequential(steps=20)"),
    ("lsb_sequential_steps100", "LSB detect_sequential(steps=100)"),
    ("dct_parity", "DCT detect_dct_parity()"),
)

CSV_FIELDS = (
    "run_mode",
    "row_type",
    "scope",
    "image_index",
    "image",
    "category",
    "method",
    "payload_type",
    "requested_rate_pct",
    "actual_rate_pct",
    "capacity_bits",
    "payload_bytes",
    "payload_bits",
    "payload_one_fraction",
    "changed_blocks",
    "changed_block_fraction",
    "seed",
    "detector",
    "detector_threshold",
    "fired",
    "p_value",
    "p_value_copy_corrected",
    "statistic",
    "verdict",
    "steps",
    "values_total",
    "longest_high_prefix_rate_pct",
    "blocks",
    "even_fraction",
    "luminance_std",
    "flat_block_fraction",
    "distinct_rgb_values",
    "crop_width",
    "crop_height",
    "crop_index",
    "crop_seed",
    "elapsed_seconds",
    "detector_result_json",
)


def _read_usable_images() -> list[dict]:
    """Read usable manifest rows and attach stable zero-based indices."""
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row.get("usable_for_main_studies", "").strip().lower() == "true"
        ]
    rows.sort(key=lambda row: (row["category"].lower(), row["new_filename"].lower()))
    for index, row in enumerate(rows):
        row["image_index"] = index
        row["image_path"] = (
            ROOT / "images" / "dataset" / row["category"] / row["new_filename"]
        )
        if not row["image_path"].is_file():
            raise FileNotFoundError(f"Dataset image is missing: {row['image_path']}")
    return rows


def _quick_images(images: list[dict]) -> list[dict]:
    """Choose one image from each earliest sorted category for a quick check."""
    selected: list[dict] = []
    selected_categories: set[str] = set()
    for row in images:
        category = row["category"]
        if category not in selected_categories:
            selected.append(row)
            selected_categories.add(category)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        selected = images[:3]
    return selected


def _actual_rate(payload: bytes, capacity_bits: int) -> float:
    return len(payload) * 8 / capacity_bits * 100.0 if capacity_bits else 0.0


def _payload_seed(image_index: int, method: str, rate: int, payload_type: str) -> int:
    return (
        BASE_PAYLOAD_SEED
        + image_index * 1_000_000
        + METHOD_ID[method] * 100_000
        + rate * 100
        + CONTENT_ID[payload_type]
    )


def _crop_seed(image_index: int, size_id: int, crop_index: int) -> int:
    return BASE_CROP_SEED + image_index * 1_000_000 + size_id * 10_000 + crop_index


def _payload_one_fraction(payload: bytes) -> float:
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    return float(bits.mean()) if bits.size else float("nan")


def _describe_image(path: Path) -> dict[str, float | int]:
    """Describe Y luminance, flat complete blocks, and exact RGB color count."""
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)

    luminance = np.asarray(Image.fromarray(rgb).convert("YCbCr"), dtype=np.uint8)[:, :, 0]
    height, width = luminance.shape
    block_rows, block_columns = height // 8, width // 8
    covered_height, covered_width = block_rows * 8, block_columns * 8
    if block_rows and block_columns:
        blocks = (
            luminance[:covered_height, :covered_width]
            .reshape(block_rows, 8, block_columns, 8)
            .transpose(0, 2, 1, 3)
        )
        block_ranges = blocks.max(axis=(2, 3)) - blocks.min(axis=(2, 3))
        flat_block_fraction = float(np.mean(block_ranges <= 2))
    else:
        flat_block_fraction = float("nan")

    # Count exact 24-bit RGB values in bounded chunks to avoid a large full-image sort.
    seen_colors = np.zeros(1 << 24, dtype=np.bool_)
    flat_rgb = rgb.reshape(-1, 3)
    distinct_rgb_values = 0
    chunk_size = 1_000_000
    for start in range(0, len(flat_rgb), chunk_size):
        chunk = flat_rgb[start : start + chunk_size].astype(np.uint32)
        codes = (chunk[:, 0] << 16) | (chunk[:, 1] << 8) | chunk[:, 2]
        colors = np.unique(codes)
        unseen = ~seen_colors[colors]
        distinct_rgb_values += int(np.count_nonzero(unseen))
        seen_colors[colors] = True

    return {
        "luminance_std": float(np.std(luminance, dtype=np.float64)),
        "flat_block_fraction": flat_block_fraction,
        "distinct_rgb_values": distinct_rgb_values,
    }


def _threshold_for_detector(detector: str) -> float:
    return DCT_THRESHOLD if detector == "dct_parity" else LSB_THRESHOLD


def _run_detector(detector: str, path: Path, threshold: float) -> tuple[dict, float]:
    started = time.perf_counter()
    if detector == "lsb_detect":
        result = detect(path, threshold=threshold)
    elif detector == "lsb_sequential_steps20":
        result = detect_sequential(path, threshold=threshold, steps=20)
    elif detector == "lsb_sequential_steps100":
        result = detect_sequential(path, threshold=threshold, steps=100)
    elif detector == "dct_parity":
        result = detect_dct_parity(path, threshold=threshold)
    else:
        raise ValueError(f"Unknown detector: {detector}")
    return result, time.perf_counter() - started


def _p_value_for_result(detector: str, result: dict) -> float:
    if detector.startswith("lsb_sequential"):
        values = result.get("p_values", [])
        return float(values[0]) if values else float("nan")
    return float(result.get("p_value", float("nan")))


def _statistic_for_result(detector: str, result: dict) -> float:
    if detector.startswith("lsb_sequential"):
        curve = result.get("curve", [])
        return float(curve[0]["statistic"]) if curve else float("nan")
    return float(result.get("statistic", float("nan")))


def _fired(detector: str, result: dict, p_value: float, threshold: float) -> bool:
    if detector.startswith("lsb_sequential"):
        return result.get("verdict") == "embedding likely"
    return math.isfinite(p_value) and p_value >= threshold


def _row(
    *,
    run_mode: str,
    row_type: str,
    scope: str,
    image: dict,
    method: str,
    payload_type: str,
    requested_rate_pct: int,
    actual_rate_pct: float,
    capacity_bits: int,
    detector: str,
    result: dict,
    elapsed_seconds: float,
    payload_bytes: int = 0,
    payload_one_fraction: float | str = "",
    changed_blocks: int | str = "",
    changed_block_fraction: float | str = "",
    image_descriptors: dict[str, float | int] | None = None,
    seed: int | str = "",
    steps: int | str = "",
    crop_width: int | str = "",
    crop_height: int | str = "",
    crop_index: int | str = "",
    crop_seed: int | str = "",
) -> dict:
    threshold = _threshold_for_detector(detector)
    p_value = _p_value_for_result(detector, result)
    statistic = _statistic_for_result(detector, result)
    p_value_copy_corrected = (
        float(chi2.sf(statistic / REPETITION, 1))
        if detector == "dct_parity"
        else ""
    )
    descriptors = image_descriptors or {}
    curve = result.get("curve", []) if detector.startswith("lsb_sequential") else []
    longest_rate = result.get("longest_high_prefix_rate", "")
    return {
        "run_mode": run_mode,
        "row_type": row_type,
        "scope": scope,
        "image_index": image["image_index"],
        "image": image["new_filename"],
        "category": image["category"],
        "method": method,
        "payload_type": payload_type,
        "requested_rate_pct": requested_rate_pct,
        "actual_rate_pct": actual_rate_pct,
        "capacity_bits": capacity_bits,
        "payload_bytes": payload_bytes,
        "payload_bits": payload_bytes * 8,
        "payload_one_fraction": payload_one_fraction,
        "changed_blocks": changed_blocks,
        "changed_block_fraction": changed_block_fraction,
        "seed": seed,
        "detector": detector,
        "detector_threshold": threshold,
        "fired": _fired(detector, result, p_value, threshold),
        "p_value": p_value,
        "p_value_copy_corrected": p_value_copy_corrected,
        "statistic": statistic,
        "verdict": result.get("verdict", ""),
        "steps": steps,
        "values_total": result.get("values_total", ""),
        "longest_high_prefix_rate_pct": longest_rate,
        "blocks": result.get("blocks", ""),
        "even_fraction": result.get("even_fraction", ""),
        "luminance_std": descriptors.get("luminance_std", ""),
        "flat_block_fraction": descriptors.get("flat_block_fraction", ""),
        "distinct_rgb_values": descriptors.get("distinct_rgb_values", ""),
        "crop_width": crop_width,
        "crop_height": crop_height,
        "crop_index": crop_index,
        "crop_seed": crop_seed,
        "elapsed_seconds": elapsed_seconds,
        "detector_result_json": json.dumps(result, sort_keys=True, allow_nan=True),
    }


def _evaluate_path(
    *,
    run_mode: str,
    row_type: str,
    scope: str,
    image: dict,
    path: Path,
    method: str,
    payload_type: str,
    requested_rate_pct: int,
    actual_rate_pct: float,
    capacity_bits: int,
    detector_names: tuple[str, ...],
    payload_bytes: int = 0,
    payload_one_fraction: float | str = "",
    changed_blocks: int | str = "",
    changed_block_fraction: float | str = "",
    image_descriptors: dict[str, float | int] | None = None,
    seed: int | str = "",
    phase_times: dict | None = None,
    crop_width: int | str = "",
    crop_height: int | str = "",
    crop_index: int | str = "",
    crop_seed: int | str = "",
) -> list[dict]:
    records = []
    for detector in detector_names:
        threshold = _threshold_for_detector(detector)
        result, elapsed = _run_detector(detector, path, threshold)
        if phase_times is not None:
            phase_times["controls"] += elapsed if row_type in {
                "clean_cover", "processed_control"
            } else 0.0
        records.append(
            _row(
                run_mode=run_mode,
                row_type=row_type,
                scope=scope,
                image=image,
                method=method,
                payload_type=payload_type,
                requested_rate_pct=requested_rate_pct,
                actual_rate_pct=actual_rate_pct,
                capacity_bits=capacity_bits,
                detector=detector,
                result=result,
                elapsed_seconds=elapsed,
                payload_bytes=payload_bytes,
                payload_one_fraction=payload_one_fraction,
                changed_blocks=changed_blocks,
                changed_block_fraction=changed_block_fraction,
                image_descriptors=image_descriptors,
                seed=seed,
                steps=(20 if detector.endswith("steps20") else 100)
                if detector.startswith("lsb_sequential")
                else "",
                crop_width=crop_width,
                crop_height=crop_height,
                crop_index=crop_index,
                crop_seed=crop_seed,
            )
        )
    return records


def _make_lsb_payload(
    *, capacity_bits: int, rate: int, seed: int, password: str | None, payload_type: str
) -> bytes:
    target_bytes = lsb_helpers._target_payload_bytes(capacity_bits, rate)
    if payload_type == "uniform_random_bytes":
        payload = lsb_helpers._random_byte_payload(capacity_bits, rate, seed, password)
    else:
        message = lsb_helpers._message_for_rate(
            capacity_bits, rate, random.Random(seed)
        )
        payload = lsb_helpers._framed_text(message, password)
    if len(payload) != target_bytes:
        raise AssertionError("LSB helper produced a payload of unexpected size")
    return payload


def _make_dct_payload(
    *, capacity_bits: int, rate: int, seed: int, payload_type: str
) -> bytes:
    payload_bytes = dct_helpers._payload_size_bytes(capacity_bits, rate)
    if payload_type == "uniform_random_bytes":
        return dct_helpers._uniform_payload(payload_bytes, seed)
    _, payload = dct_helpers._printable_payload(payload_bytes, seed)
    return payload


def _grouped_rate_rows(records: list[dict], category: str | None = None) -> list[dict]:
    selected = [
        record
        for record in records
        if record["row_type"] in {"control", "processed_control", "stego"}
        and (category is None or record["category"] == category)
    ]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in selected:
        key = (
            record["method"],
            record["payload_type"],
            record["detector"],
            record["requested_rate_pct"],
        )
        groups[key].append(record)
    result = []
    for key, group in sorted(groups.items()):
        hits = sum(bool(item["fired"]) for item in group)
        copy_corrected_hits = (
            sum(
                math.isfinite(float(item["p_value_copy_corrected"]))
                and float(item["p_value_copy_corrected"]) >= DCT_THRESHOLD
                for item in group
            )
            if key[2] == "dct_parity"
            else None
        )
        result.append(
            {
                "method": key[0],
                "payload_type": key[1],
                "detector": key[2],
                "requested": key[3],
                "actual": statistics.mean(float(item["actual_rate_pct"]) for item in group),
                "hits": hits,
                "n": len(group),
                "copy_corrected_hits": copy_corrected_hits,
            }
        )
    return result


def _print_detection_rate_table(records: list[dict], category: str | None = None) -> None:
    title = "Overall detection rate k/n" if category is None else (
        f"Exploratory, small n - category: {category}"
    )
    print(f"\n{title} (actual rate is the mean across rows; rows retain individual rates)")
    print(
        "method        | payload type          | detector                         | requested | mean actual | "
        "original fires k/n | copy-corrected fires k/n"
    )
    print(
        "--------------|-----------------------|----------------------------------|-----------|-------------|"
        "------------------|--------------------------"
    )
    for item in _grouped_rate_rows(records, category):
        corrected = (
            f"{item['copy_corrected_hits']}/{item['n']}"
            if item["copy_corrected_hits"] is not None
            else "-"
        )
        print(
            f"{item['method']:<13} | {item['payload_type']:<21} | "
            f"{item['detector']:<32} | {item['requested']:>8}% | "
            f"{item['actual']:>10.4f}% | {item['hits']}/{item['n']:<15} | {corrected}"
        )


def _print_false_alarm_table(records: list[dict]) -> None:
    clean = [record for record in records if record["row_type"] == "clean_cover"]
    crops = [record for record in records if record["row_type"] == "clean_crop"]
    print("\nFalse alarms on raw clean covers and raw crops (fires k/n)")
    print(
        "detector                         | sample            | original fires k/n | "
        "copy-corrected fires k/n | original false-alarm rate"
    )
    print(
        "---------------------------------|-------------------|-------------------|"
        "-------------------------|--------------------------"
    )
    for detector, _ in DETECTORS:
        groups = [("raw clean covers", clean)]
        for width, height in CROP_SIZES:
            groups.append(
                (
                    f"raw crops {width}x{height}",
                    [
                        record
                        for record in crops
                        if record["crop_width"] == width
                        and record["crop_height"] == height
                    ],
                )
            )
        groups.append(("all raw crops", crops))
        for label, group in groups:
            values = [record for record in group if record["detector"] == detector]
            hits = sum(bool(record["fired"]) for record in values)
            rate = hits / len(values) if values else float("nan")
            corrected_hits = (
                sum(
                    math.isfinite(float(record["p_value_copy_corrected"]))
                    and float(record["p_value_copy_corrected"]) >= DCT_THRESHOLD
                    for record in values
                )
                if detector == "dct_parity"
                else None
            )
            corrected_text = (
                f"{corrected_hits}/{len(values)}"
                if corrected_hits is not None
                else "-"
            )
            print(
                f"{detector:<32} | {label:<17} | {hits:>3}/{len(values):<14} | "
                f"{corrected_text:<23} | {rate:.6f}"
            )


def _print_false_alarm_descriptor_table(records: list[dict]) -> None:
    """Compare clean-sample descriptors for false alarms and non-alarms."""
    clean = [
        record
        for record in records
        if record["row_type"] in {"clean_cover", "clean_crop"}
    ]
    scopes = (
        ("clean_cover", "raw covers"),
        ("clean_crop", "raw crops"),
    )
    print("\nFalse alarms versus clean-sample content descriptors")
    print(
        "detector                         | scope      | p version       | class     | "
        "n / total | mean luminance SD | mean flat-block fraction | mean distinct RGB"
    )
    print(
        "---------------------------------|------------|-----------------|-----------|"
        "----------|------------------|--------------------------|------------------"
    )
    for detector, _ in DETECTORS:
        versions = ("original", "copy-corrected") if detector == "dct_parity" else ("original",)
        for row_type, scope_label in scopes:
            for version in versions:
                values = [
                    record
                    for record in clean
                    if record["row_type"] == row_type and record["detector"] == detector
                ]
                def did_fire(record: dict) -> bool:
                    if version == "copy-corrected":
                        p_value = float(record["p_value_copy_corrected"])
                    else:
                        p_value = float(record["p_value"])
                    return math.isfinite(p_value) and p_value >= _threshold_for_detector(detector)

                total = len(values)
                for alarm_class, predicate in (
                    ("fired", did_fire),
                    ("not-fired", lambda record: not did_fire(record)),
                ):
                    group = [record for record in values if predicate(record)]
                    if not group:
                        means = (float("nan"),) * 3
                    else:
                        means = tuple(
                            statistics.mean(float(record[field]) for record in group)
                            for field in (
                                "luminance_std",
                                "flat_block_fraction",
                                "distinct_rgb_values",
                            )
                        )
                    print(
                        f"{detector:<32} | {scope_label:<10} | {version:<15} | "
                        f"{alarm_class:<9} | {len(group):>3}/{total:<5} | "
                        f"{means[0]:>16.5f} | {means[1]:>24.6f} | {means[2]:>16.2f}"
                    )


def _minimum_rates(records: list[dict]) -> tuple[list[dict], list[dict]]:
    stego = [record for record in records if record["row_type"] == "stego"]
    image_groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in stego:
        image_groups[
            (
                record["method"],
                record["payload_type"],
                record["detector"],
                record["image"],
                record["image_index"],
            )
        ].append(record)
    per_image = []
    for (method, payload, detector, image_name, image_index), group in sorted(image_groups.items()):
        fired_rates = [float(item["actual_rate_pct"]) for item in group if item["fired"]]
        copy_corrected_rates = (
            [
                float(item["actual_rate_pct"])
                for item in group
                if math.isfinite(float(item["p_value_copy_corrected"]))
                and float(item["p_value_copy_corrected"]) >= DCT_THRESHOLD
            ]
            if detector == "dct_parity"
            else []
        )
        per_image.append(
            {
                "method": method,
                "payload_type": payload,
                "detector": detector,
                "image": image_name,
                "image_index": image_index,
                "minimum": min(fired_rates) if fired_rates else None,
                "minimum_copy_corrected": (
                    min(copy_corrected_rates) if copy_corrected_rates else None
                ),
            }
        )
    aggregates: dict[tuple, list[dict]] = defaultdict(list)
    for record in per_image:
        aggregates[(record["method"], record["payload_type"], record["detector"])].append(record)
    summary = []
    for key, group in sorted(aggregates.items()):
        detected = [float(item["minimum"]) for item in group if item["minimum"] is not None]
        corrected_detected = [
            float(item["minimum_copy_corrected"])
            for item in group
            if item["minimum_copy_corrected"] is not None
        ]
        summary.append(
            {
                "method": key[0],
                "payload_type": key[1],
                "detector": key[2],
                "hits": len(detected),
                "n": len(group),
                "median": statistics.median(detected) if detected else None,
                "minimum": min(detected) if detected else None,
                "maximum": max(detected) if detected else None,
                "never": len(group) - len(detected),
                "copy_corrected_hits": len(corrected_detected),
                "copy_corrected_median": (
                    statistics.median(corrected_detected) if corrected_detected else None
                ),
                "copy_corrected_minimum": (
                    min(corrected_detected) if corrected_detected else None
                ),
                "copy_corrected_maximum": (
                    max(corrected_detected) if corrected_detected else None
                ),
                "copy_corrected_never": len(group) - len(corrected_detected),
            }
        )
    return per_image, summary


def _print_minimum_rates(records: list[dict]) -> None:
    per_image, summary = _minimum_rates(records)
    print("\nPer-image smallest tested actual embedding rate that fires (or never)")
    print(
        "method        | payload type          | detector                         | image             | "
        "original minimum | copy-corrected minimum"
    )
    print(
        "--------------|-----------------------|----------------------------------|-------------------|"
        "-----------------|------------------------"
    )
    for item in per_image:
        value = "never" if item["minimum"] is None else f"{item['minimum']:.4f}%"
        corrected_value = (
            "-"
            if item["minimum_copy_corrected"] is None
            else f"{item['minimum_copy_corrected']:.4f}%"
        )
        print(
            f"{item['method']:<13} | {item['payload_type']:<21} | "
            f"{item['detector']:<32} | {item['image']:<17} | "
            f"{value:<15} | {corrected_value}"
        )
    print("\nMinimum-rate summary by method, payload type, and detector")
    print(
        "method        | payload type          | detector                         | original fired k/n | "
        "median min rate | detected range | never | corrected fired k/n | corrected median/range | never"
    )
    print(
        "--------------|-----------------------|----------------------------------|-------------------|"
        "-----------------|----------------|-------|---------------------|------------------------|------"
    )
    for item in summary:
        median = "n/a" if item["median"] is None else f"{item['median']:.4f}%"
        value_range = (
            "n/a"
            if item["minimum"] is None
            else f"{item['minimum']:.4f}-{item['maximum']:.4f}%"
        )
        corrected_median = (
            "n/a"
            if item["copy_corrected_median"] is None
            else f"{item['copy_corrected_median']:.4f}%"
        )
        corrected_range = (
            "n/a"
            if item["copy_corrected_minimum"] is None
            else f"{item['copy_corrected_minimum']:.4f}-{item['copy_corrected_maximum']:.4f}%"
        )
        corrected_fires = (
            f"{item['copy_corrected_hits']}/{item['n']}"
            if item["detector"] == "dct_parity"
            else "-"
        )
        corrected_never = (
            item["copy_corrected_never"] if item["detector"] == "dct_parity" else "-"
        )
        print(
            f"{item['method']:<13} | {item['payload_type']:<21} | "
            f"{item['detector']:<32} | {item['hits']}/{item['n']:<15} | "
            f"{median:<15} | {value_range:<14} | {item['never']:<5} | "
            f"{corrected_fires:<19} | {corrected_median:<22} / {corrected_range:<15} | "
            f"{corrected_never}"
        )


def _print_roc_table(records: list[dict]) -> None:
    clean_by_detector: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["row_type"] == "clean_cover":
            clean_by_detector[record["detector"]].append(record)
    full_rate = [
        record
        for record in records
        if record["row_type"] == "stego" and record["requested_rate_pct"] == 100
    ]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in full_rate:
        groups[(record["method"], record["payload_type"], record["detector"])].append(record)
    print("\nROC-style threshold sweep (100% embedding detection vs raw clean-cover false alarms)")
    print("DCT has separate original and copy-corrected p-value results; non-DCT corrected cells are '-'.")
    print(
        "method        | payload type          | detector                         | p threshold | "
        "original detect k/n | corrected detect k/n | original FA k/n | corrected FA k/n"
    )
    print(
        "--------------|-----------------------|----------------------------------|-------------|"
        "--------------------|-----------------------|-----------------|-----------------"
    )
    for key, group in sorted(groups.items()):
        clean = clean_by_detector[key[2]]
        for threshold in ROC_THRESHOLDS:
            detections = sum(
                math.isfinite(float(item["p_value"]))
                and float(item["p_value"]) >= threshold
                for item in group
            )
            false_alarms = sum(
                math.isfinite(float(item["p_value"]))
                and float(item["p_value"]) >= threshold
                for item in clean
            )
            corrected_detections = (
                sum(
                    math.isfinite(float(item["p_value_copy_corrected"]))
                    and float(item["p_value_copy_corrected"]) >= threshold
                    for item in group
                )
                if key[2] == "dct_parity"
                else None
            )
            corrected_false_alarms = (
                sum(
                    math.isfinite(float(item["p_value_copy_corrected"]))
                    and float(item["p_value_copy_corrected"]) >= threshold
                    for item in clean
                )
                if key[2] == "dct_parity"
                else None
            )
            corrected_detection_text = (
                f"{corrected_detections}/{len(group)}"
                if corrected_detections is not None
                else "-"
            )
            corrected_false_alarm_text = (
                f"{corrected_false_alarms}/{len(clean)}"
                if corrected_false_alarms is not None
                else "-"
            )
            print(
                f"{key[0]:<13} | {key[1]:<21} | {key[2]:<32} | "
                f"{threshold:>10.3g} | {detections}/{len(group):<17} | "
                f"{corrected_detection_text:<21} | {false_alarms}/{len(clean):<13} | "
                f"{corrected_false_alarm_text}"
            )


def _print_payload_one_bit_summary(records: list[dict]) -> None:
    full_rate = [
        record
        for record in records
        if record["row_type"] == "stego" and record["requested_rate_pct"] == 100
    ]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in full_rate:
        groups[(record["method"], record["payload_type"], record["detector"])].append(record)
    print("\n100% target detection versus payload one-bit fraction")
    print(
        "method        | payload type          | detector                         | n | "
        "one-bit fraction mean (range) | original fires k/n | copy-corrected fires k/n"
    )
    print(
        "--------------|-----------------------|----------------------------------|---|"
        "------------------------------|--------------------|--------------------------"
    )
    for key, group in sorted(groups.items()):
        fractions = [float(item["payload_one_fraction"]) for item in group]
        mean_fraction = statistics.mean(fractions)
        fraction_range = f"{min(fractions):.6f}-{max(fractions):.6f}"
        original_hits = sum(bool(item["fired"]) for item in group)
        corrected_hits = (
            sum(
                math.isfinite(float(item["p_value_copy_corrected"]))
                and float(item["p_value_copy_corrected"]) >= DCT_THRESHOLD
                for item in group
            )
            if key[2] == "dct_parity"
            else None
        )
        corrected_text = (
            f"{corrected_hits}/{len(group)}"
            if corrected_hits is not None
            else "-"
        )
        print(
            f"{key[0]:<13} | {key[1]:<21} | {key[2]:<32} | {len(group):>1} | "
            f"{mean_fraction:.6f} ({fraction_range}) | "
            f"{original_hits}/{len(group):<16} | {corrected_text}"
        )


def _print_raw_measurements(records: list[dict]) -> None:
    print("\nAll per-row measurements (actual rate shown on every row; fires=True is a detector positive)")
    print(
        "row type         | image             | method        | payload type          | requested | actual rate | "
        "detector                         | fires | original p | corrected p"
    )
    print(
        "-----------------|-------------------|---------------|-----------------------|-----------|-------------|"
        "----------------------------------|-------|------------|-------------"
    )
    for item in records:
        p_value = float(item["p_value"]) if item["p_value"] != "" else float("nan")
        p_text = f"{p_value:.8g}" if math.isfinite(p_value) else "nan"
        corrected = item["p_value_copy_corrected"]
        corrected_text = (
            f"{float(corrected):.8g}"
            if corrected != "" and math.isfinite(float(corrected))
            else "-"
        )
        actual_text = (
            f"{float(item['actual_rate_pct']):.4f}%"
            if item["actual_rate_pct"] != ""
            else "unavailable"
        )
        print(
            f"{item['row_type']:<16} | {item['image']:<17} | {item['method']:<13} | "
            f"{item['payload_type']:<21} | {item['requested_rate_pct']:>8}% | "
            f"{actual_text:>10} | {item['detector']:<32} | "
            f"{str(bool(item['fired'])):<5} | {p_text:>10} | {corrected_text:>11}"
        )


def _save_plots(records: list[dict]) -> None:
    rate_records = [
        record
        for record in records
        if record["row_type"] in {"control", "processed_control", "stego"}
    ]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in rate_records:
        groups[
            (
                record["method"],
                record["payload_type"],
                record["detector"],
            )
        ].append(record)

    figure, axis = plt.subplots(figsize=(12, 7))
    for key, group in sorted(groups.items()):
        by_rate: dict[int, list[dict]] = defaultdict(list)
        for record in group:
            by_rate[int(record["requested_rate_pct"])].append(record)
        versions = ("original", "copy-corrected") if key[2] == "dct_parity" else ("original",)
        for version in versions:
            points = []
            for _, rows in sorted(by_rate.items()):
                if version == "copy-corrected":
                    fires = sum(
                        math.isfinite(float(item["p_value_copy_corrected"]))
                        and float(item["p_value_copy_corrected"]) >= DCT_THRESHOLD
                        for item in rows
                    )
                else:
                    fires = sum(bool(item["fired"]) for item in rows)
                points.append(
                    (
                        statistics.mean(float(item["actual_rate_pct"]) for item in rows),
                        fires / len(rows),
                    )
                )
            label = f"{key[0]} / {key[1]} / {key[2]}"
            if version == "copy-corrected":
                label += " / copy-corrected"
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                linestyle="--" if version == "copy-corrected" else "-",
                label=label,
            )
    axis.set(xlabel="Actual payload rate (% of method capacity)", ylabel="Detection rate")
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlim(left=0)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7, ncol=2, loc="best")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "detection_rate_vs_embedding_rate.png", dpi=160)
    plt.close(figure)

    clean = [record for record in records if record["row_type"] == "clean_cover"]
    figure, axis = plt.subplots(figsize=(10, 6))
    for detector, label in DETECTORS:
        values = [
            float(record["p_value"])
            for record in clean
            if record["detector"] == detector
            and math.isfinite(float(record["p_value"]))
        ]
        axis.hist(values, bins=np.linspace(0, 1, 21), alpha=0.45, label=label)
    corrected_values = [
        float(record["p_value_copy_corrected"])
        for record in clean
        if record["detector"] == "dct_parity"
        and math.isfinite(float(record["p_value_copy_corrected"]))
    ]
    axis.hist(
        corrected_values,
        bins=np.linspace(0, 1, 21),
        alpha=0.45,
        histtype="step",
        linewidth=2,
        label="DCT parity (copy-corrected)",
    )
    axis.set(xlabel="p-value on raw clean cover", ylabel="Image count", xlim=(0, 1))
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "clean_cover_pvalue_distributions.png", dpi=160)
    plt.close(figure)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _write_run_info(
    *,
    run_mode: str,
    started_at: str,
    finished_at: str,
    selected: list[dict],
    all_images: list[dict],
    crop_count: int,
    total_runtime: float,
    estimate_seconds: float,
) -> None:
    lines = [
        "Phase 5B dataset detection study",
        f"Started: {started_at}",
        f"Finished: {finished_at}",
        f"Mode: {run_mode}",
        f"Git commit hash: {_git_commit()}",
        f"Dataset manifest: {MANIFEST_PATH.relative_to(ROOT).as_posix()}",
        f"Images selected: {len(selected)} of {len(all_images)} usable images",
        "Selected images: " + ", ".join(row["new_filename"] for row in selected),
        f"Total runtime seconds: {total_runtime:.6f}",
        (
            f"Estimated full-run seconds from quick sample: {estimate_seconds:.3f}"
            if run_mode == "quick"
            else f"Measured full-run seconds: {total_runtime:.3f}"
        ),
        f"Python: {sys.version.split()[0]}",
        f"NumPy: {np.__version__}",
        f"Pillow: {PIL.__version__}",
        f"SciPy: {scipy.__version__}",
        f"Matplotlib: {matplotlib.__version__}",
        f"DCT clamp margin: MIN_Y={MIN_Y}, MAX_Y={MAX_Y}",
        f"LSB detector threshold: {LSB_THRESHOLD}",
        f"DCT parity detector threshold: {DCT_THRESHOLD}",
        f"DCT copy-corrected p-value: chi2.sf(statistic / REPETITION, 1), with REPETITION={REPETITION}; this counts the three repeated copies of each payload bit once.",
        "Original detector p-values and verdicts are retained alongside copy-corrected DCT values.",
        "CSV luminance descriptors: standard deviation of Y from RGB-to-YCbCr; flat-block fraction is complete 8x8 blocks with Y range <= 2; distinct RGB is the exact number of unique 24-bit RGB colors.",
        "ROC p-value thresholds: " + ", ".join(map(str, ROC_THRESHOLDS)),
        f"Embedding quality for DCT: {EMBED_QUALITY} (project EMBED_QUALITY)",
        "DCT coefficient position: " + str(dct_helpers.DCT_POSITION),
        "Payload framing: marker and tag included in target capacity rate; actual rate = framed payload bits / method capacity bits * 100.",
        f"Payload base seed: {BASE_PAYLOAD_SEED}",
        "Payload seed formula: base + sorted_manifest_image_index*1000000 + method_id*100000 + requested_rate*100 + content_id; method_id plain_lsb=1, password_lsb=2, dct=3; content_id uniform_random_bytes=1, printable_text=2.",
        "Exact per-image/per-rate payload seeds are recorded in the CSV seed column.",
        f"Crop base seed: {BASE_CROP_SEED}",
        "Crop seed formula: base + sorted_manifest_image_index*1000000 + size_id*10000 + zero_based_crop_index; size_id 1=256x192, 2=512x384.",
        f"Crops per image per size in this run: {crop_count}",
        "Crop coordinates are sampled by random.Random(crop_seed) from the full cover; pixel values are saved unchanged as RGB PNG for detector input.",
        "Crop requests larger than a cover are recorded as unavailable_crop rows in the CSV and output; no detector result or descriptor exists for those requests.",
        "DCT requested-rate rows below the fixed framing overhead are recorded as unavailable rows in the CSV and output and excluded from detection-rate denominators.",
        "Uniform-byte and printable-text payload generation/embedding reuse helpers from analysis.detect_check and analysis.dct_detect_check.",
        "Clean-cover false alarms use raw covers/crops. DCT processed controls are reported separately and are not used for the clean-cover false-alarm rate.",
        "Quick duration estimate scales measured per-phase means by full-vs-quick payload row counts and crop count; it is a linear estimate, not a full-run measurement.",
    ]
    (OUTPUT_DIR / "run_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_dct_rows(records: list[dict]) -> None:
    rows = [
        record
        for record in records
        if record["detector"] == "dct_parity"
        and (
            record["row_type"] in {"clean_cover", "processed_control"}
            or (
                record["row_type"] == "stego"
                and record["payload_type"] == "uniform_random_bytes"
                and record["requested_rate_pct"] in {50, 100}
            )
        )
    ]
    print(
        "\nDCT parity diagnostic rows: raw cover, processed control, uniform-byte "
        "stego at 50% and 100%"
    )
    print(
        "image             | row type           | target | payload bits/capacity | "
        "changed blocks / fraction | blocks | even fraction | statistic | original p | corrected p | verdict"
    )
    print(
        "-------------------|--------------------|--------|-----------------------|"
        "---------------------------|--------|---------------|-----------|------------|-------------|--------"
    )
    for item in rows:
        p_value = float(item["p_value"])
        p_text = f"{p_value:.8g}" if math.isfinite(p_value) else "nan"
        corrected_value = float(item["p_value_copy_corrected"])
        corrected_text = (
            f"{corrected_value:.8g}" if math.isfinite(corrected_value) else "nan"
        )
        even_fraction = float(item["even_fraction"])
        even_text = f"{even_fraction:.8f}" if math.isfinite(even_fraction) else "nan"
        changed_fraction = item["changed_block_fraction"]
        changed_text = (
            f"{item['changed_blocks']}/{item['blocks']} ({float(changed_fraction):.6f})"
            if changed_fraction != ""
            else "n/a"
        )
        row_label = {
            "clean_cover": "RAW cover",
            "processed_control": "PROCESSED control",
            "stego": "DCT stego",
        }[item["row_type"]]
        print(
            f"{item['image']:<18} | {row_label:<18} | "
            f"{item['requested_rate_pct']:>5}% | "
            f"{int(item['payload_bits'])}/{item['capacity_bits']:<14} | "
            f"{changed_text:<25} | {item['blocks']:>6} | {even_text:>13} | "
            f"{float(item['statistic']):>9.6g} | {p_text:>10} | "
            f"{corrected_text:>11} | {item['verdict']}"
        )


def _print_crop_rows(records: list[dict]) -> None:
    crop_groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for item in records:
        if item["row_type"] != "clean_crop":
            continue
        key = (
            item["image_index"],
            item["image"],
            item["crop_width"],
            item["crop_height"],
            item["crop_index"],
        )
        crop_groups[key][item["detector"]] = item

    print(
        f"\nAll {len(crop_groups)} available raw small-crop samples "
        "(false alarms are marked in each detector cell)"
    )
    print(
        "image             | crop size | index | detect() p / verdict | "
        "sequential 20 p / verdict | sequential 100 p / verdict | "
        "DCT parity original p / corrected p / verdict"
    )
    print(
        "-------------------|-----------|-------|----------------------|"
        "--------------------------|----------------------------|------------------------------------------"
    )
    detector_order = (
        "lsb_detect",
        "lsb_sequential_steps20",
        "lsb_sequential_steps100",
        "dct_parity",
    )
    for key, detector_rows in sorted(crop_groups.items()):
        image_index, image_name, width, height, crop_index = key
        cells = []
        for detector in detector_order:
            item = detector_rows[detector]
            p_value = float(item["p_value"])
            p_text = f"{p_value:.5g}" if math.isfinite(p_value) else "nan"
            false_alarm = " [FALSE ALARM]" if bool(item["fired"]) else ""
            if detector == "dct_parity":
                corrected = float(item["p_value_copy_corrected"])
                corrected_text = f"{corrected:.5g}" if math.isfinite(corrected) else "nan"
                cells.append(
                    f"p={p_text}; pc={corrected_text}; {item['verdict']}{false_alarm}"
                )
                continue
            cells.append(f"p={p_text}; {item['verdict']}{false_alarm}")
        print(
            f"{image_name:<18} | {width}x{height:<5} | {crop_index:>5} | "
            + " | ".join(cells)
        )


def _print_unavailable_rows(records: list[dict]) -> None:
    unavailable = [
        record
        for record in records
        if record["row_type"] in {"unavailable", "unavailable_crop"}
    ]
    if not unavailable:
        return
    print("\nRequested rows unavailable due to framing overhead or crop dimensions")
    print(
        "image             | row type        | method    | payload type          | target/sample | "
        "capacity | reason"
    )
    print(
        "-------------------|-----------------|-----------|-----------------------|---------------|"
        "----------|------------------------------"
    )
    for record in unavailable:
        target = (
            f"{record['requested_rate_pct']}%"
            if record["row_type"] == "unavailable"
            else f"{record['crop_width']}x{record['crop_height']}#{record['crop_index']}"
        )
        capacity = record["capacity_bits"] if record["capacity_bits"] != "" else "n/a"
        print(
            f"{record['image']:<18} | {record['row_type']:<15} | "
            f"{record['method']:<9} | {record['payload_type']:<21} | {target:<13} | "
            f"{capacity:>8} | {record['verdict']}"
        )


def _estimated_full_seconds(phase_totals: list[dict], image_count: int) -> tuple[float, dict]:
    if not phase_totals:
        return 0.0, {}
    means = {
        name: statistics.mean(item[name] for item in phase_totals)
        for name in ("controls", "plain_lsb", "password_lsb", "dct", "crops")
    }
    per_image = (
        means["controls"]
        + means["plain_lsb"] * (7 / 4)
        + means["password_lsb"] * (5 / 2)
        + means["dct"] * 2
        + means["crops"] * (FULL_CROPS_PER_SIZE / QUICK_CROPS_PER_SIZE)
    )
    return per_image * image_count, means


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="run three images, reduced rates and two crops per size"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_images = _read_usable_images()
    selected = _quick_images(all_images) if args.quick else all_images
    run_mode = "quick" if args.quick else "full"
    uniform_rates = QUICK_UNIFORM_RATES if args.quick else FULL_UNIFORM_RATES
    crop_count = QUICK_CROPS_PER_SIZE if args.quick else FULL_CROPS_PER_SIZE
    crops_per_size = crop_count
    plain_rates = tuple(sorted(set(uniform_rates + PLAIN_EXTRA_RATES)))
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    study_started = time.perf_counter()
    phase_totals: list[dict] = []
    records: list[dict] = []
    crop_size_id = {CROP_SIZES[0]: 1, CROP_SIZES[1]: 2}
    detector_fields = tuple(item[0] for item in DETECTORS)
    crop_detector_fields = detector_fields
    work_root = OUTPUT_DIR
    csv_path = OUTPUT_DIR / "detection_results.csv"

    print(f"Phase 5B detection study mode: {run_mode}")
    print(f"Usable images selected: {len(selected)} of {len(all_images)}")
    print(
        "Uniform-byte rates: "
        + ", ".join(f"{rate}%" for rate in uniform_rates)
        + "; plain-mode LSB additionally: 5%, 25%; printable text: 100% only."
    )
    print("Every payload target includes the 5-byte marker and 64-byte tag.")
    print(
        f"False-alarm crop design: {crops_per_size} fixed-seed crops per image "
        f"at each of {CROP_SIZES[0][0]}x{CROP_SIZES[0][1]} and "
        f"{CROP_SIZES[1][0]}x{CROP_SIZES[1][1]}."
    )
    print(
        f"Thresholds: LSB={LSB_THRESHOLD}; DCT parity={DCT_THRESHOLD}; "
        f"ROC sweep={ROC_THRESHOLDS}; DCT clamp MIN_Y={MIN_Y}, MAX_Y={MAX_Y}."
    )
    print(
        f"DCT copy-corrected p-value: chi2.sf(statistic / {REPETITION}, 1); "
        "the correction counts the repeated copies of each payload bit once."
    )
    print(
        "Reused helpers: detect_check._target_payload_bytes, _random_byte_payload, "
        "_message_for_rate, _framed_text, _embed_payload_bits; "
        "dct_detect_check._save_processed_control, _payload_size_bytes, "
        "_uniform_payload, _printable_payload, _embed_payload_bytes."
    )

    with csv_path.open("w", newline="", encoding="utf-8") as csv_stream:
        writer = csv.DictWriter(csv_stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        with tempfile.TemporaryDirectory(prefix=".detection-work-", dir=work_root) as temp_name:
            work_dir = Path(temp_name)
            for selected_position, image in enumerate(selected, start=1):
                image_record_start = len(records)
                image_started = time.perf_counter()
                phases = {
                    "controls": 0.0,
                    "plain_lsb": 0.0,
                    "password_lsb": 0.0,
                    "dct": 0.0,
                    "crops": 0.0,
                }
                source_path = image["image_path"]
                with Image.open(source_path) as source:
                    width, height = source.size
                raw_descriptors = _describe_image(source_path)
                lsb_capacity = width * height * 3
                dct_capacity = dct_capacity_bits(width, height)
                image_label = f"{image['category']}/{image['new_filename']}"
                print(f"\n[{selected_position}/{len(selected)}] {image_label}", flush=True)

                clean_records = []
                for detector in detector_fields:
                    detector_capacity = dct_capacity if detector == "dct_parity" else lsb_capacity
                    clean_records.extend(
                        _evaluate_path(
                            run_mode=run_mode,
                            row_type="clean_cover",
                            scope="raw_full_cover",
                            image=image,
                            path=source_path,
                            method="raw_cover",
                            payload_type="none",
                            requested_rate_pct=0,
                            actual_rate_pct=0.0,
                            capacity_bits=detector_capacity,
                            detector_names=(detector,),
                            image_descriptors=raw_descriptors,
                            phase_times=phases,
                        )
                    )
                records.extend(clean_records)

                # Zero-rate LSB rows are the same raw cover control measured above.
                for method in ("plain_lsb", "password_lsb"):
                    for detector in detector_fields[:3]:
                        source_result = next(
                            row for row in clean_records if row["detector"] == detector
                        )
                        control = dict(source_result)
                        control.update(
                            {
                                "row_type": "control",
                                "scope": "raw_full_cover",
                                "method": method,
                                "payload_type": "uniform_random_bytes",
                                "requested_rate_pct": 0,
                                "actual_rate_pct": 0.0,
                                "capacity_bits": lsb_capacity,
                                "payload_bytes": 0,
                                "seed": "",
                                "elapsed_seconds": 0.0,
                            }
                        )
                        records.append(control)

                rates_for_method = {
                    "plain_lsb": plain_rates,
                    "password_lsb": uniform_rates,
                }
                for method, password in (
                    ("plain_lsb", None),
                    ("password_lsb", lsb_helpers.PASSWORD),
                ):
                    for payload_type in ("uniform_random_bytes", "printable_text"):
                        rates = rates_for_method[method] if payload_type == "uniform_random_bytes" else (100,)
                        for rate in rates:
                            seed = _payload_seed(image["image_index"], method, rate, payload_type)
                            phase_started = time.perf_counter()
                            payload = _make_lsb_payload(
                                capacity_bits=lsb_capacity,
                                rate=rate,
                                seed=seed,
                                password=password,
                                payload_type=payload_type,
                            )
                            output_path = work_dir / f"{image['image_index']}_{method}_{payload_type}_{rate}.png"
                            lsb_helpers._embed_payload_bits(
                                source_path, output_path, payload, password
                            )
                            actual = _actual_rate(payload, lsb_capacity)
                            embedded_records = _evaluate_path(
                                run_mode=run_mode,
                                row_type="stego",
                                scope="embedded_image",
                                image=image,
                                path=output_path,
                                method=method,
                                payload_type=payload_type,
                                requested_rate_pct=rate,
                                actual_rate_pct=actual,
                                capacity_bits=lsb_capacity,
                                detector_names=detector_fields[:3],
                                payload_bytes=len(payload),
                                payload_one_fraction=_payload_one_fraction(payload),
                                seed=seed,
                            )
                            records.extend(embedded_records)
                            phases[method] += time.perf_counter() - phase_started

                processed_path = work_dir / f"{image['image_index']}_dct_processed_control.png"
                control_started = time.perf_counter()
                dct_helpers._save_processed_control(source_path, processed_path)
                processed_descriptors = _describe_image(processed_path)
                raw_changed_blocks, raw_changed_fraction = (
                    dct_helpers._modified_block_fraction(processed_path, source_path)
                )
                raw_dct_record = next(
                    row for row in clean_records if row["detector"] == "dct_parity"
                )
                raw_dct_record["changed_blocks"] = raw_changed_blocks
                raw_dct_record["changed_block_fraction"] = raw_changed_fraction
                processed_records = _evaluate_path(
                    run_mode=run_mode,
                    row_type="processed_control",
                    scope="processed_control",
                    image=image,
                    path=processed_path,
                    method="dct",
                    payload_type="control",
                    requested_rate_pct=0,
                    actual_rate_pct=0.0,
                    capacity_bits=dct_capacity,
                    detector_names=("dct_parity",),
                    changed_blocks=0,
                    changed_block_fraction=0.0,
                    image_descriptors=processed_descriptors,
                )
                phases["controls"] += time.perf_counter() - control_started
                records.extend(processed_records)

                for payload_type in ("uniform_random_bytes", "printable_text"):
                    rates = uniform_rates if payload_type == "uniform_random_bytes" else (100,)
                    for rate in rates:
                        seed = _payload_seed(image["image_index"], "dct", rate, payload_type)
                        target_payload_bytes = dct_capacity * rate // 800
                        if target_payload_bytes < dct_helpers.OVERHEAD_BYTES:
                            unavailable = {field: "" for field in CSV_FIELDS}
                            unavailable.update(
                                {
                                    "run_mode": run_mode,
                                    "row_type": "unavailable",
                                    "scope": "unavailable_rate",
                                    "image_index": image["image_index"],
                                    "image": image["new_filename"],
                                    "category": image["category"],
                                    "method": "dct",
                                    "payload_type": payload_type,
                                    "requested_rate_pct": rate,
                                    "capacity_bits": dct_capacity,
                                    "seed": seed,
                                    "detector": "dct_parity",
                                    "detector_threshold": DCT_THRESHOLD,
                                    "verdict": (
                                        f"unavailable: {target_payload_bytes} target bytes "
                                        f"< {dct_helpers.OVERHEAD_BYTES}-byte framing overhead"
                                    ),
                                    "luminance_std": raw_descriptors["luminance_std"],
                                    "flat_block_fraction": raw_descriptors["flat_block_fraction"],
                                    "distinct_rgb_values": raw_descriptors["distinct_rgb_values"],
                                    "detector_result_json": json.dumps(
                                        {
                                            "status": "unavailable",
                                            "target_payload_bytes": target_payload_bytes,
                                            "framing_overhead_bytes": dct_helpers.OVERHEAD_BYTES,
                                        },
                                        sort_keys=True,
                                    ),
                                }
                            )
                            records.append(unavailable)
                            continue
                        phase_started = time.perf_counter()
                        payload = _make_dct_payload(
                            capacity_bits=dct_capacity,
                            rate=rate,
                            seed=seed,
                            payload_type=payload_type,
                        )
                        output_path = work_dir / f"{image['image_index']}_dct_{payload_type}_{rate}.png"
                        dct_helpers._embed_payload_bytes(source_path, output_path, payload)
                        actual = _actual_rate(payload, dct_capacity)
                        changed_blocks, changed_block_fraction = (
                            dct_helpers._modified_block_fraction(
                                processed_path, output_path
                            )
                        )
                        embedded_records = _evaluate_path(
                            run_mode=run_mode,
                            row_type="stego",
                            scope="embedded_image",
                            image=image,
                            path=output_path,
                            method="dct",
                            payload_type=payload_type,
                            requested_rate_pct=rate,
                            actual_rate_pct=actual,
                            capacity_bits=dct_capacity,
                            detector_names=("dct_parity",),
                            payload_bytes=len(payload),
                            payload_one_fraction=_payload_one_fraction(payload),
                            changed_blocks=changed_blocks,
                            changed_block_fraction=changed_block_fraction,
                            seed=seed,
                        )
                        records.extend(embedded_records)
                        phases["dct"] += time.perf_counter() - phase_started

                crop_started = time.perf_counter()
                with Image.open(source_path) as source:
                    rgb = source.convert("RGB")
                    for size in CROP_SIZES:
                        crop_width, crop_height = size
                        size_id = crop_size_id[size]
                        for crop_index in range(crops_per_size):
                            seed = _crop_seed(image["image_index"], size_id, crop_index)
                            if width < crop_width or height < crop_height:
                                for detector in crop_detector_fields:
                                    unavailable_crop = {field: "" for field in CSV_FIELDS}
                                    unavailable_crop.update(
                                        {
                                            "run_mode": run_mode,
                                            "row_type": "unavailable_crop",
                                            "scope": "raw_crop_unavailable",
                                            "image_index": image["image_index"],
                                            "image": image["new_filename"],
                                            "category": image["category"],
                                            "method": "raw_crop",
                                            "payload_type": "none",
                                            "requested_rate_pct": 0,
                                            "detector": detector,
                                            "detector_threshold": _threshold_for_detector(detector),
                                            "crop_width": crop_width,
                                            "crop_height": crop_height,
                                            "crop_index": crop_index,
                                            "crop_seed": seed,
                                            "verdict": (
                                                f"unavailable: cover is {width}x{height}; "
                                                f"cannot crop {crop_width}x{crop_height}"
                                            ),
                                            "detector_result_json": json.dumps(
                                                {
                                                    "status": "unavailable",
                                                    "cover_width": width,
                                                    "cover_height": height,
                                                    "crop_width": crop_width,
                                                    "crop_height": crop_height,
                                                },
                                                sort_keys=True,
                                            ),
                                        }
                                    )
                                    records.append(unavailable_crop)
                                continue
                            rng = random.Random(seed)
                            left = rng.randrange(width - crop_width + 1)
                            top = rng.randrange(height - crop_height + 1)
                            crop_path = work_dir / (
                                f"{image['image_index']}_crop_{crop_width}x{crop_height}_{crop_index}.png"
                            )
                            rgb.crop((left, top, left + crop_width, top + crop_height)).save(
                                crop_path, format="PNG"
                            )
                            crop_image = dict(image)
                            crop_image["new_filename"] = image["new_filename"]
                            crop_image["image_index"] = image["image_index"]
                            crop_descriptors = _describe_image(crop_path)
                            for detector in crop_detector_fields:
                                detector_capacity = (
                                    dct_capacity_bits(crop_width, crop_height)
                                    if detector == "dct_parity"
                                    else crop_width * crop_height * 3
                                )
                                records.extend(
                                    _evaluate_path(
                                        run_mode=run_mode,
                                        row_type="clean_crop",
                                        scope="raw_crop",
                                        image=crop_image,
                                        path=crop_path,
                                        method="raw_crop",
                                        payload_type="none",
                                        requested_rate_pct=0,
                                        actual_rate_pct=0.0,
                                        capacity_bits=detector_capacity,
                                        detector_names=(detector,),
                                        image_descriptors=crop_descriptors,
                                        crop_width=crop_width,
                                        crop_height=crop_height,
                                        crop_index=crop_index,
                                        crop_seed=seed,
                                    )
                                )
                phases["crops"] += time.perf_counter() - crop_started

                image_elapsed = time.perf_counter() - image_started
                phases["image_elapsed"] = image_elapsed
                phase_totals.append(phases)
                print(
                    f"  runtime: {image_elapsed:.3f}s "
                    f"(plain LSB {phases['plain_lsb']:.3f}s; "
                    f"password LSB {phases['password_lsb']:.3f}s; "
                    f"DCT {phases['dct']:.3f}s; controls {phases['controls']:.3f}s; "
                    f"crops {phases['crops']:.3f}s)",
                    flush=True,
                )

                # Persist this image's measurements so partial runs remain auditable.
                writer.writerows(records[image_record_start:])
                csv_stream.flush()

    total_runtime = time.perf_counter() - study_started
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    _save_plots(records)
    matplotlib_cache = OUTPUT_DIR / ".mplconfig"
    if matplotlib_cache.exists():
        shutil.rmtree(matplotlib_cache)
    if args.quick:
        estimate_seconds, phase_means = _estimated_full_seconds(
            phase_totals, len(all_images)
        )
    else:
        estimate_seconds, phase_means = total_runtime, {}
    _write_run_info(
        run_mode=run_mode,
        started_at=started_at,
        finished_at=finished_at,
        selected=selected,
        all_images=all_images,
        crop_count=crop_count,
        total_runtime=total_runtime,
        estimate_seconds=estimate_seconds,
    )
    _print_raw_measurements(records)
    _print_detection_rate_table(records)
    for category in sorted({image["category"] for image in selected}):
        _print_detection_rate_table(records, category)
    _print_false_alarm_table(records)
    _print_false_alarm_descriptor_table(records)
    _print_minimum_rates(records)
    _print_roc_table(records)
    _print_payload_one_bit_summary(records)
    _print_dct_rows(records)
    _print_crop_rows(records)
    _print_unavailable_rows(records)

    print("\nPer-image runtime")
    for image, phases in zip(selected, phase_totals):
        print(f"{image['new_filename']}: {phases['image_elapsed']:.3f}s")
    if args.quick:
        print("\nFull-run duration estimate from quick per-image timings")
        print(
            f"Estimated full run: {estimate_seconds:.3f}s "
            f"({estimate_seconds / 60:.2f} minutes; {estimate_seconds / 3600:.2f} hours)"
        )
        print(
            "Estimate scaling: mean quick phase times per image; plain LSB rows x1.75, "
            "password LSB rows x2.5, DCT rows x2, crop checks x2.5 (2 to 5 crops per size), "
            f"then x{len(all_images)} usable images. Phase means: "
            + ", ".join(f"{name}={value:.3f}s" for name, value in phase_means.items())
        )
    print(f"\nTotal runtime: {total_runtime:.3f} seconds")
    print(f"CSV: {csv_path}")
    print(f"Run info: {OUTPUT_DIR / 'run_info.txt'}")
    print(f"Plot: {OUTPUT_DIR / 'detection_rate_vs_embedding_rate.png'}")
    print(f"Plot: {OUTPUT_DIR / 'clean_cover_pvalue_distributions.png'}")


if __name__ == "__main__":
    main()
