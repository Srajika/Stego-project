"""Measure how Pillow JPEG recompression affects this project's stego payloads."""

from __future__ import annotations

import argparse
import base64
import csv
import math
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL
import scipy
import skimage
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stego.common import STOP_MARKER, TAG_BYTES, build_payload, bytes_to_bits
from stego.dct import (
    extract as dct_extract,
    hide as dct_hide,
    read_majority_bits,
)
from stego.lsb import _position_order, extract as lsb_extract, hide as lsb_hide
from stego.tamper import derive_keys, make_tag


JPEG_QUALITIES = (100, 90, 75, 70, 50, 30)
QUICK_JPEG_QUALITIES = (100, 70)
SUBSAMPLING_VARIANTS = ("4:4:4", "Pillow default")
PAYLOAD_SEED = 20261105
PAYLOAD_SOURCE_BYTES = 40
STUDY_PASSWORD = "phase5a-compression-study-fixed-password"
PASSWORD_LABEL = "phase5a-fixed-study-password"
CSV_COLUMNS = (
    "record_type",
    "image",
    "category",
    "method",
    "embed_quality",
    "jpeg_quality",
    "subsampling",
    "payload_seed",
    "payload_source_bytes",
    "payload_message_bytes",
    "payload_framed_bits",
    "extraction_status",
    "raw_bit_errors",
    "raw_ber",
    "message_matches",
    "exact_recovery",
    "stego_png_psnr",
    "stego_png_ssim",
    "stego_reconstruction_psnr",
    "stego_reconstruction_ssim",
    "cover_reconstruction_psnr",
    "cover_reconstruction_ssim",
    "baseline_excluded",
    "baseline_exclusion_reason",
    "error",
)


def _metric_pair(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """Return RGB PSNR and SSIM with the 8-bit image range fixed at 255."""
    return (
        float(peak_signal_noise_ratio(reference, candidate, data_range=255)),
        float(
            structural_similarity(
                reference,
                candidate,
                data_range=255,
                channel_axis=-1,
            )
        ),
    )


def _message_and_payload(seed: int) -> tuple[str, bytes, str]:
    """Make one reproducible text-safe message from 40 uniformly random bytes."""
    source = np.random.default_rng(seed).bytes(PAYLOAD_SOURCE_BYTES)
    message = base64.b64encode(source).decode("ascii")
    tag = make_tag(message.encode("utf-8"), STUDY_PASSWORD)
    payload = build_payload(message, tag)
    return message, payload, bytes_to_bits(payload)


def _read_lsb_payload_bits(path: Path, bit_count: int) -> str:
    """Read exactly the keyed LSB positions used by lsb.hide, without tag parsing."""
    order_key, _ = derive_keys(STUDY_PASSWORD)
    with Image.open(path) as image:
        flat = np.asarray(image.convert("RGB"), dtype=np.uint8).reshape(-1)
    if bit_count > flat.size:
        raise ValueError(f"Payload needs {bit_count} bits; LSB capacity is {flat.size}")
    positions = _position_order(order_key, flat.size)[:bit_count]
    return "".join(str(int(bit)) for bit in (flat[positions] & 1))


def _raw_bits(path: Path, method: str, embed_quality: int | None, bit_count: int) -> str:
    if method == "LSB":
        return _read_lsb_payload_bits(path, bit_count)
    assert embed_quality is not None
    return read_majority_bits(
        str(path), bit_count=bit_count, quality=embed_quality
    )


def _bit_errors(expected: str, observed: str) -> int:
    mismatches = sum(left != right for left, right in zip(expected, observed))
    return mismatches + abs(len(expected) - len(observed))


def _save_jpeg(
    source_path: Path,
    output_path: Path,
    jpeg_quality: int,
    subsampling: str,
) -> None:
    """Save RGB JPEG with explicit 4:4:4 or Pillow's default subsampling."""
    with Image.open(source_path) as image:
        save_args: dict[str, Any] = {
            "format": "JPEG",
            "quality": jpeg_quality,
            "optimize": False,
            "progressive": False,
        }
        if subsampling == "4:4:4":
            save_args["subsampling"] = 0
        elif subsampling != "Pillow default":
            raise ValueError(f"Unknown subsampling variant: {subsampling}")
        image.convert("RGB").save(output_path, **save_args)


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _make_record(**values: Any) -> dict[str, Any]:
    return {column: values.get(column, "") for column in CSV_COLUMNS}


def _recovery_fields(
    path: Path,
    method: str,
    embed_quality: int | None,
    message: str,
    expected_bits: str,
) -> dict[str, Any]:
    observed_bits = _raw_bits(path, method, embed_quality, len(expected_bits))
    errors = _bit_errors(expected_bits, observed_bits)
    if method == "LSB":
        extracted = lsb_extract(str(path), password=STUDY_PASSWORD)
    else:
        assert embed_quality is not None
        extracted = dct_extract(
            str(path), password=STUDY_PASSWORD, quality=embed_quality
        )
    status = str(extracted.get("status", "unknown"))
    message_matches = extracted.get("message") == message
    exact = status == "intact" and message_matches
    return {
        "extraction_status": status,
        "raw_bit_errors": errors,
        "raw_ber": errors / len(expected_bits),
        "message_matches": message_matches,
        "exact_recovery": exact,
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def _format_rate(k: int, n: int) -> str:
    percentage = 100.0 * k / n if n else float("nan")
    return f"{k}/{n} ({percentage:.1f}%)"


def _recovery_summary(
    records: list[dict[str, Any]],
    category: str | None = None,
) -> str:
    eligible = [
        row
        for row in records
        if row["record_type"] == "stego"
        and row["jpeg_quality"] != "none"
        and row["baseline_excluded"] is False
        and (category is None or row["category"] == category)
    ]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        groups[(row["method"], str(row["jpeg_quality"]), row["subsampling"])].append(row)
    lines = [
        "method     | JPEG q | subsampling   | exact recovery k/n (%) | mean BER (SD) | n",
        "-----------|--------|---------------|------------------------|---------------|---",
    ]
    method_order = ("LSB", "DCT-q50", "DCT-q75", "DCT-q90")
    for method in method_order:
        for quality in sorted(
            {key[1] for key in groups if key[0] == method}, key=int
        ):
            for subsampling in SUBSAMPLING_VARIANTS:
                values = groups.get((method, quality, subsampling), [])
                if not values:
                    continue
                recovered = sum(bool(row["exact_recovery"]) for row in values)
                bers = np.asarray([float(row["raw_ber"]) for row in values])
                lines.append(
                    f"{method:<10} | {quality:>6} | {subsampling:<13} | "
                    f"{_format_rate(recovered, len(values)):>22} | "
                    f"{np.mean(bers):.6f} / {np.std(bers):.6f} | {len(values)}"
                )
    if len(lines) == 2:
        lines.append("(no eligible JPEG rows)")
    return "\n".join(lines)


def _embedding_quality_summary(records: list[dict[str, Any]]) -> str:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["record_type"] == "stego" and row["jpeg_quality"] == "none":
            groups[row["method"]].append(row)
    lines = ["method     | mean PNG embedding PSNR | mean PNG embedding SSIM | n images",
             "-----------|-------------------------|-------------------------|---"]
    for method in ("LSB", "DCT-q50", "DCT-q75", "DCT-q90"):
        values = groups.get(method, [])
        if values:
            psnr = np.mean([float(row["stego_png_psnr"]) for row in values])
            ssim = np.mean([float(row["stego_png_ssim"]) for row in values])
            lines.append(f"{method:<10} | {psnr:>23.4f} | {ssim:>23.6f} | {len(values)}")
    return "\n".join(lines)


def _cover_baseline_summary(records: list[dict[str, Any]]) -> str:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["record_type"] == "cover_baseline":
            groups[(str(row["jpeg_quality"]), row["subsampling"])].append(row)
    lines = [
        "JPEG q | subsampling   | cover-only PSNR (dB) | cover-only SSIM | n",
        "-------|---------------|---------------------|-----------------|---",
    ]
    for quality in sorted({key[0] for key in groups}, key=int):
        for subsampling in SUBSAMPLING_VARIANTS:
            values = groups.get((quality, subsampling), [])
            if values:
                psnr = np.mean([float(row["cover_reconstruction_psnr"]) for row in values])
                ssim = np.mean([float(row["cover_reconstruction_ssim"]) for row in values])
                lines.append(
                    f"{quality:>6} | {subsampling:<13} | {psnr:>19.4f} | "
                    f"{ssim:>15.6f} | {len(values)}"
                )
    return "\n".join(lines)


def _save_plots(records: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    methods = ("LSB", "DCT-q50", "DCT-q75", "DCT-q90")
    colors = ("#2563eb", "#dc2626", "#059669", "#9333ea")
    for axis, subsampling in zip(axes, SUBSAMPLING_VARIANTS):
        for method, color in zip(methods, colors):
            x_values: list[int] = []
            y_values: list[float] = []
            for quality in sorted(JPEG_QUALITIES):
                values = [
                    row for row in records
                    if row["record_type"] == "stego"
                    and row["method"] == method
                    and row["jpeg_quality"] == quality
                    and row["subsampling"] == subsampling
                    and row["baseline_excluded"] is False
                ]
                x_values.append(quality)
                y_values.append(
                    100.0 * sum(bool(row["exact_recovery"]) for row in values) / len(values)
                    if values else math.nan
                )
            axis.plot(x_values, y_values, marker="o", label=method, color=color)
        axis.set_title(subsampling)
        axis.set_xlabel("JPEG save quality")
        axis.set_xticks(sorted(JPEG_QUALITIES))
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Exact recovery (%)")
    axes[1].legend(loc="best")
    fig.suptitle("Payload exact recovery after JPEG recompression")
    fig.tight_layout()
    recovery_path = output_dir / "recovery_vs_jpeg_quality.png"
    fig.savefig(recovery_path, dpi=160)
    plt.close(fig)

    method_values: dict[str, list[float]] = defaultdict(list)
    for row in records:
        if row["record_type"] == "stego" and row["jpeg_quality"] == "none":
            method_values[row["method"]].append(float(row["stego_png_psnr"]))
    qualities = (50, 75, 90)
    dct_means = [float(np.mean(method_values[f"DCT-q{quality}"])) for quality in qualities]
    lsb_mean = float(np.mean(method_values["LSB"]))
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(qualities, dct_means, marker="o", linewidth=2, label="DCT embedding quality")
    axis.axhline(lsb_mean, linestyle="--", color="#2563eb", label="LSB (no quality setting)")
    axis.set_xticks(qualities)
    axis.set_xlabel("DCT embedding quality")
    axis.set_ylabel("Mean PNG PSNR against cover (dB)")
    axis.set_title("Embedding cost by method and DCT quality")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    psnr_path = output_dir / "embedding_psnr_vs_quality.png"
    fig.savefig(psnr_path, dpi=160)
    plt.close(fig)
    return [recovery_path, psnr_path]


def _write_run_info(
    path: Path,
    started: datetime,
    finished: datetime,
    quick: bool,
    eligible_count: int,
    unusable_count: int,
    error_count: int,
) -> None:
    quality_list = QUICK_JPEG_QUALITIES if quick else JPEG_QUALITIES
    content = [
        "Phase 5A compression-survival study",
        f"Started: {started.astimezone().isoformat(timespec='seconds')}",
        f"Finished: {finished.astimezone().isoformat(timespec='seconds')}",
        f"Mode: {'quick' if quick else 'full'}",
        f"Payload RNG: numpy.random.default_rng; base seed={PAYLOAD_SEED}; "
        "per-image seed=base seed + zero-based index after sorted usable-row selection",
        f"Payload source: {PAYLOAD_SOURCE_BYTES} uniformly random bytes per image; "
        "Base64 ASCII message so the project's UTF-8 APIs preserve the source bytes",
        f"Framing: {len(STOP_MARKER)}-byte stop marker + {TAG_BYTES}-byte HMAC-SHA256 hex tag",
        f"Password label: {PASSWORD_LABEL} (password text omitted)",
        f"JPEG qualities: {', '.join(map(str, quality_list))}",
        "JPEG save settings: format=JPEG, quality=<listed quality>, optimize=False, "
        "progressive=False; 4:4:4 passes subsampling=0; Pillow default variant omits "
        "subsampling (Pillow/libjpeg RGB default, 4:2:0)",
        f"Pillow: {PIL.__version__}",
        f"NumPy: {np.__version__}",
        f"SciPy: {scipy.__version__}",
        f"scikit-image: {skimage.__version__}",
        f"Matplotlib: {matplotlib.__version__}",
        f"Usable images selected: {eligible_count}",
        f"Manifest rows skipped as unusable: {unusable_count}",
        f"Images skipped after errors: {error_count}",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def run_study(root: str | Path, quick: bool = False) -> dict[str, Any]:
    """Run the study with all generated outputs confined to root/results/compression."""
    root_path = Path(root).resolve()
    dataset_root = (root_path / "images" / "dataset").resolve()
    output_dir = (root_path / "results" / "compression").resolve()
    expected_output = (root_path / "results" / "compression").resolve()
    if output_dir != expected_output or root_path not in output_dir.parents:
        raise AssertionError(f"Refusing output path outside results/compression: {output_dir}")
    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as source:
        manifest_rows = list(csv.DictReader(source))
    usable_rows = [
        row for row in manifest_rows
        if row.get("usable_for_main_studies", "").strip().lower() == "true"
    ]
    usable_rows.sort(key=lambda row: (row.get("category", ""), row.get("new_filename", "")))
    unusable_count = len(manifest_rows) - len(usable_rows)
    if quick:
        usable_rows = usable_rows[:3]
    jpeg_qualities = QUICK_JPEG_QUALITIES if quick else JPEG_QUALITIES

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone()
    start = time.monotonic()
    records: list[dict[str, Any]] = []
    log_lines: list[str] = []

    def log(message: str = "") -> None:
        print(message, flush=True)
        log_lines.append(message)

    log(f"Compression study mode: {'quick' if quick else 'full'}")
    log(f"Usable images selected: {len(usable_rows)}; unusable manifest rows skipped: {unusable_count}")
    log(
        f"JPEG qualities: {', '.join(map(str, jpeg_qualities))}; variants: "
        "4:4:4 (subsampling=0), Pillow default (subsampling omitted)"
    )
    log(
        f"Payload per image: {PAYLOAD_SOURCE_BYTES} seeded random bytes -> "
        "Base64 UTF-8 message + 5-byte marker + 64-byte HMAC tag"
    )

    methods: tuple[tuple[str, int | None], ...] = (
        ("LSB", None),
        ("DCT-q50", 50),
        ("DCT-q75", 75),
        ("DCT-q90", 90),
    )
    errors: list[tuple[str, str]] = []

    with tempfile.TemporaryDirectory(prefix=".compression-tmp-", dir=output_dir) as temp_name:
        scratch_root = Path(temp_name)
        for image_index, manifest_row in enumerate(usable_rows):
            image_name = manifest_row["new_filename"]
            category = manifest_row["category"]
            cover_path = dataset_root / category / image_name
            image_records: list[dict[str, Any]] = []
            exclusion_by_method: dict[str, str] = {}
            image_scratch = scratch_root / f"image-{image_index:03d}"
            image_scratch.mkdir()
            try:
                cover_rgb = _read_rgb(cover_path)
                message_seed = PAYLOAD_SEED + image_index
                message, payload, expected_bits = _message_and_payload(message_seed)
                log(f"[{image_index + 1}/{len(usable_rows)}] {category}/{image_name}")

                # A cover-only JPEG is saved once per image and JPEG setting.
                cover_metrics: dict[tuple[int, str], tuple[float, float]] = {}
                for quality in jpeg_qualities:
                    for subsampling in SUBSAMPLING_VARIANTS:
                        cover_jpeg = image_scratch / f"cover-q{quality}-{subsampling.replace(':', '')}.jpg"
                        _save_jpeg(cover_path, cover_jpeg, quality, subsampling)
                        cover_psnr, cover_ssim = _metric_pair(cover_rgb, _read_rgb(cover_jpeg))
                        cover_metrics[(quality, subsampling)] = (cover_psnr, cover_ssim)
                        image_records.append(
                            _make_record(
                                record_type="cover_baseline",
                                image=image_name,
                                category=category,
                                method="cover",
                                embed_quality="none",
                                jpeg_quality=quality,
                                subsampling=subsampling,
                                cover_reconstruction_psnr=cover_psnr,
                                cover_reconstruction_ssim=cover_ssim,
                            )
                        )

                for method, embed_quality in methods:
                    method_slug = method.lower().replace("-", "_")
                    stego_png = image_scratch / f"{method_slug}.png"
                    if method == "LSB":
                        lsb_hide(
                            str(cover_path), str(stego_png), message,
                            password=STUDY_PASSWORD,
                        )
                    else:
                        assert embed_quality is not None
                        dct_hide(
                            str(cover_path), str(stego_png), message,
                            password=STUDY_PASSWORD,
                            quality=embed_quality,
                        )

                    stego_png_rgb = _read_rgb(stego_png)
                    png_psnr, png_ssim = _metric_pair(cover_rgb, stego_png_rgb)
                    baseline = _recovery_fields(
                        stego_png, method, embed_quality, message, expected_bits
                    )
                    baseline_reason = ""
                    excluded = baseline["raw_bit_errors"] > 0
                    if excluded:
                        if method == "LSB":
                            baseline_reason = "unexpected loss in lossless PNG pixel/channel representation"
                        else:
                            baseline_reason = "DCT inverse-transform integer rounding / RGB-YCbCr round trip"
                        exclusion_by_method[method] = baseline_reason
                    image_records.append(
                        _make_record(
                            record_type="stego",
                            image=image_name,
                            category=category,
                            method=method,
                            embed_quality=embed_quality if embed_quality is not None else "n/a",
                            jpeg_quality="none",
                            subsampling="none",
                            payload_seed=message_seed,
                            payload_source_bytes=PAYLOAD_SOURCE_BYTES,
                            payload_message_bytes=len(message.encode("utf-8")),
                            payload_framed_bits=len(expected_bits),
                            **baseline,
                            stego_png_psnr=png_psnr,
                            stego_png_ssim=png_ssim,
                            baseline_excluded=excluded,
                            baseline_exclusion_reason=baseline_reason,
                        )
                    )

                    for quality in jpeg_qualities:
                        for subsampling in SUBSAMPLING_VARIANTS:
                            compressed = image_scratch / (
                                f"{method_slug}-q{quality}-"
                                f"{subsampling.replace(':', '')}.jpg"
                            )
                            _save_jpeg(stego_png, compressed, quality, subsampling)
                            compressed_psnr, compressed_ssim = _metric_pair(
                                cover_rgb, _read_rgb(compressed)
                            )
                            recovery = _recovery_fields(
                                compressed, method, embed_quality, message, expected_bits
                            )
                            image_records.append(
                                _make_record(
                                    record_type="stego",
                                    image=image_name,
                                    category=category,
                                    method=method,
                                    embed_quality=embed_quality if embed_quality is not None else "n/a",
                                    jpeg_quality=quality,
                                    subsampling=subsampling,
                                    payload_seed=message_seed,
                                    payload_source_bytes=PAYLOAD_SOURCE_BYTES,
                                    payload_message_bytes=len(message.encode("utf-8")),
                                    payload_framed_bits=len(expected_bits),
                                    **recovery,
                                    stego_png_psnr=png_psnr,
                                    stego_png_ssim=png_ssim,
                                    stego_reconstruction_psnr=compressed_psnr,
                                    stego_reconstruction_ssim=compressed_ssim,
                                    baseline_excluded=excluded,
                                    baseline_exclusion_reason=baseline_reason,
                                )
                            )
                records.extend(image_records)
            except Exception as exc:  # Keep the rest of the dataset moving.
                reason = f"{type(exc).__name__}: {exc}"
                errors.append((f"{category}/{image_name}", reason))
                log(f"  SKIPPED image: {category}/{image_name} — {reason}")

    log()
    log("No-compression baseline failures (excluded from JPEG summary tables):")
    exclusions = [
        row for row in records
        if row["record_type"] == "stego"
        and row["jpeg_quality"] == "none"
        and row["baseline_excluded"]
    ]
    if exclusions:
        for row in exclusions:
            log(
                f"  {row['image']} / {row['method']}: "
                f"{row['raw_bit_errors']} bit errors; {row['baseline_exclusion_reason']}"
            )
    else:
        log("  none")

    log()
    log("Cover-only JPEG loss against the original cover:")
    log(_cover_baseline_summary(records))
    log()
    log("Exact recovery and BER across images (only no-compression-clean image/method pairs):")
    recovery_report = _recovery_summary(records)
    log(recovery_report)
    log()
    log("Embedding cost from stego PNG against the original cover:")
    cost_report = _embedding_quality_summary(records)
    log(cost_report)
    categories = sorted({row["category"] for row in records if row["record_type"] == "stego"})
    for category in categories:
        log()
        log(f"Exploratory, small n - category: {category}")
        log(_recovery_summary(records, category=category))
    log()
    if errors:
        log("Images skipped after errors:")
        for name, reason in errors:
            log(f"  {name}: {reason}")
    else:
        log("Images skipped after errors: none")
    csv_path = output_dir / "compression_results.csv"
    _write_csv(csv_path, records)
    plot_paths = _save_plots(records, output_dir)
    elapsed = time.monotonic() - start
    log(f"Total runtime: {elapsed:.3f} seconds")
    log(f"CSV: {csv_path}")
    for plot_path in plot_paths:
        log(f"Plot: {plot_path}")
    finished_at = datetime.now().astimezone()
    _write_run_info(
        output_dir / "run_info.txt",
        started_at,
        finished_at,
        quick,
        len(usable_rows),
        unusable_count,
        len(errors),
    )
    summary_path = output_dir / (
        "compression_quick_summary.txt" if quick else "compression_summary.txt"
    )
    summary_path.write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )
    return {
        "records": records,
        "errors": errors,
        "excluded": exclusions,
        "runtime_seconds": elapsed,
        "csv_path": csv_path,
        "plot_paths": plot_paths,
        "summary_path": summary_path,
        "run_info_path": output_dir / "run_info.txt",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the first 3 usable images and JPEG qualities 100 and 70",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project root (defaults to this script's project)",
    )
    args = parser.parse_args()
    run_study(args.root, quick=args.quick)


if __name__ == "__main__":
    main()
