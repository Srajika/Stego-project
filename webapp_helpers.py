"""Small image, capacity, and temporary-file helpers for the local demo."""

from __future__ import annotations

import io
import math
import secrets
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.stats import chi2
from skimage.metrics import structural_similarity

from stego.dct import REPETITION, capacity_bits as dct_capacity_bits

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 4000
PAYLOAD_OVERHEAD_BYTES = 69

METHODS = {
    "lsb_plain": {
        "name": "LSB plain",
        "kind": "lsb",
        "quality": None,
        "password_mode": "none",
        "description": "Writes message bits in order to RGB channel LSBs.",
    },
    "lsb_password": {
        "name": "LSB password-shuffled",
        "kind": "lsb",
        "quality": None,
        "password_mode": "required",
        "description": "Uses a password to shuffle RGB channel positions and authenticate the message.",
    },
    "dct_robust": {
        "name": "DCT robust (quality 50)",
        "kind": "dct",
        "quality": 50,
        "password_mode": "optional",
        "description": "Embeds repeated bits in image blocks; quality 50 is more robust to JPEG compression.",
    },
    "dct_balanced": {
        "name": "DCT balanced (quality 75)",
        "kind": "dct",
        "quality": 75,
        "password_mode": "optional",
        "description": "Embeds repeated bits in image blocks with a balance of robustness and visual quality.",
    },
}

STATUS_LABELS = {
    "intact": "INTACT",
    "tampered or corrupted": "TAMPERED OR CORRUPTED",
    "no message found": "NO MESSAGE FOUND",
    "checksum matches": "CHECKSUM MATCHES",
    "checksum mismatch": "CHECKSUM MISMATCH",
}


def method_items() -> list[tuple[str, dict]]:
    """Return methods in their display order for the templates."""
    return list(METHODS.items())


def get_method(key: str) -> dict:
    """Return one method description or explain an invalid selection."""
    try:
        return METHODS[key]
    except KeyError as error:
        raise ValueError("Choose one of the listed embedding methods.") from error


def method_capacity(width: int, height: int, method_key: str) -> dict[str, int]:
    """Return raw payload capacity and maximum message bytes after framing."""
    method = get_method(method_key)
    bits = (
        width * height * 3
        if method["kind"] == "lsb"
        else dct_capacity_bits(width, height)
    )
    maximum = max(0, bits // 8 - PAYLOAD_OVERHEAD_BYTES)
    return {"capacity_bits": bits, "max_message_bytes": maximum}


def random_path(temp_root: Path, suffix: str) -> Path:
    """Make an unpredictable path inside the configured temporary folder."""
    return temp_root / f"{secrets.token_hex(16)}{suffix}"


def save_verified_upload(upload, temp_root: Path) -> tuple[Path, tuple[int, int]]:
    """Verify a bounded PNG/JPEG upload with Pillow, then save it temporarily."""
    if upload is None:
        raise ValueError("Choose a PNG, JPG, or JPEG image first.")
    filename = upload.filename or ""
    extension = Path(filename).suffix.lower()
    expected = "PNG" if extension == ".png" else "JPEG"
    if extension not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Choose a PNG, JPG, or JPEG image.")

    data = upload.stream.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("The image must be 10 MB or smaller.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            actual_format = image.format
            size = image.size
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("That file is not a valid PNG or JPEG image.") from error

    if actual_format != expected:
        raise ValueError("The file extension does not match the image contents.")
    width, height = size
    if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
        raise ValueError("Images cannot be wider or taller than 4000 pixels.")

    suffix = ".png" if actual_format == "PNG" else ".jpg"
    path = random_path(temp_root, suffix)
    path.write_bytes(data)
    return path, size


def remove_file(path: Path | None) -> None:
    """Remove one temporary file without failing a request during cleanup."""
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def expire_file_later(path: Path, delay_seconds: int = 3600) -> None:
    """Schedule removal of a one-time download even if the app gets no traffic."""
    timer = threading.Timer(delay_seconds, remove_file, args=(path,))
    timer.daemon = True
    timer.start()


def cleanup_old_files(temp_root: Path, max_age_seconds: int = 3600) -> None:
    """Remove temporary files older than the configured one-hour lifetime."""
    cutoff = time.time() - max_age_seconds
    if not temp_root.exists():
        return
    for path in temp_root.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            continue


def save_jpeg_copy(source: Path, temp_root: Path, quality: int) -> Path:
    """Save an RGB image as JPEG with Pillow's default 4:2:0 subsampling."""
    output = random_path(temp_root, ".jpg")
    with Image.open(source) as image:
        image.convert("RGB").save(
            output, format="JPEG", quality=quality, subsampling=2
        )
    return output


def save_tampered_copy(source: Path, temp_root: Path, count: int) -> tuple[Path, int]:
    """Flip one random RGB channel LSB in each of N distinct random pixels."""
    with Image.open(source) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    flat = pixels.reshape(-1, 3)
    actual_count = min(max(count, 0), len(flat))
    if actual_count:
        rng = np.random.default_rng(secrets.randbits(64))
        positions = rng.choice(len(flat), size=actual_count, replace=False)
        channels = rng.integers(0, 3, size=actual_count)
        flat[positions, channels] ^= np.uint8(1)
    output = random_path(temp_root, ".png")
    Image.fromarray(flat.reshape(pixels.shape)).save(output, format="PNG")
    return output, actual_count


def quality_metrics(cover_path: Path, stego_path: Path) -> dict[str, float | int]:
    """Measure PSNR, SSIM, and largest RGB change for equal-sized images."""
    with Image.open(cover_path) as image:
        cover = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(stego_path) as image:
        stego = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if cover.shape != stego.shape:
        raise ValueError("The output image dimensions do not match the cover.")

    delta = stego.astype(np.int16) - cover.astype(np.int16)
    mse = float(np.mean(delta.astype(np.float64) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * math.log10((255.0**2) / mse)
    minimum_side = min(cover.shape[:2])
    if minimum_side >= 3:
        window = min(7, minimum_side)
        if window % 2 == 0:
            window -= 1
        ssim = float(
            structural_similarity(
                cover, stego, channel_axis=2, data_range=255, win_size=window
            )
        )
    else:
        # Tiny images cannot hold SSIM's minimum 3x3 window; use its global form.
        first = cover.astype(np.float64)
        second = stego.astype(np.float64)
        mean_first, mean_second = float(first.mean()), float(second.mean())
        var_first = float(np.mean((first - mean_first) ** 2))
        var_second = float(np.mean((second - mean_second) ** 2))
        covariance = float(np.mean((first - mean_first) * (second - mean_second)))
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        ssim = ((2 * mean_first * mean_second + c1) * (2 * covariance + c2)) / (
            (mean_first**2 + mean_second**2 + c1) * (var_first + var_second + c2)
        )
    return {
        "psnr": psnr,
        "ssim": ssim,
        "max_pixel_change": int(np.max(np.abs(delta))),
    }


def copy_corrected_p_value(statistic: float) -> float:
    """Count each of the DCT embedder's repeated copies once in its p-value."""
    if not math.isfinite(statistic):
        return float("nan")
    return float(chi2.sf(statistic / REPETITION, 1))
