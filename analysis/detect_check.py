"""Measure LSB chi-square results by payload size, content, and embedding mode."""

import random
import string
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from stego.common import STOP_MARKER, TAG_BYTES, build_payload
from stego.detect import detect, detect_prefix_curve, detect_sequential
from stego.lsb import _position_order, hide
from stego.tamper import derive_keys, make_tag


PRINTABLE_RATES = (1, 5, 10, 25, 50, 100)
UNIFORM_BYTE_RATES = (1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
PAYLOAD_OVERHEAD_BYTES = len(STOP_MARKER) + TAG_BYTES
PRINTABLE = string.ascii_letters + string.digits + string.punctuation + " "
RANDOM_SEED = 20261104
THRESHOLD = 0.01
PASSWORD = "analysis password"


def _target_payload_bytes(capacity_bits: int, rate: int) -> int:
    """Return the byte-aligned payload size for a requested capacity rate."""
    return min(int(capacity_bits * rate // 800), capacity_bits // 8)


def _message_for_rate(
    capacity_bits: int, rate: int, rng: random.Random
) -> str:
    """Return printable ASCII text sized including marker and tag overhead."""
    payload_bytes = _target_payload_bytes(capacity_bits, rate)
    if payload_bytes < PAYLOAD_OVERHEAD_BYTES:
        raise ValueError(
            f"{rate}% target is smaller than the {PAYLOAD_OVERHEAD_BYTES}-byte overhead"
        )
    message_size = payload_bytes - PAYLOAD_OVERHEAD_BYTES
    message = "".join(rng.choices(PRINTABLE, k=message_size))
    while STOP_MARKER.decode("ascii") in message:
        message = "".join(rng.choices(PRINTABLE, k=message_size))
    return message


def _framed_text(message: str, password: str | None) -> bytes:
    """Build the same message, stop-marker, and tag bytes used by lsb.hide."""
    message_bytes = message.encode("utf-8")
    if password is None:
        hmac_key = None
    else:
        _, hmac_key = derive_keys(password)
    tag = make_tag(message_bytes, password, hmac_key)
    return build_payload(message, tag)


def _random_byte_payload(
    capacity_bits: int, rate: int, seed: int, password: str | None
) -> bytes:
    """Return uniformly random payload bytes followed by the usual framing."""
    payload_bytes = _target_payload_bytes(capacity_bits, rate)
    if payload_bytes < PAYLOAD_OVERHEAD_BYTES:
        raise ValueError(
            f"{rate}% target is smaller than the {PAYLOAD_OVERHEAD_BYTES}-byte overhead"
        )
    random_message = np.random.default_rng(seed).bytes(
        payload_bytes - PAYLOAD_OVERHEAD_BYTES
    )
    if password is None:
        hmac_key = None
    else:
        _, hmac_key = derive_keys(password)
    tag = make_tag(random_message, password, hmac_key)
    return random_message + STOP_MARKER + tag.encode("ascii")


def _embed_payload_bits(
    cover_path: Path,
    output_path: Path,
    payload: bytes,
    password: str | None,
) -> float:
    """Write raw payload bits using the same channel assignment as lsb.hide.

    ``lsb.hide`` accepts text and UTF-8 encodes it. This helper writes arbitrary
    byte payloads directly: NumPy unpacks bytes most-significant-bit first,
    then each selected channel is assigned ``(channel & 0xFE) | bit``. Plain
    mode selects channels sequentially; password mode uses the existing
    ``_position_order`` derived from ``derive_keys``.
    """
    payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    with Image.open(cover_path) as image:
        pixels = np.array(image.convert("RGB"), dtype=np.uint8)
    flat = pixels.reshape(-1).copy()
    if len(payload_bits) > flat.size:
        raise ValueError("Raw payload exceeds image capacity")

    if password is None:
        positions = np.arange(flat.size)
    else:
        order_key, _ = derive_keys(password)
        positions = _position_order(order_key, flat.size)
    selected = positions[:len(payload_bits)]
    flat[selected] = (flat[selected] & np.uint8(254)) | payload_bits
    Image.fromarray(flat.reshape(pixels.shape)).save(output_path, format="PNG")
    return float(np.mean(payload_bits))


def _format_p_value(p_value: float) -> str:
    """Print extremely small p-values without rounding them to zero."""
    if p_value < 1e-300:
        return "< 1e-300"
    return f"{p_value:.12g}"


def _print_row(
    payload_type: str,
    requested_rate: int,
    actual_rate: float,
    result: dict,
    sequential_verdict: str,
    one_fraction: float | None,
) -> None:
    """Print all requested measurements for one analysis row."""
    one_bits = "n/a" if one_fraction is None else f"{one_fraction:.6f}"
    print(
        f"{payload_type:<18} | {requested_rate:>4}% | {actual_rate:>10.4f}% | "
        f"{result['statistic']:>11.6g} | {_format_p_value(result['p_value']):>12} | "
        f"{result['verdict']:<29} | {sequential_verdict:<39} | {one_bits:>10}"
    )


def _run_mode(
    cover_path: Path,
    output_dir: Path,
    mode: str,
    capacity_bits: int,
) -> dict[str, Path]:
    """Print one mode's payload table and return plain-mode curve images."""
    password = PASSWORD if mode == "Password-mode" else None
    print(f"\n{mode} LSB (threshold={THRESHOLD})")
    print(
        "Payload type        | Requested | Actual rate | Statistic | "
        "p-value      | Verdict                       | Sequential verdict (plain only)       | 1-bit fraction"
    )
    print(
        "--------------------|-----------|-------------|-----------|"
        "--------------|-------------------------------|---------------------------------------|---------------"
    )
    clean_result = detect(cover_path, threshold=THRESHOLD)
    clean_sequential = detect_sequential(cover_path, threshold=THRESHOLD)
    _print_row(
        "clean",
        0,
        0.0,
        clean_result,
        clean_sequential["verdict"]
        if mode == "Plain-mode"
        else "n/a (password mode)",
        None,
    )

    curve_paths: dict[str, Path] = {}
    text_rng = random.Random(RANDOM_SEED)
    mode_offset = 0 if password is None else 100000

    for rate in PRINTABLE_RATES:
        message = _message_for_rate(capacity_bits, rate, text_rng)
        text_payload = _framed_text(message, password)
        text_path = output_dir / f"{mode.lower()}_printable_{rate}.png"
        hide(str(cover_path), str(text_path), message, password=password)
        text_result = detect(text_path, threshold=THRESHOLD)
        text_sequential = (
            detect_sequential(text_path, threshold=THRESHOLD)
            if mode == "Plain-mode"
            else None
        )
        text_actual_rate = len(text_payload) * 8 / capacity_bits * 100.0
        text_one_fraction = float(
            np.unpackbits(np.frombuffer(text_payload, dtype=np.uint8)).mean()
        )
        _print_row(
            "printable text",
            rate,
            text_actual_rate,
            text_result,
            text_sequential["verdict"]
            if text_sequential is not None
            else "n/a (password mode)",
            text_one_fraction,
        )
        if mode == "Plain-mode" and rate == 50:
            curve_paths["printable text"] = text_path

    for rate in UNIFORM_BYTE_RATES:
        seed = RANDOM_SEED + rate + mode_offset
        binary_payload = _random_byte_payload(capacity_bits, rate, seed, password)
        binary_path = output_dir / f"{mode.lower()}_random_bytes_{rate}.png"
        one_fraction = _embed_payload_bits(
            cover_path, binary_path, binary_payload, password
        )
        binary_result = detect(binary_path, threshold=THRESHOLD)
        binary_sequential = (
            detect_sequential(binary_path, threshold=THRESHOLD)
            if mode == "Plain-mode"
            else None
        )
        actual_rate = len(binary_payload) * 8 / capacity_bits * 100.0
        _print_row(
            "uniform bytes",
            rate,
            actual_rate,
            binary_result,
            binary_sequential["verdict"]
            if binary_sequential is not None
            else "n/a (password mode)",
            one_fraction,
        )
        if mode == "Plain-mode" and rate == 50:
            curve_paths["uniform bytes"] = binary_path

    return curve_paths


def _print_prefix_curve(label: str, path: Path) -> None:
    """Print the detector's sequential prefix curve for one plain-mode image."""
    result = detect_prefix_curve(path, steps=20)
    print(f"\nPlain-mode 50% sequential prefix curve: {label}")
    print("Prefix rate | Values used | Statistic | p-value      | Pairs used")
    print("------------|-------------|-----------|--------------|-----------")
    for point in result["curve"]:
        rate = point["values_used"] / result["values_total"] * 100.0
        print(
            f"{rate:>10.2f}% | {point['values_used']:>11} | "
            f"{point['statistic']:>9.6g} | "
            f"{_format_p_value(point['p_value']):>12} | {point['pairs_used']:>10}"
        )


def main() -> None:
    """Run both modes and both payload types using temporary output images."""
    project_root = Path(__file__).resolve().parent.parent
    cover_path = project_root / "images" / "small.png"
    if not cover_path.is_file():
        raise FileNotFoundError(f"Required test image is missing: {cover_path}")

    with Image.open(cover_path) as image:
        width, height = image.size
    capacity_bits = width * height * 3
    print(f"Cover: images/small.png ({width}x{height})")
    print(f"LSB capacity: {capacity_bits} channel bits")
    print(
        "Each non-clean payload includes a 5-byte stop marker and 64-byte tag "
        f"({PAYLOAD_OVERHEAD_BYTES}-byte overhead); actual rate includes both."
    )
    print(f"Threshold: {THRESHOLD}; printable seed: {RANDOM_SEED}")
    print(
        "Uniform-byte path: random bytes are framed with the same marker/tag, "
        "unpacked MSB-first, then written as (channel & 0xFE) | bit. Password "
        "mode uses the existing derived-key _position_order; stego/lsb.py is unchanged."
    )
    print(
        "Sequential verdicts are computed only for plain-mode rows; password-mode "
        "rows show n/a because their payload positions are shuffled."
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        printable_curve = _run_mode(
            cover_path, output_dir, "Plain-mode", capacity_bits
        )
        _run_mode(cover_path, output_dir, "Password-mode", capacity_bits)
        _print_prefix_curve("printable text", printable_curve["printable text"])
        _print_prefix_curve("uniform bytes", printable_curve["uniform bytes"])


if __name__ == "__main__":
    main()
