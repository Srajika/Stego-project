"""Assert-based tests for DCT payload embedding and extraction."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from stego.common import build_payload, bytes_to_bits
from stego.dct import EMBED_QUALITY, extract, hide, read_majority_bits
from stego.tamper import (
    STATUS_CHECKSUM_MISMATCH,
    STATUS_CHECKSUM_MATCHES,
    STATUS_INTACT,
    STATUS_TAMPERED,
    make_tag,
)


def _assert_exact_payload(path: Path, message: str, password: str | None) -> None:
    """Assert every majority-voted payload bit matches the embedded payload."""
    tag = make_tag(message.encode("utf-8"), password)
    expected = bytes_to_bits(build_payload(message, tag))
    observed = read_majority_bits(str(path), bit_count=len(expected), quality=EMBED_QUALITY)
    assert observed == expected


def _raw_payload_errors(path: Path, message: str, password: str | None) -> tuple[int, int]:
    """Return the number of raw majority-bit errors and the payload bit count."""
    tag = make_tag(message.encode("utf-8"), password)
    expected = bytes_to_bits(build_payload(message, tag))
    observed = read_majority_bits(str(path), bit_count=len(expected), quality=EMBED_QUALITY)
    return sum(left != right for left, right in zip(expected, observed)), len(expected)


def run_tests() -> None:
    """Exercise DCT round trips, edge handling, capacity, and UTF-8 messages."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        height, width = 421, 419
        rows, columns = np.indices((height, width))
        plane = ((rows * 3 + columns * 5) % 100 + 70).astype(np.uint8)
        pixels = np.stack((plane, plane, plane), axis=2)
        cover_path = root / "cover.png"
        Image.fromarray(pixels).save(cover_path)

        # The odd-sized image has 50 x 50 complete blocks and leaves edge pixels.
        plain_path = root / "plain.png"
        assert hide(str(cover_path), str(plain_path), "plain DCT")["status"] == "embedded"
        plain_result = extract(str(plain_path))
        assert plain_result["status"] == STATUS_CHECKSUM_MATCHES
        assert plain_result["message"] == "plain DCT"
        _assert_exact_payload(plain_path, "plain DCT", None)

        keyed_path = root / "keyed.png"
        hide(str(cover_path), str(keyed_path), "keyed DCT", password="correct horse")
        keyed_result = extract(str(keyed_path), password="correct horse")
        assert keyed_result["status"] == STATUS_INTACT
        assert keyed_result["message"] == "keyed DCT"
        _assert_exact_payload(keyed_path, "keyed DCT", "correct horse")

        # Sequential DCT locations reveal the message even with a wrong password;
        # the failed HMAC reports tampering instead of concealing the message.
        wrong_password_result = extract(str(keyed_path), password="wrong password")
        assert wrong_password_result["status"] == STATUS_TAMPERED
        assert wrong_password_result["message"] == keyed_result["message"]

        # With sequential DCT placement, omitting the password still reveals
        # the message; only the HMAC check fails because its key is unavailable.
        no_password_result = extract(str(keyed_path))
        assert no_password_result["status"] == STATUS_CHECKSUM_MISMATCH
        assert no_password_result["message"] == keyed_result["message"]

        # DCT uses complete blocks only. Incomplete bottom/right edge strips are copied.
        original = np.array(Image.open(cover_path).convert("RGB"))
        embedded = np.array(Image.open(plain_path).convert("RGB"))
        assert np.array_equal(original[416:, :, :], embedded[416:, :, :])
        assert np.array_equal(original[:, 416:, :], embedded[:, 416:, :])

        unicode_message = "नमस्ते दुनिया 🌍"
        unicode_path = root / "unicode.png"
        hide(str(cover_path), str(unicode_path), unicode_message)
        assert extract(str(unicode_path))["message"] == unicode_message
        _assert_exact_payload(unicode_path, unicode_message, None)

        empty_path = root / "empty.png"
        hide(str(cover_path), str(empty_path), "")
        assert extract(str(empty_path))["status"] == STATUS_CHECKSUM_MATCHES
        assert extract(str(empty_path))["message"] == ""
        _assert_exact_payload(empty_path, "", None)

        tiny_path = root / "tiny.png"
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(tiny_path)
        try:
            hide(str(tiny_path), str(root / "too_large.png"), "this payload is too large")
        except ValueError as error:
            assert "capacity is" in str(error)
        else:
            raise AssertionError("Oversized DCT message should raise ValueError")

        # Uniform extremes and a bright-to-dark gradient test clipping headroom.
        gradient_cover = np.empty((400, 400, 3), dtype=np.uint8)
        gradient = np.tile(np.linspace(0, 255, 400, dtype=np.uint8), (400, 1))
        gradient_cover[:, :, :] = gradient[:, :, None]
        gradient_cover[:, :300, :] = 255  # Large white region beside a gradient.
        exact_cases = {
            "dark": np.zeros((400, 400, 3), dtype=np.uint8),
            "bright": np.full((400, 400, 3), 255, dtype=np.uint8),
            "white area plus gradient": gradient_cover,
        }
        extreme_errors: dict[str, int] = {}
        edge_message = "edge test"
        for label, cover_pixels in exact_cases.items():
            extreme_cover = root / f"{label}_cover.png"
            extreme_stego = root / f"{label}_stego.png"
            Image.fromarray(cover_pixels).save(extreme_cover)
            hide(str(extreme_cover), str(extreme_stego), edge_message)
            errors, bit_count = _raw_payload_errors(extreme_stego, edge_message, None)
            extreme_errors[label] = errors
            assert errors == 0, f"{label} DCT had {errors} raw bit errors out of {bit_count}"
            assert extract(str(extreme_stego))["status"] == STATUS_CHECKSUM_MATCHES
        print(
            "Headroom-image DCT raw bit errors: "
            + ", ".join(f"{name}={count}/624" for name, count in extreme_errors.items())
        )

        photo_path = Path(__file__).resolve().parent.parent / "images" / "small.png"
        assert photo_path.is_file(), f"Required real photo is missing: {photo_path}"

        photo_long_message = "long photo payload " + "x" * 128
        photo_long = root / "photo_long.png"
        hide(str(photo_path), str(photo_long), photo_long_message)
        _assert_exact_payload(photo_long, photo_long_message, None)
        photo_long_result = extract(str(photo_long))
        assert photo_long_result["status"] == STATUS_CHECKSUM_MATCHES
        assert photo_long_result["message"] == photo_long_message
        long_errors, long_payload_bits = _raw_payload_errors(
            photo_long, photo_long_message, None
        )
        assert long_errors == 0

        photo_short_message = "photo DCT"
        photo_short = root / "photo_short.png"
        hide(str(photo_path), str(photo_short), photo_short_message)
        photo_short_result = extract(str(photo_short))
        assert photo_short_result["status"] == STATUS_CHECKSUM_MATCHES
        assert photo_short_result["message"] == photo_short_message
        _assert_exact_payload(photo_short, photo_short_message, None)

        # Report image-quality cost for the longer real-photo payload.
        photo_original_pixels = np.array(Image.open(photo_path).convert("RGB"), dtype=np.int16)
        photo_stego_pixels = np.array(Image.open(photo_long).convert("RGB"), dtype=np.int16)
        pixel_difference = photo_stego_pixels.astype(np.int32) - photo_original_pixels.astype(np.int32)
        mse = float(np.mean(pixel_difference.astype(np.float64) ** 2))
        psnr = float("inf") if mse == 0 else 10 * np.log10((255.0**2) / mse)
        maximum_change = int(np.max(np.abs(pixel_difference)))
        print(
            f"Real-photo long-message raw errors: {long_errors}/{long_payload_bits}; "
            f"PSNR={psnr:.4f} dB; maximum absolute RGB pixel change={maximum_change}"
        )

        photo_plain = root / "photo_plain.png"
        photo_keyed = root / "photo_keyed.png"
        for label, message in (("short", "real photo test"), ("long", photo_long_message)):
            plain_output = root / f"photo_plain_{label}.png"
            hide(str(photo_path), str(plain_output), message)
            plain_result = extract(str(plain_output))
            assert plain_result["status"] == STATUS_CHECKSUM_MATCHES
            assert plain_result["message"] == message
            _assert_exact_payload(plain_output, message, None)

            keyed_output = root / f"photo_keyed_{label}.png"
            hide(str(photo_path), str(keyed_output), message, password="photo password")
            keyed_result = extract(str(keyed_output), password="photo password")
            assert keyed_result["status"] == STATUS_INTACT
            assert keyed_result["message"] == message
            _assert_exact_payload(keyed_output, message, "photo password")


if __name__ == "__main__":
    run_tests()
    print("Phase 3 DCT tests passed")
