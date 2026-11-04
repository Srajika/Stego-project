"""Small assert-based checks for Phase 1 LSB behavior."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from stego.lsb import _position_order, extract, hide
from stego.tamper import (
    STATUS_CHECKSUM_MATCHES,
    STATUS_CHECKSUM_MISMATCH,
    STATUS_INTACT,
    STATUS_NO_MESSAGE,
    STATUS_TAMPERED,
    derive_keys,
    verify,
)


def run_tests() -> None:
    """Run round-trip, key, capacity, and odd-dimension checks."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        # 37 x 29 deliberately has dimensions that are not multiples of eight.
        cover_path = root / "cover.png"
        original_pixels = np.full((29, 37, 3), 128, dtype=np.uint8)
        # 3,219 colour values leave three incomplete trailing bits.
        assert original_pixels.size % 8 != 0
        Image.fromarray(original_pixels).save(cover_path)

        plain_path = root / "plain.png"
        assert hide(str(cover_path), str(plain_path), "plain hello")["status"] == "embedded"
        result = extract(str(plain_path))
        assert result["status"] == STATUS_CHECKSUM_MATCHES
        assert result["message"] == "plain hello"

        keyed_path = root / "keyed.png"
        hide(str(cover_path), str(keyed_path), "keyed hello", password="correct horse")
        keyed_result = extract(str(keyed_path), password="correct horse")
        assert keyed_result["status"] == STATUS_INTACT
        assert keyed_result["message"] == "keyed hello"
        assert extract(str(keyed_path), password="wrong password")["status"] == STATUS_NO_MESSAGE
        # Without the key, the sequential reader should not find the shuffled payload.
        assert extract(str(keyed_path))["status"] == STATUS_NO_MESSAGE

        # A changed message checked against the original tag fails HMAC verification.
        changed_message = "Keyed hello"
        assert verify(changed_message, keyed_result["tag"], "correct horse") == STATUS_TAMPERED

        # Flipping one embedded message bit leaves the marker/tag readable but breaks HMAC.
        tampered_pixels = np.array(Image.open(keyed_path).convert("RGB"))
        tampered_flat = tampered_pixels.reshape(-1)
        order_key, _ = derive_keys("correct horse")
        first_embedded_position = _position_order(order_key, tampered_flat.size)[0]
        tampered_flat[first_embedded_position] ^= 1
        tampered_path = root / "tampered.png"
        Image.fromarray(tampered_flat.reshape(tampered_pixels.shape)).save(tampered_path)
        assert extract(str(tampered_path), password="correct horse")["status"] in {
            STATUS_TAMPERED,
            STATUS_NO_MESSAGE,
        }

        # UTF-8 payload encoding should preserve Hindi and emoji text.
        unicode_message = "नमस्ते दुनिया 🌍"
        unicode_path = root / "unicode.png"
        hide(str(cover_path), str(unicode_path), unicode_message)
        assert extract(str(unicode_path))["status"] == STATUS_CHECKSUM_MATCHES
        assert extract(str(unicode_path))["message"] == unicode_message

        # LSB replacement changes each channel by at most one intensity level.
        stego_pixels = np.array(Image.open(plain_path).convert("RGB"), dtype=int)
        original_ints = original_pixels.astype(int)
        assert np.abs(original_ints - stego_pixels).max() <= 1

        # An empty message is valid: extraction returns an empty string.
        empty_path = root / "empty.png"
        hide(str(cover_path), str(empty_path), "")
        assert extract(str(empty_path))["status"] == STATUS_CHECKSUM_MATCHES
        assert extract(str(empty_path))["message"] == ""

        # A checksum can detect accidental changes but cannot authenticate content.
        plain_tag = extract(str(plain_path))["tag"]
        assert verify("plain hellO", plain_tag) == STATUS_CHECKSUM_MISMATCH

        tiny_path = root / "tiny.png"
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(tiny_path)
        try:
            hide(str(tiny_path), str(root / "too_large.png"), "too much data")
        except ValueError as error:
            assert "too large" in str(error)
        else:
            raise AssertionError("Oversized message should raise ValueError")

        try:
            hide(str(cover_path), str(root / "marker.png"), "contains ##### marker")
        except ValueError as error:
            assert "stop marker" in str(error)
        else:
            raise AssertionError("Message containing marker should raise ValueError")

        # Use the project image without modifying it; keep generated PNGs temporary.
        photo_path = Path(__file__).resolve().parent.parent / "images" / "small.png"
        assert photo_path.is_file(), f"Required real photo is missing: {photo_path}"

        photo_plain_path = root / "real_photo_plain.png"
        hide(str(photo_path), str(photo_plain_path), "real photo test")
        assert extract(str(photo_plain_path))["status"] == STATUS_CHECKSUM_MATCHES
        assert extract(str(photo_plain_path))["message"] == "real photo test"

        photo_keyed_path = root / "real_photo_keyed.png"
        hide(str(photo_path), str(photo_keyed_path), "real photo test", password="photo password")
        photo_result = extract(str(photo_keyed_path), password="photo password")
        assert photo_result["status"] == STATUS_INTACT
        assert photo_result["message"] == "real photo test"

        photo_original = np.array(Image.open(photo_path).convert("RGB"), dtype=int)
        photo_stego = np.array(Image.open(photo_keyed_path).convert("RGB"), dtype=int)
        assert np.abs(photo_original - photo_stego).max() <= 1


if __name__ == "__main__":
    run_tests()
    print("Phase 1 LSB tests passed")
